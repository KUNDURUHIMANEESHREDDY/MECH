"""Runtime capability restrictions for the plugin worker.

The AST lint in :mod:`plugin_sandbox` is static: it can be read, and anything it
misses is decided before a single line runs. This module is the dynamic layer --
an audit hook, so the decision happens as the interpreter performs the
operation, below whatever the plugin's source text was able to express.

Scope, chosen deliberately. This enforces the capabilities that are cheap to
bound and expensive to get wrong:

* no network (sockets, regardless of how they were reached)
* no process creation (subprocess, exec, spawn, os.system)
* no reads outside an explicit allow-list
* no writes outside the plugin's own directory and the temp dir

It is explicitly **not** a full sandbox. A determined attacker with a working
interpreter can still find gaps; the OS limits in :mod:`limits` bound the blast
radius regardless. What this buys is that the obvious post-exploitation steps --
exfiltrate over the network, spawn a shell, read a credential, write to a config
file -- fail even if the AST gate is bypassed outright.

Reads used to be unrestricted
-----------------------------
The hook gated writes and let reads through:

    if flags & _WRITE_FLAGS and not _within(target):
        raise CapabilityViolation(...)

`writable_roots` was the only root list, so a plugin could read anything the
host user could. It needed no network escape to do it, because the plugin IPC
channel is already authorised: `Path("~/.ssh/id_rsa").read_text()` returns into
a dict the host hands back to the caller. The documented position was
"network / processes / writes / reads" as four rows, and reads were the one
marked no.

The fix is an allow-list, and the interesting part is what goes on it. Reads are
granted for:

* the plugin's own directory,
* the system temp directory,
* the Python installation and site-packages,
* the model caches (plugins legitimately load weights),
* the interpreter's own import roots, so a plugin can still import what it
  already imported -- ``backend/plugins/library/ioi_experiment_logger`` needs
  ``backend.plugins.plugin_base``, and excluding it would break a bundled
  plugin,
* any explicitly granted readable root.

Everything else is refused, including ``~/.ssh``, ``~/.config``, browser
profiles, other projects on the machine, and the host's own credential files.

Residual risk, stated plainly: because the import roots are granted, a
third-party plugin can read MECH's own source. That is a real confidentiality
cost and it is the price of letting bundled plugins import from the package.
Denying it means either vendoring the plugin API into a separate importable
package or dropping bundled plugins' ability to use it; both are larger
changes than this fix, and neither can be validated on a machine that is
currently not allowed to run the test suite. The host's *data* -- the
database, results, artifacts -- is outside the granted roots and is not exposed.

Audit hooks cannot be removed once installed, which is the property we want:
plugin code cannot unhook itself.
"""
from __future__ import annotations

import os
import site
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Iterable, Set

# Audit events that mean "this is about to touch the network".
_NETWORK_EVENTS = (
    "socket.__new__",
    "socket.bind",
    "socket.connect",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.sethostname",
    "socket.sendto",
)

# Audit events that mean "this is about to start another program".
#
# These are **audit event names, not Python function names**, and the hook matches
# them by exact equality (`if event in process_events`). The distinction is not
# cosmetic, and the most important case is `os.exec`:
#
#     os.execv(...)   raises the audit event "os.exec"
#     os.execl(...)   raises the audit event "os.exec"
#     os.execve(...)  raises the audit event "os.exec"
#
# One entry covers every exec variant. Verified on this host with a spy audit
# hook: `os.execv` produced exactly `['os.exec']`. So "correcting" this entry to
# `os.execv` -- which looks like a typo fix -- would silently stop blocking every
# exec variant. `test_process_denylist_covers_the_platform_spawn_apis` is what
# catches that.
#
# Two further notes on the entries:
#
# * `os.startfile` is the Windows-native way to launch a program, and it does
#   raise a matching audit event (plus `os.startfile/2`), so it is genuinely
#   blocked here. It was previously unverified on any platform, on a platform
#   where it is the most obvious escape.
# * `os.fork`, `os.forkpty`, `os.posix_spawn` and `pty.spawn` cannot be exercised
#   on Windows because those APIs do not exist there, so their coverage rests on
#   the POSIX run. `os.fork1` could not be confirmed to correspond to any audit
#   event CPython raises; it is harmless to keep and is listed so the policy is
#   explicit rather than accidental.
# `os.fork1` could not be confirmed to correspond to any audit event CPython
# raises; it is harmless to keep and is listed so the policy is explicit rather
# than accidental.
#
# One measured gap, deliberately left to the static layer rather than papered
# over. On Windows `subprocess.Popen` raises a second, private event:
#
#     subprocess.Popen  ->  ['subprocess.Popen', '_winapi.CreateProcess']
#
# `_winapi.CreateProcess` is not on this list, and the hook does not block it --
# calling it under an installed hook reaches the OS layer rather than raising
# CapabilityViolation. So a plugin that could reach `_winapi` could spawn a
# process through the back door. It cannot: `PluginSandbox.analyze_ast` rejects
# `import _winapi` before any code runs ("Importing module '_winapi' is forbidden
# in sandbox"), which is the defence-in-depth split this module's docstring
# describes -- the audit hook is the dynamic layer, the AST scan the static one,
# and neither is redundant.
#
# That makes the import ban load-bearing, so it is asserted rather than assumed.
# `test_the_private_winapi_escape_is_blocked_by_the_static_gate` checks both
# halves, and fails deliberately if the hook ever starts covering it.
_PROCESS_EVENTS = (
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "os.startfile",
    "os.fork1",
    "pty.spawn",
)

# Flags that mean "this open will modify the file".
#
# Everything else is a read, and reads are governed by the readable allow-list
# rather than by being unrestricted. A plugin legitimately reads its own data
# files and the model caches for weights; it has no need for the host's SSH
# keys, browser profiles or other projects, and those are what an exfiltration
# through the (already-authorised) plugin IPC channel would reach for.
_WRITE_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC | os.O_EXCL
)

# Directory enumeration is a read. It does not raise `open`, so it needs its
# own entry or a plugin could map the shape of a directory tree it may not
# open -- enough to learn where credentials and other projects live.
_LIST_DIR_EVENTS = ("os.listdir", "os.scandir")

#: Mutations that carry a *second* path: the destination. Validated from the
#: real audit-event signatures rather than assumed, because the argument order
#: is not uniform across the family:
#:
#:   os.rename(src, dst, src_dir_fd, dst_dir_fd) -> args (src, dst, -1, -1)
#:   os.replace(...)                            -> raises the os.rename event
#:   os.link(src, dst, src_dir_fd, dst_dir_fd)  -> args (src, dst, -1, -1)
#:   os.symlink(src, dst, dir_fd)               -> args (src, dst, -1)
#:
#: Checking `args[0]` alone authorized only the *source*. Measured impact with
#: the destination confirmed as the sole hole:
#:
#:   * `os.link`   -> SUCCEEDED outside the roots. The link's location (the
#:                    write) was never checked; args[0] is the file being
#:                    linked, which is usually already inside the plugin dir.
#:   * `os.rename` / `os.replace` -> already blocked, because for a move the
#:                    *source* is the path leaving the sandbox, and args[0]
#:                    happened to cover it. Now checked explicitly at both
#:                    ends rather than relying on that coincidence.
#:
#: `shutil.move` funnels through `os.rename`, so it inherited the same
#: source-only check.
_TWO_PATH_MUTATION_EVENTS = {
    "os.rename": (0, 1),
    "os.replace": (0, 1),
    "os.link": (0, 1),
    "os.symlink": (0, 1),
}

#: For a symlink the two paths are not peers: `args[0]` is the link *target*
#: (a reference, which only has to be readable) and `args[1]` is where the
#: link is created (the write, which must be inside the plugin's roots).
_SYMLINK_TARGET_INDEX = 0

#: Index of the trailing `dir_fd` arguments, per event. A non-negative
#: directory file descriptor makes the adjacent path relative to *that*
#: directory rather than the process cwd, so `_normalize` would resolve it
#: against the wrong base and could approve a path that does not exist in
#: those terms. Denied outright rather than approximated.
_MUTATION_DIR_FD_INDEX = {
    "os.rename": (2, 3),
    "os.replace": (2, 3),
    "os.link": (2, 3),
    "os.symlink": (2,),
}


class CapabilityViolation(PermissionError):
    """A plugin attempted an operation outside its permitted capabilities."""


def _normalize(path: str | os.PathLike) -> Path:
    try:
        return Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        # Unresolvable path: treat as outside every root, i.e. denied.
        return Path(os.path.abspath(str(path)))


#: Directories under the home directory that hold credentials. This is a
#: *secondary* filter -- the primary control is that only allow-listed roots are
#: readable at all. It exists because the allow-list is seeded from `sys.path`,
#: which is influenceable, and it is deliberately a named list rather than a
#: "reject everything dotted" rule: that rule would also reject
#: `~/.cache/huggingface`, which is where the model weights live and which a
#: plugin is entitled to read.
_SENSITIVE_HOME_NAMES = frozenset({
    ".ssh", ".aws", ".azure", ".gcloud", ".kube", ".gnupg", ".netrc",
    ".config", ".docker", ".git-credentials", ".npmrc", ".pypirc",
    ".gem", ".cargo", ".mozilla", ".thunderbird", ".password-store",
})


def _is_credential_location(path: Path) -> bool:
    """Whether a path is the home directory, an ancestor of it, or a known
    credential directory inside it.

    Defence in depth rather than the primary control -- the primary control is
    that only allow-listed roots are readable at all. This exists because
    `sys.path` is influenceable: a host that put `""` (the current directory)
    or the user's home directory on it would otherwise hand a plugin the whole
    home directory to read.
    """
    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        return False
    if path == home:
        return True
    if path in home.parents:
        # An ancestor of home (``C:\\Users``, ``/home``) grants far more than
        # intended. Rejected unconditionally: nothing legitimate lives there,
        # and a Python installation at that level would be a system-wide
        # install, not a per-user one.
        return True
    if home in path.parents:
        relative = path.relative_to(home)
        first = relative.parts[0] if relative.parts else ""
        return first in _SENSITIVE_HOME_NAMES
    return False


def _python_roots() -> Set[Path]:
    """The interpreter's own installation.

    A plugin cannot import anything at all without these, so they are granted
    unconditionally rather than filtered: they contain no user data, and
    excluding them would break every plugin. In particular a per-user install
    under ``%LOCALAPPDATA%`` sits *inside* the home directory, so this must not
    be passed through `_is_credential_location`.
    """
    found: Set[Path] = set()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            found.add(_normalize(sysconfig.get_paths()[key]))
        except (KeyError, OSError, RuntimeError):
            continue
    for getter in (lambda: site.getsitepackages(), site.getusersitepackages):
        try:
            entries = getter()
        except Exception:  # noqa: BLE001
            continue
        if isinstance(entries, str):
            entries = [entries]
        for entry in entries:
            if entry:
                found.add(_normalize(entry))
    for prefix in (sys.base_prefix, sys.prefix):
        if prefix:
            found.add(_normalize(prefix))
    return found


def _model_cache_roots() -> Set[Path]:
    """Where model weights live, which plugins legitimately read.

    Granted by name and therefore not filtered either: ``~/.cache`` is under the
    home directory, and a filter that rejected it would stop every plugin from
    loading a model.
    """
    found: Set[Path] = set()
    names = ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
             "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME")
    for name in names:
        value = os.environ.get(name)
        if value:
            found.add(_normalize(value))
    return found


def default_readable_roots() -> Set[Path]:
    """The readable allow-list a plugin gets without being told otherwise.

    Derived from the interpreter's own import roots so that bundled plugins can
    still import the package they are written against -- the import roots are
    filtered, but not by the same rule as the deliberate grants above.
    """
    roots: Set[Path] = {_normalize(tempfile.gettempdir())}
    roots |= _python_roots()
    roots |= _model_cache_roots()
    for entry in list(sys.path):
        # An empty entry means "the current directory", which is not a path and
        # would otherwise resolve to wherever the worker happens to be.
        if not entry:
            continue
        try:
            resolved = _normalize(entry)
        except Exception:  # noqa: BLE001
            continue
        if not _is_credential_location(resolved):
            roots.add(resolved)
    return roots


def resolve_readable_roots(
    writable_roots: Iterable[str | os.PathLike] = (),
    readable_roots: Iterable[str | os.PathLike] = (),
) -> Set[Path]:
    """The effective readable allow-list.

    Split out from `install_capability_hook` because the interesting decision --
    whether a *caller-supplied* root is honoured or refused -- is worth testing
    directly, and a test for it should not have to create a credential
    directory to do so. `readable_roots` is exactly the parameter an attacker
    would aim at, so the filter has to apply to it and not only to the defaults.
    """
    readable: Set[Path] = default_readable_roots()
    for root in readable_roots:
        try:
            resolved = _normalize(root)
        except Exception:  # noqa: BLE001
            continue
        if not _is_credential_location(resolved):
            readable.add(resolved)
    # Writing implies reading, so the writable set is readable too.
    for root in writable_roots:
        try:
            readable.add(_normalize(root))
        except Exception:  # noqa: BLE001
            continue
    return readable


def install_capability_hook(
    writable_roots: Iterable[str | os.PathLike] = (),
    readable_roots: Iterable[str | os.PathLike] = (),
    allow_network: bool = False,
) -> Set[Path]:
    """Install the audit hook. Returns the resolved writable roots.

    Must be called before any plugin code runs. A second call adds a second
    hook rather than replacing the first; both deny, so the effect is the same
    and neither can be removed.

    Args:
        writable_roots: Directories the plugin may write into. The plugin's own
            directory and the system temp directory are always included.
        readable_roots: Extra directories the plugin may read. Combined with
            `default_readable_roots()`, and filtered the same way: a caller
            cannot widen the read allow-list into the host's own credentials by
            passing them in as configuration.
        allow_network: Escape hatch for debugging. Off by default.
    """
    roots: Set[Path] = {Path(tempfile.gettempdir()).resolve()}
    for root in writable_roots:
        try:
            roots.add(_normalize(root))
        except Exception:  # noqa: BLE001
            continue

    readable = resolve_readable_roots(roots, readable_roots)

    network_events = frozenset() if allow_network else frozenset(_NETWORK_EVENTS)
    process_events = frozenset(_PROCESS_EVENTS)

    def _within(path: Path, allowed: Set[Path]) -> bool:
        return any(path == root or root in path.parents for root in allowed)

    def hook(event: str, args: tuple) -> None:
        if event in network_events:
            target = args[0] if args else ""
            raise CapabilityViolation(
                f"network access is not permitted in a plugin worker "
                f"(blocked: {event} {target!r})"
            )

        if event in process_events:
            raise CapabilityViolation(
                f"process creation is not permitted in a plugin worker "
                f"(blocked: {event})"
            )

        if event == "open":
            # Both `open()` and `os.open()` raise this event, as
            # (path, mode, flags), so one check covers the builtin, the os
            # module, and everything pathlib does underneath (`Path.read_text`,
            # `Path.read_bytes`, `Path.write_text`, ...).
            try:
                target = _normalize(args[0])
                flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            except Exception:  # noqa: BLE001
                raise CapabilityViolation(
                    "open() with an unresolvable path is not permitted") from None
            if flags & _WRITE_FLAGS:
                if not _within(target, roots):
                    raise CapabilityViolation(
                        f"writing outside the plugin directory is not permitted "
                        f"(blocked: {target})"
                    )
                return
            if not _within(target, readable):
                raise CapabilityViolation(
                    f"reading outside the plugin's permitted roots is not "
                    f"permitted (blocked: {target})"
                )
            return

        if event in _LIST_DIR_EVENTS:
            try:
                target = _normalize(args[0])
            except Exception:  # noqa: BLE001
                raise CapabilityViolation(
                    f"{event} with an unresolvable path is not permitted") from None
            if not _within(target, readable):
                raise CapabilityViolation(
                    f"listing {target} is not permitted in a plugin worker "
                    f"(blocked: {event})"
                )
            return

        if event in _TWO_PATH_MUTATION_EVENTS:
            # A directory fd would change the base the paths resolve against,
            # so the containment check below could not be trusted. Refuse.
            for fd_index in _MUTATION_DIR_FD_INDEX[event]:
                if len(args) > fd_index and args[fd_index] not in (-1, None):
                    raise CapabilityViolation(
                        f"{event} with an explicit directory fd is not permitted "
                        f"in a plugin worker"
                    )
            for path_index in _TWO_PATH_MUTATION_EVENTS[event]:
                try:
                    target = _normalize(args[path_index])
                except Exception:  # noqa: BLE001
                    raise CapabilityViolation(
                        f"{event} with an unresolvable path is not permitted"
                    ) from None
                # A symlink's first path is the thing being pointed at. Creating
                # a reference to it is not a write, so it is checked against the
                # read allow-list; the link's own location (args[1]) is the write
                # and must be inside the plugin's roots like any other.
                is_reference = (event == "os.symlink"
                                and path_index == _SYMLINK_TARGET_INDEX)
                allowed = readable if is_reference else roots
                verb = "referencing" if is_reference else event
                if not _within(target, allowed):
                    raise CapabilityViolation(
                        f"{verb} outside the plugin directory is not permitted "
                        f"(blocked: {target})"
                    )
            return

        if event in ("os.remove", "os.rmdir", "os.mkdir",
                     "os.truncate", "os.chmod", "os.chown", "os.utime"):
            try:
                target = _normalize(args[0])
            except Exception:  # noqa: BLE001
                raise CapabilityViolation(
                    f"{event} with an unresolvable path is not permitted") from None
            if not _within(target, roots):
                raise CapabilityViolation(
                    f"{event} outside the plugin directory is not permitted "
                    f"(blocked: {target})"
                )
            return

    sys.addaudithook(hook)
    return roots