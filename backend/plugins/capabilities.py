"""Runtime capability restrictions for the plugin worker.

The AST lint in :mod:`plugin_sandbox` is static: it can be read, and anything it
misses is decided before a single line runs. This module is the dynamic layer --
an audit hook, so the decision happens as the interpreter performs the
operation, below whatever the plugin's source text was able to express.

Scope, chosen deliberately. This enforces the capabilities that are cheap to
bound and expensive to get wrong:

* no network (sockets, regardless of how they were reached)
* no process creation (subprocess, exec, spawn, os.system)
* no writes outside the plugin's own directory and the temp dir

It is explicitly **not** a full sandbox. A determined attacker with a working
interpreter can still find gaps; the OS limits in :mod:`limits` bound the blast
radius regardless. What this buys is that the obvious post-exploitation steps --
exfiltrate over the network, spawn a shell, write to a config file -- fail even
if the AST gate is bypassed outright.

Audit hooks cannot be removed once installed, which is the property we want:
plugin code cannot unhook itself.
"""
from __future__ import annotations

import os
import sys
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

# Modes that mutate. Reads are permitted: a plugin legitimately reads its own
# data files and the HF cache for model weights.
_WRITE_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC | os.O_EXCL
)


class CapabilityViolation(PermissionError):
    """A plugin attempted an operation outside its permitted capabilities."""


def _normalize(path: str | os.PathLike) -> Path:
    try:
        return Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        # Unresolvable path: treat as outside every root, i.e. denied.
        return Path(os.path.abspath(str(path)))


def install_capability_hook(
    writable_roots: Iterable[str | os.PathLike] = (),
    allow_network: bool = False,
) -> Set[Path]:
    """Install the audit hook. Returns the resolved writable roots.

    Must be called before any plugin code runs. Idempotent: a second call
    installs no additional hook, because the first one already denies.

    Args:
        writable_roots: Directories the plugin may write into. The plugin's own
            directory and the system temp directory are always included.
        allow_network: Escape hatch for debugging. Off by default.
    """
    roots: Set[Path] = {Path(tempfile.gettempdir()).resolve()}
    for root in writable_roots:
        try:
            roots.add(_normalize(root))
        except Exception:  # noqa: BLE001
            continue

    network_events = frozenset() if allow_network else frozenset(_NETWORK_EVENTS)
    process_events = frozenset(_PROCESS_EVENTS)

    def _within(path: Path) -> bool:
        return any(path == root or root in path.parents for root in roots)

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
            # open(path, mode, flags)
            try:
                target = _normalize(args[0])
                flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            except Exception:  # noqa: BLE001
                raise CapabilityViolation(
                    "open() with an unresolvable path is not permitted") from None
            if flags & _WRITE_FLAGS and not _within(target):
                raise CapabilityViolation(
                    f"writing outside the plugin directory is not permitted "
                    f"(blocked: {target})"
                )
            return

        if event in ("os.remove", "os.rename", "os.rmdir", "os.mkdir",
                     "os.link", "os.symlink", "os.truncate", "os.chmod",
                     "os.chown"):
            try:
                target = _normalize(args[0])
            except Exception:  # noqa: BLE001
                raise CapabilityViolation(
                    f"{event} with an unresolvable path is not permitted") from None
            if not _within(target):
                raise CapabilityViolation(
                    f"{event} outside the plugin directory is not permitted "
                    f"(blocked: {target})"
                )
            return

    sys.addaudithook(hook)
    return roots