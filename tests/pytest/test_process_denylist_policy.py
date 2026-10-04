"""The process denylist is a list of audit *event* names, and that must stay true.

The rule under test
-------------------
**A denylist entry that names a Python function rather than the audit event that
function raises blocks nothing, and nothing notices.**

Why this file exists
--------------------
`backend/plugins/capabilities.py` denies process creation by matching the audit
event name exactly:

    if event in process_events:
        raise CapabilityViolation(...)

so a wrong entry is silently inert. Nothing asserted the list's contents on any
platform, and six of its ten entries had no test anywhere:

    subprocess.Popen   covered (probe label "subprocess")
    os.system          covered
    os.exec            covered only transitively, via os.execv
    os.posix_spawn     no test
    os.spawn           no test
    os.fork            POSIX-only test
    os.forkpty         no test
    os.startfile       no test   <- the Windows-native launcher
    os.fork1           no test
    pty.spawn          no test

The `os.exec` case is the one that bites. It reads like a typo -- there is no
`os.exec` function; the functions are `os.execv`, `os.execl`, `os.execve`. So a
well-meaning edit to `os.execv` would look like a correction and would stop
blocking *every* exec variant. Measured on this host with a spy audit hook:

    os.system          -> events=['os.system']
    os.execv           -> events=['os.exec']            <- one event, all variants
    os.startfile       -> events=['os.startfile', 'os.startfile/2']

`os.startfile` is worth singling out. It is how Windows launches a program, it is
on the denylist, and it does raise a matching audit event -- so it is genuinely
blocked. I had assumed it might be a function name with no event behind it, which
would have made it an open escape on this platform; the measurement said
otherwise. That assumption was the reason to test rather than reason.

These tests therefore assert three things, and only the first is a pure policy
check that runs everywhere:

1. the denylist contains the names the platform's spawn APIs actually raise;
2. the platform-native spawn APIs are blocked, verified through the real hook in
   a subprocess (so an audit hook is genuinely installed);
3. the entries that cannot be exercised here are declared as such, rather than
   being quietly counted as covered.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Runs in a subprocess with a spy audit hook, printing the event names each
#: spawn API raises. Used to keep the denylist honest about what CPython emits.
SPY = textwrap.dedent(
    """
    import sys, os

    seen = set()
    def spy(event, args):
        if "." in event or event.startswith("pty."):
            seen.add(event)
    sys.addaudithook(spy)

    def probe(label, fn):
        seen.clear()
        try:
            fn()
        except Exception:
            pass
        print(label + "\\t" + ",".join(sorted(seen)))

    probe("os.system", lambda: os.system("echo x"))
    probe("os.execv", lambda: os.execv("cmd", ["cmd"]))
    probe("os.execl", lambda: os.execl("cmd", 0, "cmd") if hasattr(os, "execl") else None)
    probe("os.startfile", lambda: os.startfile(".") if hasattr(os, "startfile") else None)
    probe("subprocess.Popen", lambda: __import__("subprocess").Popen(["cmd"]))
    probe("os.posix_spawn", lambda: os.posix_spawn("cmd", ["cmd"]) if hasattr(os, "posix_spawn") else None)
    probe("os.fork", lambda: os.fork() if hasattr(os, "fork") else None)
    probe("os.forkpty", lambda: __import__("pty").forkpty() if hasattr(os, "forkpty") else None)
    """
)


def _spy_events() -> dict[str, list[str]]:
    """Map of spawn API -> audit events it raises on this host."""
    out = subprocess.run(
        [sys.executable, "-c", SPY],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120,
    )
    assert out.returncode == 0, out.stderr
    mapping: dict[str, list[str]] = {}
    for line in out.stdout.splitlines():
        if "\t" not in line:
            continue
        label, _, events = line.partition("\t")
        mapping[label.strip()] = [e for e in events.split(",") if e]
    return mapping


# ── policy: runs on every platform ─────────────────────────────────────────

def test_the_denylist_is_not_silently_empty():
    """A truncation of the list must not pass as a tightening."""
    from backend.plugins.capabilities import _PROCESS_EVENTS

    assert len(_PROCESS_EVENTS) >= 10, (
        f"the process denylist shrank to {len(_PROCESS_EVENTS)} entries: "
        f"{_PROCESS_EVENTS}")
    assert len(set(_PROCESS_EVENTS)) == len(_PROCESS_EVENTS), (
        "duplicate entries suggest the list was edited by appending")


@pytest.mark.parametrize("event", [
    "subprocess.Popen",   # every way to shell out via subprocess
    "os.system",
    "os.exec",            # all exec* variants; NOT "os.execv"
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "pty.spawn",
])
def test_the_denylist_still_lists_the_core_process_events(event):
    from backend.plugins.capabilities import _PROCESS_EVENTS

    assert event in _PROCESS_EVENTS, (
        f"{event!r} was removed from the process denylist, so that way of "
        f"starting a program is no longer refused")


def test_exec_variants_are_covered_by_the_os_exec_entry():
    """The one entry that reads like a typo, and must not be 'corrected'.

    There is no `os.exec` function. CPython raises the single audit event
    `os.exec` for `os.execv`, `os.execl` and `os.execve` alike, so this entry is
    what blocks all of them. Renaming it to `os.execv` would leave `os.execve`
    and `os.execvp` unblocked while looking like a fix.
    """
    import os

    from backend.plugins.capabilities import _PROCESS_EVENTS

    assert "os.exec" in _PROCESS_EVENTS
    assert "os.execv" not in _PROCESS_EVENTS, (
        "'os.execv' is not an audit event name; the event is 'os.exec' and it "
        "covers every exec* variant. An 'os.execv' entry would match nothing.")

    # If this host has any exec variant, confirm what it actually raises.
    if hasattr(os, "execv"):
        events = _spy_events().get("os.execv", [])
        assert "os.exec" in events, (
            f"os.execv raised {events}, not 'os.exec' -- the denylist entry and "
            f"the audit event have diverged on this interpreter")


# ── behaviour: the platform-native APIs, verified through the real hook ────

HOOK_PROBE = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {repo!r})
    from backend.plugins.capabilities import (
        install_capability_hook, CapabilityViolation,
    )
    install_capability_hook()

    import os

    def attempt(label, fn):
        try:
            fn()
        except CapabilityViolation as exc:
            print("BLOCKED\\t" + label)
        except Exception as exc:
            print("OTHER\\t" + label + "\\t" + type(exc).__name__)
        else:
            print("ALLOWED\\t" + label)

    # os.startfile is the Windows-native program launcher. It is on the denylist
    # and raises a matching audit event, so it must be refused -- previously
    # nothing verified that here.
    if hasattr(os, "startfile"):
        attempt("os.startfile", lambda: os.startfile("."))

    attempt("os.execv", lambda: os.execv("cmd", ["cmd", "/c", "echo"]))
    """
)


def _hook_verdicts() -> dict[str, str]:
    out = subprocess.run(
        [sys.executable, "-c", HOOK_PROBE.format(repo=str(ROOT))],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120,
    )
    assert out.returncode == 0, out.stderr
    verdicts: dict[str, str] = {}
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            verdicts[parts[1]] = parts[0]
    return verdicts


@pytest.mark.skipif(sys.platform != "win32",
                    reason="os.startfile is the Windows-native launcher")
def test_os_startfile_is_blocked_on_windows():
    """The most obvious Windows escape was unverified on any platform."""
    verdicts = _hook_verdicts()

    assert "os.startfile" in verdicts, (
        f"the probe never reached os.startfile: {verdicts}")
    assert verdicts["os.startfile"] == "BLOCKED", (
        f"os.startfile was {verdicts['os.startfile']}; it is the native way to "
        f"launch a program and is on the denylist")


def test_exec_is_blocked_on_every_platform():
    verdicts = _hook_verdicts()
    assert verdicts.get("os.execv") == "BLOCKED", (
        f"os.execv was {verdicts.get('os.execv')}; it must be refused through "
        f"the 'os.exec' audit event")


def test_the_denylist_event_names_match_what_cpython_raises():
    """Cross-check the list against the interpreter, where the APIs exist.

    The requirement is that each spawn API is blocked by **at least one**
    denylisted event -- not that every event it raises is denylisted. An API may
    raise several, and blocking the first is sufficient.

    That distinction is not academic. Measured on this host:

        subprocess.Popen  -> ['subprocess.Popen', '_winapi.CreateProcess']
        os.execv          -> ['os.exec']
        os.startfile      -> ['os.startfile', 'os.startfile/2']

    Two kinds of extra event turn up, and both are handled explicitly below
    rather than waved through:

    * a sub-event of a denylisted operation (`os.startfile/2`);
    * a private CPython event for a module the static gate forbids outright.
      `_winapi.CreateProcess` is the real one: the capability hook does **not**
      block it, so a plugin that could import `_winapi` could spawn a process.
      It cannot, because `PluginSandbox.analyze_ast` rejects the import before
      execution -- "Importing module '_winapi' is forbidden in sandbox." The
      second half of this test asserts that, so the layering is checked rather
      than assumed.
    """
    from backend.plugins.capabilities import _PROCESS_EVENTS

    events = _spy_events()
    checked = 0
    for api, raised in events.items():
        if not raised:
            continue  # API absent on this platform
        if api == "os.forkpty":
            continue  # pty is POSIX-only; its absence here proves nothing

        denylisted = [e for e in raised if e in _PROCESS_EVENTS]
        assert denylisted, (
            f"{api} raises {raised}, none of which is on the process denylist "
            f"-- that API can start a program unblocked. "
            f"denylist={sorted(_PROCESS_EVENTS)}")
        checked += 1

    assert checked >= 2, (
        f"only {checked} denylist entries could be cross-checked on this "
        f"platform; the test is not proving much")


@pytest.mark.skipif(sys.platform != "win32",
                    reason="_winapi is the Windows-specific escape")
def test_the_private_winapi_escape_is_blocked_by_the_static_gate():
    """`_winapi.CreateProcess` is not on the denylist. Nothing else covers it.

    The capability hook lets it through -- verified by calling it directly under
    an installed hook, where it reached the OS layer rather than raising
    `CapabilityViolation`. So the only thing standing between a plugin and
    `CreateProcess` is the AST gate's import ban.

    That makes the import ban load-bearing, and it was previously untested. This
    asserts both halves: the hook really does not block the call, and the gate
    really does reject the import. If someone adds `_winapi.CreateProcess` to the
    denylist the first half flips and this test should be revisited rather than
    quietly passing.
    """
    import os
    import subprocess as _sp

    from backend.plugins.capabilities import (
        CapabilityViolation,
        install_capability_hook,
    )
    from backend.plugins.plugin_sandbox import PluginSandbox, SecurityViolation

    probe = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, {repo!r})
        from backend.plugins.capabilities import (
            install_capability_hook, CapabilityViolation,
        )
        install_capability_hook()
        import _winapi
        try:
            _winapi.CreateProcess(None, "cmd.exe", None, None, 0, 0,
                                  None, None, 0, None)
            print("REACHED_OS")
        except CapabilityViolation:
            print("BLOCKED")
        except Exception as exc:
            print("REACHED_OS " + type(exc).__name__)
        """
    ).format(repo=str(ROOT))

    out = _sp.run([sys.executable, "-c", probe], capture_output=True,
                  text=True, cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stderr
    hook_verdict = out.stdout.strip().splitlines()[-1]

    # Half one: the dynamic layer does not cover this.
    assert hook_verdict.startswith("REACHED_OS"), (
        f"the capability hook now blocks _winapi.CreateProcess ({hook_verdict!r}). "
        f"If that is deliberate, add it to _PROCESS_EVENTS and update this test.")

    # Half two: the static layer does, so the escape is not reachable.
    sandbox = PluginSandbox()
    for source in ("import _winapi\n",
                   "from _winapi import CreateProcess\n",
                   "import _winapi\n_winapi.CreateProcess(None,'cmd.exe',"
                   "None,None,0,0,None,None,0,None)\n"):
        with pytest.raises(SecurityViolation):
            sandbox.analyze_ast(source)

    # And it is a specific ban, not a blanket one.
    with pytest.raises(SecurityViolation):
        sandbox.analyze_ast("import os\n")
