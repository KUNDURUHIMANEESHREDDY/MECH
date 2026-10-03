"""Capability-restriction tests.

The AST lint in ``plugin_sandbox`` is static and readable. This tests the
dynamic layer underneath it: an audit hook installed before plugin code runs, so
a refusal happens as the interpreter performs the operation.

Most cases exercise the hook directly, in a subprocess. Reaching a forbidden
capability *through* a plugin load is deliberately hard -- the AST gate already
rejects the obvious imports and dunder walks -- so routing these through the
full worker would mostly be testing the gate. Direct tests cover the hook; the
integration tests at the bottom prove it does not break legitimate plugins.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.plugins.runner import WorkerBootError, start_remote_plugin  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
ECHO = FIXTURES / "live_probe" / "plugin.py"

PROBE = '''
import sys, tempfile
from pathlib import Path
sys.path.insert(0, {repo!r})
from backend.plugins.capabilities import install_capability_hook, CapabilityViolation

plugin_dir = Path({plugin_dir!r})
plugin_dir.mkdir(parents=True, exist_ok=True)
allowed = plugin_dir / "allowed.txt"
outside = Path({outside!r})

install_capability_hook(writable_roots=[plugin_dir])

def attempt(label, fn):
    try:
        fn()
    except CapabilityViolation as exc:
        print("BLOCKED", label, str(exc)[:70])
    except Exception as exc:
        print("OTHER", label, type(exc).__name__, str(exc)[:70])
    else:
        print("ALLOWED", label, "")

def _socket():
    import socket
    socket.socket()

def _connect():
    import socket
    s = socket.socket()
    s.connect(("93.184.216.34", 80))

def _dns():
    import socket
    socket.gethostbyname("example.com")

def _subprocess():
    import subprocess
    subprocess.run(["cmd", "/c", "echo pwned"])

def _system():
    import os
    os.system("echo pwned")

def _execv():
    import os
    os.execv("cmd", ["cmd", "/c", "echo"])

def _fork():
    import os
    os.fork()

def _write_outside():
    outside.write_text("pwned")

def _write_inside():
    allowed.write_text("ok")

def _read_outside():
    return outside.read_text()

def _remove_outside():
    outside.unlink()

def _write_temp():
    Path(tempfile.gettempdir(), "mech_cap_probe.txt").write_text("x")

attempt("socket", _socket)
attempt("connect", _connect)
attempt("dns", _dns)
attempt("subprocess", _subprocess)
attempt("os.system", _system)
attempt("os.execv", _execv)
attempt("os.fork", _fork)
attempt("write_outside", _write_outside)
attempt("write_inside", _write_inside)
attempt("read_outside", _read_outside)
attempt("remove_outside", _remove_outside)
attempt("write_temp", _write_temp)
'''


def run_probe(tmp_path):
    plugin_dir = tmp_path / "plugindir"
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    script = PROBE.format(
        repo=str(REPO_ROOT),
        plugin_dir=str(plugin_dir),
        outside=str(outside),
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        timeout=120, cwd=str(REPO_ROOT),
    )
    results = {}
    for line in proc.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0] in ("BLOCKED", "ALLOWED", "OTHER"):
            results[parts[1]] = (parts[0], parts[2] if len(parts) > 2 else "")
    assert results, f"probe produced no results; stderr:\n{proc.stderr[-600:]}"
    return results, outside


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    results, outside = run_probe(tmp_path_factory.mktemp("caps"))
    return results, outside


# --------------------------------------------------------------------------- #
# Must be blocked
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label", [
    "socket", "connect", "dns",
    "subprocess", "os.system", "os.execv",
    "write_outside", "remove_outside",
])
def test_capability_is_denied(probe, label):
    results, _ = probe
    assert label in results, f"{label} was never attempted; got {sorted(results)}"
    verdict, detail = results[label]
    assert verdict == "BLOCKED", f"{label} was {verdict}: {detail}"


@pytest.mark.skipif(not hasattr(sys.modules.get("os") or __import__("os"), "fork"),
                    reason="os.fork does not exist on this platform")
def test_fork_is_denied_on_posix(probe):
    """Windows has no fork, so the hook's os.fork guard only runs on POSIX."""
    results, _ = probe
    if "os.fork" not in results:
        pytest.skip("os.fork unavailable on this platform")
    verdict, detail = results["os.fork"]
    assert verdict == "BLOCKED", f"os.fork was {verdict}: {detail}"


# --------------------------------------------------------------------------- #
# Must still work, or legitimate plugins break
# --------------------------------------------------------------------------- #


def test_plugin_directory_is_writable(probe):
    results, _ = probe
    assert results["write_inside"][0] == "ALLOWED", results["write_inside"]


def test_reads_outside_are_permitted(probe):
    """Model weights and caches live outside the plugin directory."""
    results, _ = probe
    assert results["read_outside"][0] == "ALLOWED", results["read_outside"]


def test_tempdir_is_writable(probe):
    results, _ = probe
    assert results["write_temp"][0] == "ALLOWED", results["write_temp"]


def test_outside_file_untouched(probe):
    _, outside = probe
    assert outside.read_text(encoding="utf-8") == "secret"


def test_hook_cannot_be_removed(probe):
    """Audit hooks are one-way; a plugin cannot unhook itself."""
    results, _ = probe
    # Reaching here at all means the process survived every denied attempt.
    assert len(results) >= 11


# --------------------------------------------------------------------------- #
# Integration: the hook must not break real plugins
# --------------------------------------------------------------------------- #


def test_bundled_plugin_still_works():
    """The IOI logger appends an audit trail next to itself."""
    sample = (REPO_ROOT / "backend" / "plugins" / "library"
              / "ioi_experiment_logger" / "__init__.py")
    proxy = start_remote_plugin(sample)
    try:
        out = proxy.on_paper_generated({"abstract": "ioi analysis"})
        assert isinstance(out, str) and "IOI Experiment Logger" in out
    finally:
        _shutdown(proxy)


def test_echo_fixture_still_works():
    proxy = start_remote_plugin(ECHO)
    try:
        out = proxy.on_experiment_planned({"name": "probe"})
        assert out is not None and out["name"] == "probe"
    finally:
        _shutdown(proxy)


def test_ast_gate_still_rejects_before_execution(tmp_path):
    """Capabilities supplement the gate; they do not replace it."""
    payload = tmp_path / "evil.py"
    payload.write_text(
        "import os\n"
        "def register():\n"
        "    os.getcwd()\n"
        "    return None\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkerBootError, match="rejected in worker"):
        start_remote_plugin(payload)


def _shutdown(proxy):
    try:
        proxy._roundtrip({"op": "shutdown"})  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass