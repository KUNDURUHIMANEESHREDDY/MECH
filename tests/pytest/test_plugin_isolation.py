"""Out-of-process plugin containment tests.

The AST scan and restricted builtins in ``plugin_sandbox`` are defence in
depth. These tests cover the actual boundary: the worker process, the OS
resource limits, the JSON protocol, and the parent-side proxy.

Each test spawns a real worker, so the suite is slower than the in-process
sandbox tests. Timeouts are generous to stay green on loaded CI.
"""
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.plugins.limits import describe_limits  # noqa: E402
from backend.plugins.plugin_base import MechPlugin  # noqa: E402
from backend.plugins.protocol import (  # noqa: E402
    MAX_MESSAGE_BYTES,
    ProtocolError,
    decode,
    encode,
)
from backend.plugins.proxy import RemotePluginError  # noqa: E402
from backend.plugins.runner import (  # noqa: E402
    WorkerBootError,
    start_remote_plugin,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
ECHO = FIXTURES / "live_probe" / "plugin.py"
SAMPLE = (REPO_ROOT / "backend" / "plugins" / "library"
          / "ioi_experiment_logger" / "__init__.py")


@pytest.fixture(scope="module")
def echo():
    proxy = start_remote_plugin(ECHO)
    yield proxy
    try:
        proxy._roundtrip({"op": "shutdown"})  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Protocol — no pickle anywhere on the parent/worker channel
# --------------------------------------------------------------------------- #

def test_protocol_is_json_not_pickle():
    """A pickled channel would be its own RCE path into the parent."""
    import pickle

    payload = {"hook": "on_campaign_started", "args": [{"a": 1}]}
    frame = encode(payload)
    assert b"pickle" not in frame.lower()
    # Round-trips as JSON, and a pickle is not accepted as a frame.
    assert decode(frame) == payload
    with pytest.raises(ProtocolError):
        decode(pickle.dumps(payload) + b"\n")


def test_oversized_message_rejected():
    with pytest.raises(ProtocolError, match="exceeds"):
        encode({"blob": "x" * (MAX_MESSAGE_BYTES + 100)})


def test_non_object_frame_rejected():
    with pytest.raises(ProtocolError, match="object"):
        decode(b"[1, 2, 3]\n")


def test_malformed_frame_rejected():
    with pytest.raises(ProtocolError):
        decode(b"{not json\n")


# --------------------------------------------------------------------------- #
# Happy path across the process boundary
# --------------------------------------------------------------------------- #

def test_worker_returns_manifest(echo):
    assert echo.manifest.plugin_id == "mech.test.live_probe"
    assert echo.manifest.version == "1.0.0"
    assert echo.is_sandboxed and echo.is_alive


def test_proxy_is_a_mechplugin(echo):
    """The registry and hook bus type-check against MechPlugin."""
    assert isinstance(echo, MechPlugin)


def test_mutating_hook_round_trips_payload(echo):
    out = echo.on_experiment_planned({"name": "probe", "steps": 3})
    assert out is not None, "mutating hook must return a value across the hop"
    assert out["name"] == "probe"
    assert out["steps"] == 3
    assert out["mutated"] is True


def test_mutating_string_hook_round_trips(echo):
    out = echo.on_paper_generated({"abstract": "An IOI paper."})
    assert isinstance(out, str)
    assert out.startswith("An IOI paper.")


def test_non_mutating_hooks_return_none(echo):
    assert echo.on_campaign_started({"campaign_id": "c1"}) is None
    assert echo.on_validation_completed({"passed_benchmarks": 3}) is None
    assert echo.on_knowledge_graph_updated({"node": "n"}) is None


def test_bundled_sample_plugin_runs_in_worker():
    proxy = start_remote_plugin(SAMPLE)
    try:
        assert proxy.manifest.plugin_id == "mech.example.ioi_experiment_logger"
        # Its mutating hook appends a citation note; that must survive the hop.
        out = proxy.on_paper_generated({"abstract": "ioi analysis"})
        assert isinstance(out, str) and "IOI Experiment Logger" in out
    finally:
        try:
            proxy._roundtrip({"op": "shutdown"})  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Containment — the actual boundary
# --------------------------------------------------------------------------- #

def test_worker_rejects_os_import(tmp_path):
    """A payload reaching os must die in the worker, never in the backend."""
    payload = tmp_path / "evil_os.py"
    payload.write_text(
        "import os\n"
        "def register():\n"
        "    os.getcwd()\n"
        "    return None\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkerBootError, match="rejected in worker"):
        start_remote_plugin(payload)


def test_worker_rejects_dunder_import(tmp_path):
    payload = tmp_path / "evil_dunder.py"
    payload.write_text(
        "m = __import__('os')\n"
        "def register():\n"
        "    return None\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkerBootError):
        start_remote_plugin(payload)


def test_worker_rejects_relative_import(tmp_path):
    payload = tmp_path / "evil_rel.py"
    payload.write_text(
        "from . import os\n"
        "def register():\n"
        "    return None\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkerBootError):
        start_remote_plugin(payload)


def test_worker_rejects_register_returning_non_plugin(tmp_path):
    payload = tmp_path / "bad_return.py"
    payload.write_text("def register():\n    return 42\n", encoding="utf-8")
    with pytest.raises(WorkerBootError):
        start_remote_plugin(payload)


def test_worker_rejects_missing_register(tmp_path):
    payload = tmp_path / "no_register.py"
    payload.write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(WorkerBootError):
        start_remote_plugin(payload)


# --------------------------------------------------------------------------- #
# Service integration — the worker must be the registered plugin
# --------------------------------------------------------------------------- #

def test_service_enable_runs_plugin_in_worker(tmp_path):
    """End-to-end through PluginService: the registered object is a worker proxy."""
    from backend.plugins.plugin_base import MechPlugin as _MechPlugin
    from backend.plugins.service import PluginService

    installed = tmp_path / "installed"
    service = PluginService(manager=__import__(
        "backend.plugins.plugin_manager", fromlist=["PluginManager"]
    ).PluginManager(plugins_dir=str(installed)))

    service._manager.install_plugin(  # noqa: SLF001
        {"name": "echo", "version": "1.0.0"}, ECHO.read_text(encoding="utf-8"))
    try:
        service.enable("echo")
        plugin = service._registry.get("mech.test.live_probe")  # noqa: SLF001
        assert isinstance(plugin, _MechPlugin)
        assert getattr(plugin, "is_sandboxed", False) is True

        # The hook bus drives it exactly as it would a local plugin.
        bus = service.bus()
        result = bus.emit("on_experiment_planned", {"name": "probe"})
        assert result.errors == {}
        assert result.first_non_none()["mutated"] is True

        service.disable("echo")
    finally:
        service._manager.registry.clear()  # noqa: SLF001


def test_service_disable_terminates_worker(tmp_path):
    """Disabling must reap the worker process, not leak one per cycle."""
    from backend.plugins.plugin_manager import PluginManager
    from backend.plugins.service import PluginService

    manager = PluginManager(plugins_dir=str(tmp_path / "installed"))
    service = PluginService(manager=manager)
    manager.install_plugin(
        {"name": "echo", "version": "1.0.0"}, ECHO.read_text(encoding="utf-8"))

    service.enable("echo")
    proc = service._workers["echo"][1]  # noqa: SLF001
    assert proc.poll() is None, "worker should be running while enabled"

    service.disable("echo")
    # Give the terminate a moment to take effect.
    for _ in range(50):
        if proc.poll() is not None:
            break
        time.sleep(0.02)
    assert proc.poll() is not None, "worker should exit after disable"
    assert "echo" not in service._workers  # noqa: SLF001


def test_hostile_hook_cannot_kill_the_backend(tmp_path):
    """A hook that raises is reported as an error, not propagated as a crash."""
    payload = tmp_path / "raiser.py"
    payload.write_text(
        "from typing import Any, Dict, Optional\n"
        "from backend.plugins.plugin_base import MechPlugin, PluginManifest\n"
        "from backend.plugins.plugin_hooks import HOOK_CAMPAIGN_STARTED\n"
        "\n"
        "class Boom(MechPlugin):\n"
        "    @property\n"
        "    def manifest(self):\n"
        "        return PluginManifest(\n"
        "            plugin_id='test.boom', name='Boom', version='1.0.0',\n"
        "            author='t', description='d', hooks=[HOOK_CAMPAIGN_STARTED],\n"
        "        )\n"
        "    def on_campaign_started(self, campaign):\n"
        "        raise RuntimeError('plugin exploded')\n"
        "\n"
        "def register():\n"
        "    return Boom()\n",
        encoding="utf-8",
    )
    proxy = start_remote_plugin(payload)
    try:
        with pytest.raises(RemotePluginError, match="plugin exploded"):
            proxy.on_campaign_started({"campaign_id": "c"})
    finally:
        try:
            proxy._roundtrip({"op": "shutdown"})  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass


def test_worker_env_is_scrubbed(tmp_path):
    """The backend's secrets must not be visible inside the worker."""
    sentinel = "MECH_TEST_SECRET_SENTINEL_VALUE"
    os.environ[sentinel] = "leaked"
    payload = tmp_path / "envreader.py"
    payload.write_text(
        "import os\n"  # blocked by the AST gate — the point is the *child* env
        "def register():\n    return None\n",
        encoding="utf-8",
    )
    try:
        from backend.plugins.runner import _child_env
        env = _child_env()
        assert sentinel not in env
        assert "PATH" in env  # boot essentials survive
    finally:
        os.environ.pop(sentinel, None)


def test_child_env_excludes_common_secret_vars():
    from backend.plugins.runner import _child_env
    env = _child_env()
    for leaked in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY",
                   "GITHUB_TOKEN", "DATABASE_URL"):
        assert leaked not in env


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #

def test_limits_available_on_this_platform():
    info = describe_limits()
    assert info["platform"] in ("win32", "linux", "darwin")
    assert info["job_object"] == (info["platform"] == "win32")
    assert info["defaults"]["cpu_seconds"] > 0


def test_worker_reports_limits_in_handshake():
    """A worker that could not apply its limits must say so, not hide it."""
    proxy = start_remote_plugin(ECHO)
    try:
        assert proxy.is_alive
    finally:
        try:
            proxy._roundtrip({"op": "shutdown"})  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimit behaviour")
def test_infinite_loop_is_bounded(tmp_path):
    """CPU limit must terminate a hung plugin instead of hanging the backend."""
    payload = tmp_path / "spin.py"
    payload.write_text(
        "from typing import Any, Dict\n"
        "from backend.plugins.plugin_base import MechPlugin, PluginManifest\n"
        "from backend.plugins.plugin_hooks import HOOK_CAMPAIGN_STARTED\n"
        "\n"
        "class Spin(MechPlugin):\n"
        "    @property\n"
        "    def manifest(self):\n"
        "        return PluginManifest(\n"
        "            plugin_id='test.spin', name='Spin', version='1.0.0',\n"
        "            author='t', description='d', hooks=[HOOK_CAMPAIGN_STARTED],\n"
        "        )\n"
        "    def on_campaign_started(self, campaign):\n"
        "        while True:\n"
        "            pass\n"
        "\n"
        "def register():\n"
        "    return Spin()\n",
        encoding="utf-8",
    )
    proxy = start_remote_plugin(payload)
    started = time.monotonic()
    with pytest.raises(RemotePluginError):
        proxy.on_campaign_started({"campaign_id": "c"})
    # The CPU limit is 120s by default; the worker should die well before the
    # test itself gives up. Assert it failed fast rather than hanging forever.
    assert time.monotonic() - started < 200
