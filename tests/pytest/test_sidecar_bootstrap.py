"""Sidecar bootstrap tests.

``desktop_service.py`` is spawned by ``electron/main.ts`` and imports two
modules that live at different roots: ``storage.database`` resolves against
``backend/`` while ``backend.neuron_inspector`` resolves against the repo root.
It previously put only ``frontend/`` on ``sys.path``, so neither import
resolved and the process died at import time.

These tests assert the process actually boots and serves a request, in both the
dev checkout and the packaged resource layout.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "frontend"
SIDECAR = FRONTEND / "scripts" / "desktop_service.py"


def _request(method, params=None, cwd=None, env_extra=None, db=None):
    """Send one request to a freshly spawned sidecar and return its reply.

    Mirrors PythonBridge.ensureStarted, which always passes ``--db``.
    """
    import tempfile
    env = dict(os.environ)
    env.update(env_extra or {})
    if db is None:
        tmpdir = tempfile.mkdtemp()
        db = str(Path(tmpdir) / "desktop.sqlite3")
    proc = subprocess.run(
        [sys.executable, str(SIDECAR), "--db", str(db)],
        input=json.dumps({"method": method, "params": params or {}}) + "\n",
        capture_output=True, text=True, timeout=180, cwd=str(cwd or REPO_ROOT),
        env=env,
    )
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return None, proc.stderr
    try:
        return json.loads(lines[-1]), proc.stderr
    except ValueError:
        return None, proc.stdout


def ok(reply):
    """The sidecar's success envelope is {"id", "result"}."""
    return isinstance(reply, dict) and "result" in reply and "error" not in reply


def err(reply):
    """The failure envelope is {"id", "error": {"code", "message"}}."""
    return isinstance(reply, dict) and isinstance(reply.get("error"), dict)


def message(reply):
    return (reply or {}).get("error", {}).get("message", "")


def test_sidecar_boots_in_repo_checkout():
    """The bug: the process died with ModuleNotFoundError on 'storage'."""
    reply, stderr = _request("settings.get")
    assert reply is not None, f"sidecar produced no response; stderr:\n{stderr[-800:]}"
    assert ok(reply), str(reply)[:300]


def test_sidecar_boots_when_cwd_is_elsewhere():
    """sys.path must be derived from __file__, not the working directory."""
    reply, stderr = _request("settings.get", cwd=REPO_ROOT / "backend")
    assert reply is not None, f"no response from backend/ cwd; stderr:\n{stderr[-800:]}"
    assert ok(reply), str(reply)[:300]


def test_sidecar_serves_settings_payload():
    reply, _ = _request("settings.get")
    assert ok(reply), str(reply)[:200]
    assert "workspacePath" in reply["result"], str(reply["result"])[:200]


def test_sidecar_boots_without_the_ml_stack(monkeypatch):
    """A broken ML dependency must not stop settings/workspace from working.

    The eager "from backend.neuron_inspector import GPT2Model" pulled in
    transformer_lens -> datasets at import time, so a bad ML install made the
    whole sidecar unbootable even though those methods need no model.
    """
    import tempfile
    # Break the ML import path the way a bad install does.
    blocker = Path(tempfile.mkdtemp())
    (blocker / "transformer_lens.py").write_text(
        "raise ImportError('simulated broken ML dependency')\n", encoding="utf-8")

    reply, stderr = _request(
        "settings.get", env_extra={"PYTHONPATH": str(blocker)})
    assert reply is not None, f"no response; stderr:\n{stderr[-800:]}"
    assert ok(reply), f"sidecar failed to boot without ML stack: {str(reply)[:300]}"


def test_neuron_method_reports_ml_failure_cleanly():
    """A neuron request must fail with a message, not kill the process."""
    reply, _ = _request("neuron.modelInfo")
    # Either it works (ML stack healthy) or it returns a structured error.
    assert reply is not None
    if not ok(reply):
        assert err(reply), str(reply)[:250]
        assert "unavailable" in message(reply) or "ImportError" in message(reply), \
            str(reply)[:250]


def test_sidecar_serves_workspace_describe():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        allowed = Path(tmp) / "work"
        allowed.mkdir()
        (allowed / "a.py").write_text("x", encoding="utf-8")
        reply, stderr = _request(
            "workspace.describe", {"path": str(allowed)},
            env_extra={"MECH_WORKSPACE_ROOTS": str(allowed)})
        assert reply is not None, f"no response; stderr:\n{stderr[-800:]}"
        assert ok(reply), str(reply)[:250]
        assert reply["result"]["fileCount"] == 1, str(reply["result"])[:200]


def test_sidecar_refuses_path_outside_roots():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        allowed = Path(tmp) / "work"
        allowed.mkdir()
        outside = Path(tmp) / "outside"
        outside.mkdir()
        reply, _ = _request(
            "workspace.describe", {"path": str(outside)},
            env_extra={"MECH_WORKSPACE_ROOTS": str(allowed)})
        assert err(reply), str(reply)[:250]
        assert "outside the allowed workspace roots" in message(reply), \
            str(reply)[:250]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink semantics")
def test_sidecar_resolves_symlinked_script(tmp_path):
    """Path resolution must follow a symlink back to the real repo."""
    link = tmp_path / "sidecar_link.py"
    link.symlink_to(SIDECAR)
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, str(link), "--db", str(tmp_path / "d.sqlite3")],
        input=json.dumps({"method": "settings.get"}) + "\n",
        capture_output=True, text=True, timeout=180, cwd=str(tmp_path), env=env,
    )
    assert proc.stdout.strip(), f"no response via symlink; stderr:\n{proc.stderr[-500:]}"


def test_packaged_layout_resolves(tmp_path):
    """Simulate the packaged resource layout: resources/backend/...

    electron-builder copies ../backend -> resources/backend and the app root is
    resourcesPath, so the sidecar must find both roots relative to the app
    root, not relative to the repo.
    """
    import shutil
    resources = tmp_path / "resources"
    app_root = resources / "frontend"
    (app_root / "scripts").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "backend", resources / "backend",
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(SIDECAR, app_root / "scripts" / SIDECAR.name)

    reply, stderr = _request(
        "settings.get", cwd=app_root, db=tmp_path / "packaged.sqlite3",
        env_extra={"MECH_APP_ROOT": str(app_root)})
    assert reply is not None, (
        f"packaged layout produced no response; stderr:\n{stderr[-800:]}")
    assert ok(reply), str(reply)[:300]
