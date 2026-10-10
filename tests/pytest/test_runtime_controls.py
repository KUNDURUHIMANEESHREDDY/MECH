"""P1 runtime controls: society admission, body cap, model identity.

Seams: dispatcher.society_run (admission), backend.main body middleware
(HTTP edge), dispatcher.get_model_info (identity). Public except one
integration test that drives the whole ASGI app.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from backend.api import dispatcher as core  # noqa: E402

os.environ.setdefault("MECH_STORAGE_DB", ":memory:")
os.environ.setdefault("MECH_API_TOKEN", "runtime-controls-token")


@pytest.fixture(autouse=True)
def _clean_runs():
    core._society_runs.clear()
    yield
    core._society_runs.clear()


def _fake_run(status="running"):
    import queue
    return {"run_id": "r_test", "goal": "g", "model_name": "gpt2",
            "status": status, "cancel_requested": False, "events": [],
            "result": None, "queue": queue.Queue(maxsize=1000),
            "created": 0.0}


# ── admission control ───────────────────────────────────────────────────

def test_active_run_limit_enforced():
    core._society_runs["a1"] = _fake_run("running")
    core._society_runs["a2"] = _fake_run("running")
    res = core.society_run({"goal": "another"})
    assert res["status"] == "busy" and "active run limit" in res["error"]
    assert len(core._society_runs) == 2, "a refused run must allocate nothing"


def test_queue_capacity_enforced():
    # One below the active cap, so the queue check is what fires.
    core._society_runs["run0"] = _fake_run("running")
    for i in range(core._SOCIETY_MAX_QUEUED_RUNS):
        core._society_runs[f"q{i}"] = _fake_run("queued")
    res = core.society_run({"goal": "queued work"})
    assert res["status"] == "busy" and "queue capacity" in res["error"]
    assert len(core._society_runs) == (
        1 + core._SOCIETY_MAX_QUEUED_RUNS)


def test_finished_runs_do_not_block():
    core._society_runs["d1"] = _fake_run("completed")
    core._society_runs["d2"] = _fake_run("cancelled")
    res = core.society_run({"goal": "fine now"})
    assert res["status"] == "started", res
    run_id = res["runId"]
    try:
        assert core._society_runs[run_id]["status"] == "running"
    finally:
        # Stop the worker: cancel before it can spawn real work.
        core._society_runs[run_id]["cancel_requested"] = True
        core._society_runs.pop(run_id, None)


def test_goal_length_cap():
    res = core.society_run({"goal": "x" * (core._SOCIETY_MAX_GOAL_CHARS + 1)})
    assert res["status"] == "error" and "exceeds" in res["error"]
    assert core._society_runs == {}


def test_model_name_cap():
    res = core.society_run({"goal": "ok", "model_name": "m" * 129})
    assert res["status"] == "error" and "too long" in res["error"]


# ── body cap (whole-app integration) ───────────────────────────────────

def _h():
    return {"Authorization": f"Bearer {os.environ['MECH_API_TOKEN']}"}


def test_oversized_body_rejected_end_to_end(monkeypatch):
    from backend.core import auth as auth_mod
    from backend import main as backend_main
    monkeypatch.setenv("MECH_API_TOKEN", "runtime-controls-token")
    monkeypatch.setenv("MECH_MAX_BODY_BYTES", "2048")
    auth_mod.reset_cache()
    try:
        app = backend_main.app
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post(
                "/api/experiments", headers=_h(),
                content=b'{"x": "' + b"y" * 5000 + b'"}')
        assert r.status_code == 413, r.text
    finally:
        auth_mod.reset_cache()


def test_chunked_body_rejected(monkeypatch):
    from backend.core import auth as auth_mod
    from backend import main as backend_main
    monkeypatch.setenv("MECH_API_TOKEN", "runtime-controls-token")
    monkeypatch.setenv("MECH_MAX_BODY_BYTES", "2048")
    auth_mod.reset_cache()
    try:
        app = backend_main.app
        # No content-length: the streaming counter must catch it.
        def oversize():
            for _ in range(8):
                yield b"z" * 2048
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post("/api/experiments", headers=_h(), content=oversize())
        assert r.status_code == 413, r.text
    finally:
        auth_mod.reset_cache()


def test_normal_body_still_accepted(monkeypatch):
    from backend.core import auth as auth_mod
    from backend import main as backend_main
    monkeypatch.setenv("MECH_API_TOKEN", "runtime-controls-token")
    monkeypatch.setenv("MECH_MAX_BODY_BYTES", "8192")
    auth_mod.reset_cache()
    try:
        app = backend_main.app
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/api/status", headers=_h())
        assert r.status_code == 200, r.text
    finally:
        auth_mod.reset_cache()


# ── model identity ──────────────────────────────────────────────────────

def test_model_info_without_weights_never_returns_dimensions(monkeypatch):
    """No weights in memory: refuse, never hand back a shape table.

    Forced by clearing the engine so the result does not depend on whether
    torch is installed on the machine running the suite.
    """
    monkeypatch.setattr(core, "_engine", None)
    monkeypatch.setattr(core, "get_engine", lambda: None)
    body = core.get_model_info("gpt2-large")
    assert body.get("layers") is None, "a shape table returned with no weights"
    assert body.get("hidden_size") is None
    assert body.get("provenance") == "unavailable"
    assert body.get("status") == "unavailable"


class FakeEngine:
    """Mirrors the real engine's attested `info()` contract.

    The live engine wraps `info()` in `attest_measurement`, so it returns
    `provenance: live` with `attested: True`. A stub that omits those fields
    tests a code path the real system never takes.
    """

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def info():
        return {"status": "loaded", "model_name": "gpt2",
                "provenance": "live", "attested": True,
                "n_layers": 12, "d_model": 768}


def test_model_info_mismatch_is_reported(monkeypatch):
    monkeypatch.setattr(core, "_engine", FakeEngine())
    body = core.get_model_info("gpt2-large")
    assert body["model_mismatch"] is True
    assert body["model_requested"] == "gpt2-large"
    assert body["model_loaded"] == "gpt2"
    assert body["model_name"] == "gpt2-large"
    assert "mismatch" in body["reason"]
    assert body["n_layers"] == 12, "dimensions must still describe loaded weights"


def test_model_info_match_is_clean(monkeypatch):
    monkeypatch.setattr(core, "_engine", FakeEngine())
    body = core.get_model_info("gpt2")
    assert body.get("model_mismatch") is None
    assert body["n_layers"] == 12
    assert body["provenance"] == "live"
