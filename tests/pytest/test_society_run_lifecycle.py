"""Society run lifecycle: no run may hold an admission slot forever.

The defect this file pins down: a Society run whose worker thread died (or
hung) kept ``status == "running"`` in ``_society_runs`` for the lifetime of
the process, so every later submission was refused with
``{"status": "busy", "error": "active run limit reached (2/2)"}`` and the
tool was dead until the backend was restarted.

The fix lives in backend/api/dispatcher.py:

* every admitted run carries a deadline (``MECH_SOCIETY_RUN_TIMEOUT_S``,
  default 600s) and the thread that runs it;
* a run that outlives its deadline is finalized as a failure whose reason
  names the timeout, and a run whose worker is gone is finalized as
  ``interrupted`` -- both release the admission slot;
* ``GET /api/society/runs/{id}`` never reports ``running`` for a run that can
  no longer progress;
* ``POST /api/society/runs/{id}/cancel`` asks a run to stop and reports what
  actually happened: 404 for an unknown run, 409 for a finished one,
  cancelled outright when the pipeline never started, still running
  otherwise.

The admission limit itself is deliberate and is kept: the third concurrent
run is *supposed* to be refused. The negative test below asserts that
refusal, because a refusal that is correct behaviour is not a bug to fix.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.api import dispatcher  # noqa: E402
from backend.core import auth as auth_mod  # noqa: E402
from backend.core import evidence_graph as eg  # noqa: E402

AUTH_HEADERS = auth_mod.auth_headers()

#: Structurally valid run id that was never minted, so no record exists.
UNKNOWN_RUN_ID = "r000000000000"


class _StubSociety:
    """A ResearchSocietyV2 stand-in whose pipeline blocks until released.

    The real pipeline loads GPT-2 and runs for minutes. These tests are
    about the lifecycle *around* the pipeline, so the model is stubbed and
    the run is held open deterministically.
    """

    def __init__(self, block=True, die=False, result=None):
        self.block = block
        self.die = die
        self.result = (result if result is not None
                       else {"status": "completed", "summary": "stub"})
        self.release = threading.Event()
        self.calls = 0
        self._lock = threading.Lock()

    def run_blocking(self, goal, model_name=None, on_event=None, run_id=""):
        with self._lock:
            self.calls += 1
        if on_event is not None:
            on_event({"type": "stage", "goal": goal})
        if self.die:
            # A hard worker death. SystemExit ends the thread without
            # running the finalisation path -- the same shape as a native
            # call that takes the worker down mid-run.
            raise SystemExit("simulated hard worker death")
        if self.block:
            self.release.wait(timeout=30)
        return dict(self.result)

    def stop(self):
        self.release.set()


@pytest.fixture
def stubs():
    """Every stub handed to the backend, so the test can release them."""
    return []


@pytest.fixture(autouse=True)
def _clean_society(stubs):
    with dispatcher._society_lock:
        dispatcher._society_runs.clear()
    yield
    for stub in stubs:
        stub.stop()
    with dispatcher._society_lock:
        dispatcher._society_runs.clear()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient whose durable run records land in a temp directory."""
    monkeypatch.setattr(eg, "EVIDENCE_DIR", str(tmp_path))
    from backend.main import app
    return TestClient(app, raise_server_exceptions=False)


def _install(monkeypatch, stub):
    monkeypatch.setattr(dispatcher, "_society_get", lambda: stub)
    return stub


def _submit(client, goal="lifecycle probe"):
    res = client.post("/api/society/run", json={"goal": goal},
                      headers=AUTH_HEADERS)
    assert res.status_code == 200, res.text
    return res.json()


def _status(run_id):
    with dispatcher._society_lock:
        run = dispatcher._society_runs.get(run_id)
        return None if run is None else run.get("status")


def _pipeline_started(run_id):
    with dispatcher._society_lock:
        run = dispatcher._society_runs.get(run_id)
        return bool(run is not None and run.get("pipeline_started"))


def _worker_alive(run_id):
    with dispatcher._society_lock:
        run = dispatcher._society_runs.get(run_id)
        worker = None if run is None else run.get("worker")
    return bool(worker is not None and worker.is_alive())


def _deadline(run_id):
    with dispatcher._society_lock:
        run = dispatcher._society_runs.get(run_id)
        return 0.0 if run is None else float(run.get("deadline") or 0.0)


def _wait_for(predicate, message, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for %s" % message)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_dead_worker_is_not_running_and_no_longer_holds_a_slot(client, monkeypatch,
                                                              stubs):
    """A worker that died without finalizing must not hold the slot.

    The death is simulated with SystemExit, which CPython's own
    threading.excepthook ignores; pytest replaces that hook, so the
    thread-exception warning is muted here rather than left to look like
    a failure.
    """
    stub = _StubSociety(die=True)
    stubs.append(stub)
    _install(monkeypatch, stub)

    body = _submit(client, "the worker dies mid-run")
    assert body["status"] == "started", body
    run_id = body["runId"]

    _wait_for(lambda: not _worker_alive(run_id), "the worker thread to die")

    status = client.get("/api/society/runs/%s" % run_id,
                        headers=AUTH_HEADERS).json()
    assert status["status"] != "running", status
    assert status["status"] == "interrupted", status
    assert status["reason"], status

    # The slot is free again: two more runs are admitted, the third is not.
    stub.die = False
    first = _submit(client, "after the dead run")
    second = _submit(client, "after the dead run")
    assert first["status"] == "started", first
    assert second["status"] == "started", second
    third = _submit(client, "overflow")
    assert third["status"] == "busy", third
    assert "active run limit reached (2/2)" in third["error"], third


def test_timeout_finalizes_the_run_with_a_reason_naming_the_timeout(client,
                                                                    monkeypatch,
                                                                    stubs):
    """A hung pipeline is finalized as a failure whose reason says why."""
    monkeypatch.setenv("MECH_SOCIETY_RUN_TIMEOUT_S", "1")
    stub = _StubSociety(block=True)
    stubs.append(stub)
    _install(monkeypatch, stub)

    body = _submit(client, "hangs forever")
    run_id = body["runId"]

    _wait_for(lambda: time.time() >= _deadline(run_id),
              "the run to pass its deadline")

    status = client.get("/api/society/runs/%s" % run_id,
                        headers=AUTH_HEADERS).json()
    assert status["status"] == "failed", status
    assert "timeout" in status["reason"].lower(), status
    assert "timeout" in str(status["result"]).lower(), status

    # The slot is released even though the worker thread is still stuck.
    monkeypatch.setenv("MECH_SOCIETY_RUN_TIMEOUT_S", "600")
    first = _submit(client, "after the timeout")
    second = _submit(client, "after the timeout")
    assert first["status"] == "started", first
    assert second["status"] == "started", second
    third = _submit(client, "overflow")
    assert third["status"] == "busy", third

    # The stuck worker returns later; a late result must not overwrite the
    # timeout, and the run must not go back to "running".
    stub.stop()
    _wait_for(lambda: not _worker_alive(run_id), "the stuck worker to return")
    again = client.get("/api/society/runs/%s" % run_id,
                       headers=AUTH_HEADERS).json()
    assert again["status"] == "failed", again
    assert "timeout" in again["reason"].lower(), again


def test_cancelling_a_finished_run_is_409(client, monkeypatch, stubs):
    """A finished run cannot be cancelled: that is a 409, not a success."""
    stub = _StubSociety(block=False, result={"status": "completed",
                                              "summary": "done"})
    stubs.append(stub)
    _install(monkeypatch, stub)

    body = _submit(client, "finishes quickly")
    run_id = body["runId"]
    _wait_for(lambda: _status(run_id) == "completed", "the run to finish")

    res = client.post("/api/society/runs/%s/cancel" % run_id,
                      headers=AUTH_HEADERS)
    assert res.status_code == 409, res.text
    assert "cannot be cancelled" in res.json()["detail"], res.text


def test_cancelling_an_unknown_run_is_404(client):
    res = client.post("/api/society/runs/%s/cancel" % UNKNOWN_RUN_ID,
                      headers=AUTH_HEADERS)
    assert res.status_code == 404, res.text


def test_cancelling_a_run_that_never_started_cancels_it(client, monkeypatch,
                                                        stubs):
    """Nothing is in flight, so the terminal state is certain."""
    stub = _StubSociety(block=True)
    stubs.append(stub)
    _install(monkeypatch, stub)

    # Admission reserves the slot before the worker thread exists: that is
    # the state a run is in while its pipeline has not started.
    reason, run = dispatcher._society_admit_and_reserve()
    assert reason == "", reason
    run_id = run["run_id"]
    assert run["pipeline_started"] is False

    res = client.post("/api/society/runs/%s/cancel" % run_id,
                      headers=AUTH_HEADERS)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "cancelled", body
    assert body["outcome"] == "cancelled", body
    assert body["reason"], body

    status = client.get("/api/society/runs/%s" % run_id,
                        headers=AUTH_HEADERS).json()
    assert status["status"] == "cancelled", status

    # The cancelled run released its slot.
    first = _submit(client, "after the cancel")
    second = _submit(client, "after the cancel")
    assert first["status"] == "started", first
    assert second["status"] == "started", second


def test_cancelling_a_running_run_flags_it_and_keeps_reporting_running(client,
                                                                       monkeypatch,
                                                                       stubs):
    """Cooperative cancel: the run is still running until the worker sees it."""
    stub = _StubSociety(block=True)
    stubs.append(stub)
    _install(monkeypatch, stub)

    body = _submit(client, "running")
    run_id = body["runId"]
    _wait_for(lambda: _pipeline_started(run_id), "the pipeline to start")

    res = client.post("/api/society/runs/%s/cancel" % run_id,
                      headers=AUTH_HEADERS)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "running", body
    assert body["cancel_requested"] is True, body
    assert body["outcome"] == "cancel_requested", body

    # The worker observes the flag and reports the run cancelled.
    stub.stop()
    _wait_for(lambda: _status(run_id) == "cancelled",
              "the worker to observe the cancel")


def test_the_admission_limit_still_refuses_the_third_concurrent_run(client,
                                                                    monkeypatch,
                                                                    stubs):
    """Negative test: the refusal is the behaviour, not the bug."""
    stub = _StubSociety(block=True)
    stubs.append(stub)
    _install(monkeypatch, stub)

    first = _submit(client, "one")
    second = _submit(client, "two")
    assert first["status"] == "started", first
    assert second["status"] == "started", second

    third = _submit(client, "three")
    assert third["status"] == "busy", third
    assert "active run limit reached (2/2)" in third["error"], third
    # A refusal is a refusal: no run is started and no result is invented.
    assert "runId" not in third, third


def test_the_cancel_route_is_mounted_at_both_prefixes(client):
    """The dispatcher is mounted at /api and /api/v1; the route follows."""
    spec = client.get("/openapi.json", headers=AUTH_HEADERS).json()
    paths = spec.get("paths", {})
    for prefix in ("/api", "/api/v1"):
        route = paths.get("%s/society/runs/{run_id}/cancel" % prefix, {})
        assert "post" in route, "missing POST %s/society/runs/{run_id}/cancel" % prefix
