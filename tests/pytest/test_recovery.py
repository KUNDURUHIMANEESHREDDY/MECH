"""Tests for failure recovery: timeouts, retries, cleanup, and the event stream.

The properties proved here are the ones the platform depends on when
something goes wrong:

* a call that overruns its budget is a failure, not a slow success;
* a job that overruns its budget is ``failed`` -- never ``completed``;
* a failed write leaves no partial file behind;
* the event stream carries the failure, with its reason.
"""

import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core import observability as obs  # noqa: E402
from backend.core.recovery import (  # noqa: E402
    EngineFailure,
    EngineTimeout,
    call_with_timeout,
    cleanup_partial_files,
    failure_payload,
    retry_call,
)


@pytest.fixture()
def restored_settings():
    """Restore the process-wide settings after a test changes them."""
    saved = dict(os.environ)
    try:
        yield obs
    finally:
        obs.reload_settings(saved)


# --------------------------------------------------------------------------- #
# timeouts and retries
# --------------------------------------------------------------------------- #


def test_a_call_that_overruns_its_budget_is_a_failure():
    """A timeout raises. It never returns a value that looks like a result."""
    def _slow():
        time.sleep(5)
        return {"status": "completed"}

    with pytest.raises(EngineTimeout) as excinfo:
        call_with_timeout(_slow, 0.25, label="slow call")
    assert "slow call" in str(excinfo.value)


def test_a_call_that_finishes_returns_its_value():
    assert call_with_timeout(lambda: 42, 5.0, label="quick call") == 42


def test_a_retry_is_bounded_and_stops_at_the_first_success():
    calls = []

    def _flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("not yet")
        return "ok"

    result, attempts = retry_call(_flaky, attempts=5, backoff_s=0.0,
                                  label="flaky")
    assert result == "ok"
    assert attempts == 3
    assert len(calls) == 3


def test_a_retry_gives_up_and_reports_the_last_failure():
    def _always_fails():
        raise RuntimeError("still broken")

    with pytest.raises(RuntimeError):
        retry_call(_always_fails, attempts=3, backoff_s=0.0, label="broken")


def test_a_timeout_is_not_retried():
    """The thread that overran may still be running; a second attempt races it."""
    calls = []

    def _slow():
        calls.append(1)
        time.sleep(5)
        return "late"

    with pytest.raises(EngineTimeout):
        retry_call(lambda: call_with_timeout(_slow, 0.2, label="slow"),
                   attempts=3, backoff_s=0.0, label="slow")
    assert len(calls) == 1


def test_a_failure_payload_cannot_be_mistaken_for_a_success():
    payload = failure_payload("the engine blew up")
    assert payload["status"] == "failed"
    assert payload["measured"] is False
    assert payload["provenance"] == "unavailable"
    assert "completed" not in payload["status"]
    assert "ok" not in payload["status"]


def test_an_engine_failure_carries_its_reason():
    error = EngineFailure("torch is not installed")
    assert "torch is not installed" in str(error)


# --------------------------------------------------------------------------- #
# cleanup
# --------------------------------------------------------------------------- #


def test_cleanup_removes_the_partial_files_and_never_raises(tmp_path):
    partial = tmp_path / "job.json.tmp"
    partial.write_text("half written", encoding="utf-8")
    keep = tmp_path / "job.json"
    keep.write_text("{}", encoding="utf-8")

    removed = cleanup_partial_files([partial, tmp_path / "missing.tmp"])

    assert removed == [str(partial)]
    assert not partial.exists()
    # A file the caller did not name is untouched: the caller knows which
    # paths are partial, this function only removes what it is given.
    assert keep.exists()


def test_cleanup_ignores_a_path_it_cannot_remove(tmp_path):
    """A cleanup failure must not replace the real error."""
    directory = tmp_path / "a-directory"
    directory.mkdir()
    assert cleanup_partial_files([directory, None, 17]) == []


# --------------------------------------------------------------------------- #
# the job queue
# --------------------------------------------------------------------------- #


def test_a_job_that_overruns_its_budget_is_failed_not_completed(
        tmp_path, monkeypatch, restored_settings):
    """The run must not be left showing "completed"."""
    from backend.api import runtime_api

    monkeypatch.setenv("MECH_V2_JOB_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MECH_JOB_TIMEOUT_S", "0.25")
    restored_settings.reload_settings()

    def _slow(payload, stop):
        time.sleep(5)
        return {"status": "completed"}

    monkeypatch.setitem(runtime_api.HANDLERS, "timeout_probe", _slow)

    submitted = runtime_api.QUEUE.submit("timeout_probe", {})
    job_id = submitted["job_id"]

    deadline = time.time() + 30
    record = None
    while time.time() < deadline:
        record = runtime_api.QUEUE.get(job_id)
        if record and record.get("status") in runtime_api.TERMINAL:
            break
        time.sleep(0.05)

    assert record is not None, "the job never reached a terminal state"
    assert record["status"] == "failed", record
    assert "timeout" in str(record.get("error") or "").lower()
    assert record.get("result") is None


def test_a_failed_persist_leaves_no_partial_file(tmp_path, monkeypatch):
    """A failed write must not leave a half-written record behind."""
    from backend.api import runtime_api

    monkeypatch.setenv("MECH_V2_JOB_DIR", str(tmp_path))

    def _boom(*args, **kwargs):
        raise OSError("the disk is full")

    monkeypatch.setattr(runtime_api.json, "dump", _boom)

    queue = runtime_api._JobQueue()
    job = {"job_id": "jpartial", "kind": "probe", "status": "queued"}
    queue._persist(job)

    assert job.get("persist_error"), "the persistence failure was not recorded"
    assert list(tmp_path.glob("*.tmp")) == [], "a partial file was left behind"


# --------------------------------------------------------------------------- #
# the event stream
# --------------------------------------------------------------------------- #


def test_the_event_stream_carries_the_failure_and_its_reason():
    """A failed job is an event the UI can show, with the reason attached."""
    from backend.api import events

    event = events.HUB.observe({
        "job_id": "jfailed000001",
        "status": "failed",
        "error": "RuntimeError: the engine raised",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
    })

    assert event is not None
    assert event["type"] == "job.failed"
    assert event["message"].startswith("job failed")
    assert "the engine raised" in event["message"]
    assert event["data"]["error"].startswith("RuntimeError")


def test_a_partial_failure_is_not_reported_as_a_clean_success():
    """A sweep that could not measure everything says so."""
    from backend.api import events

    event = events.HUB.observe({
        "job_id": "jpartial00001",
        "status": "completed",
        "result": {
            "status": "completed_with_unavailable",
            "measured": 1,
            "unavailable": 2,
        },
    })

    assert event is not None
    assert "unavailable" in event["message"]
    assert event["data"]["result"]["status"] == "completed_with_unavailable"
