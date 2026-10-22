"""A run whose evidence was not recorded must not report a plain success.

``_society_persist_record`` used to swallow every exception::

    try:
        save_run_record(...)
    except Exception:
        pass

while the run's status was set to ``completed`` regardless. A run could
therefore be reported as completed with NO durable evidence record behind
it -- a success with nothing that proves it.

The failure is now reported: the write returns False, the exception is
logged, and a successful run is downgraded to ``completed_unrecorded`` with
the reason. The status is still terminal, so the run still releases its
admission slot -- it is simply not a success anyone can cite.
"""

from __future__ import annotations

import os
import queue
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def dispatcher():
    from backend.api import dispatcher as core
    return core


def _run(core, run_id: str, status: str) -> dict:
    record = {
        "run_id": run_id,
        "status": status,
        "goal": "prove the point",
        "model_name": "gpt2",
        "result": {"status": status},
        "reason": "",
        "queue": queue.Queue(),
        "created": 0.0,
        "events": [],
    }
    core._society_runs[run_id] = record
    return record


def test_a_successful_evidence_write_reports_success(dispatcher, monkeypatch):
    from backend.core import evidence_graph
    written = {}

    def fake_save(run_id, record, directory=None):
        written["run_id"] = run_id
        return str(run_id)

    monkeypatch.setattr(evidence_graph, "save_run_record", fake_save)
    assert dispatcher._society_persist_record(
        "r_test_written", "goal", "gpt2", "completed", {"status": "completed"}) is True
    assert written["run_id"] == "r_test_written"


def test_a_failed_evidence_write_is_reported_not_swallowed(dispatcher, monkeypatch):
    from backend.core import evidence_graph

    def boom(run_id, record, directory=None):
        raise OSError("the evidence store is not writable")

    monkeypatch.setattr(evidence_graph, "save_run_record", boom)
    assert dispatcher._society_persist_record(
        "r_test_failed", "goal", "gpt2", "completed", {"status": "completed"}) is False


def test_a_completed_run_without_evidence_is_not_reported_as_completed(dispatcher, monkeypatch):
    """The run happened; the proof of it did not. It says so."""
    from backend.core import evidence_graph

    def boom(run_id, record, directory=None):
        raise OSError("the evidence store is not writable")

    monkeypatch.setattr(evidence_graph, "save_run_record", boom)
    run_id = "r_test_unrecorded"
    record = _run(dispatcher, run_id, "completed")
    try:
        assert dispatcher._society_persist_record(
            run_id, "goal", "gpt2", "completed", record["result"]) is False
        dispatcher._society_mark_unrecorded(run_id, "completed")

        assert dispatcher._society_runs[run_id]["status"] == "completed_unrecorded"
        assert "evidence" in dispatcher._society_runs[run_id]["reason"]
        assert dispatcher._society_runs[run_id]["result"]["status"] == \
            "completed_unrecorded"
    finally:
        dispatcher._society_runs.pop(run_id, None)


def test_the_unrecorded_status_is_terminal(dispatcher):
    """It has to release the admission slot, like every other end state."""
    assert dispatcher._society_is_terminal("completed_unrecorded") is True
    assert dispatcher._society_is_terminal("completed") is True
    assert dispatcher._society_is_terminal("running") is False


def test_a_failed_run_is_not_downgraded(dispatcher, monkeypatch):
    """A run that already failed is honest; the evidence failure adds nothing."""
    from backend.core import evidence_graph

    def boom(run_id, record, directory=None):
        raise OSError("the evidence store is not writable")

    monkeypatch.setattr(evidence_graph, "save_run_record", boom)
    run_id = "r_test_error_not_downgraded"
    _run(dispatcher, run_id, "error")
    try:
        dispatcher._society_mark_unrecorded(run_id, "error")
        assert dispatcher._society_runs[run_id]["status"] == "error"
    finally:
        dispatcher._society_runs.pop(run_id, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
