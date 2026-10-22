"""The benchmark history has to work through the route, not only the module.

`backend/benchmarking/run_history.py` has its own contract test
(`test_benchmark_history_contract.py`) for what it records, refuses and
compares. What that cannot show is whether the *control plane* reaches it:
a module that is perfectly correct and never called is a feature that does
not exist.

So these tests drive the dispatcher functions directly -- the same functions
the FastAPI routes are built from -- and check the whole chain:

    POST /benchmarks/run -> record_benchmark_run -> SQLite
    GET  /benchmarks/history -> history_payload -> the rows that were written
    GET  /benchmarks/compare -> compare_payload -> a real delta

The executor is stubbed. That is deliberate and narrow: what is under test
is the wiring and the storage path, not the measurement. The measurement
itself is covered by the live benchmark suites, and a stubbed executor
cannot make the storage layer report a score it did not receive.

The database is a temporary file, so the test neither reads nor writes the
developer's own run history.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """The dispatcher, pointed at a throwaway database and a stub executor."""
    monkeypatch.setenv("MECH_STORAGE_DB", str(tmp_path / "mech.db"))

    from backend.api import dispatcher
    from backend.validation import benchmark_runner

    calls: list[str] = []

    def fake_run_benchmark(self, benchmark_id="bench_default", n_prompts=6):
        calls.append(benchmark_id)
        return {
            "benchmark_id": benchmark_id,
            "status": "completed",
            "provenance": "live",
            "attested": True,
            "accuracy": 0.75,
            "robustness_score": 0.5,
            "eval_samples": n_prompts,
            "n_prompts": n_prompts,
        }

    monkeypatch.setattr(benchmark_runner.MechanisticBenchmarkRunner,
                        "run_benchmark", fake_run_benchmark)
    return dispatcher, calls


def test_a_completed_live_run_is_persisted_and_readable(wired):
    """The whole chain: run -> recorded -> readable as history."""
    dispatcher, calls = wired

    body = dispatcher.run_benchmark({"benchmark_name": "IOI",
                                     "n_prompts": 8})

    assert calls == ["bench_IOI"]
    assert body["status"] == "completed"
    assert body["provenance"] == "live"
    assert body["score"] == 0.75
    assert body["recorded"] is True
    assert body["run_id"], "a recorded run must carry the id it was stored under"
    assert body["field_provenance"]["run_id"] == "live"

    history = dispatcher.benchmark_history(benchmark="IOI", limit=5)
    assert history["count"] == 1
    row = history["runs"][0]
    assert row["run_id"] == body["run_id"]
    assert row["primary_score"] == 0.75
    assert row["status"] == "completed"
    assert row["provenance"] == "live"
    assert row["created_at"]


def test_two_recorded_runs_can_be_compared(wired):
    """The comparison endpoint reads the rows it wrote."""
    dispatcher, _ = wired

    first = dispatcher.run_benchmark({"benchmark_name": "IOI",
                                      "n_prompts": 8})
    second = dispatcher.run_benchmark({"benchmark_name": "IOI",
                                       "n_prompts": 8})

    comparison = dispatcher.benchmark_compare(first["run_id"],
                                              second["run_id"])
    assert comparison["same_benchmark"] is True
    assert comparison["primary_metric"] == "score"
    assert comparison["score_delta"] == 0.0
    assert comparison["comparable"] is True


def test_a_run_that_was_not_measured_is_not_persisted(wired, monkeypatch):
    """A refusal is never stored as a run, and never as a zero."""
    dispatcher, _ = wired
    from backend.validation import benchmark_runner

    def unavailable(self, benchmark_id="bench_default", n_prompts=6):
        return {"benchmark_id": benchmark_id, "status": "unavailable",
                "provenance": "unavailable", "reason": "no executor"}

    monkeypatch.setattr(benchmark_runner.MechanisticBenchmarkRunner,
                        "run_benchmark", unavailable)

    body = dispatcher.run_benchmark({"benchmark_name": "IOI",
                                     "n_prompts": 8})

    assert body["status"] == "unavailable"
    assert body["provenance"] == "unavailable"
    # The route's refusal branch does not carry the `recorded`/`run_id` keys
    # the success branch sets, so this asserts the absence of a stored run
    # rather than a flag: what matters is that nothing was written.
    assert body.get("recorded") is not True
    assert body.get("run_id") is None

    history = dispatcher.benchmark_history(benchmark="IOI", limit=5)
    assert history["count"] == 0


def test_an_unknown_run_id_is_a_404_not_a_delta(wired):
    dispatcher, _ = wired
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        dispatcher.benchmark_compare("no-such-run", "also-missing")
    assert excinfo.value.status_code == 404
