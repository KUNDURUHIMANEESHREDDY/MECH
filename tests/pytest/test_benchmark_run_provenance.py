"""A stored benchmark run must say which weights and which code produced it.

`run_history.py` keeps three provenance columns -- `model_id`,
`weights_fingerprint` and `git_commit` -- and copies them from the result it
is handed, "never defaulted". That contract was true and useless at once:
the route narrowed the executor's record to scores and never carried those
three across, so every stored row arrived with them NULL. A run that cannot
say which weights produced it, and which code revision, is not evidence of
anything repeatable.

The second contract is the one the refusal branches broke. The success branch
answers with `run_id` and `recorded`; the refusal branches omitted both, so a
consumer doing `body["recorded"]` raised `KeyError` instead of learning that
nothing was recorded. The absence of a run is a fact, and a fact has to be
readable.

The executor is stubbed, exactly as in `test_benchmark_history_end_to_end.py`:
what is under test is the wiring and the storage path, not the measurement.
The measurement itself is covered by the live benchmark suites, and a stub
cannot make the storage layer report a score it did not receive.

The database is a temporary file, so the developer's own run history is
neither read nor written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _measurement_record(content_hash: str, source: str) -> dict:
    """A measurement record shaped like the one `attest_measurement` writes."""
    return {
        "model_identity": {
            "name": "gpt2-small",
            "content_hash": content_hash,
            "content_hash_source": source,
        },
        "input_commitment": {},
        "output_commitment": {},
        "execution_context": {},
    }


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """The dispatcher, a throwaway database, and a stubbed executor."""
    monkeypatch.setenv("MECH_STORAGE_DB", str(tmp_path / "mech.db"))

    from backend.api import dispatcher
    from backend.validation import benchmark_runner

    state: dict = {"result": None}

    def fake_run_benchmark(self, benchmark_id="bench_default", n_prompts=6):
        result = state["result"]
        if result is None:
            raise AssertionError("the test did not set state['result']")
        return dict(result, benchmark_id=benchmark_id)

    monkeypatch.setattr(benchmark_runner.MechanisticBenchmarkRunner,
                        "run_benchmark", fake_run_benchmark)
    return dispatcher, state


def _live_result(**overrides) -> dict:
    result = {
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "accuracy": 0.75,
        "robustness_score": 0.5,
        "eval_samples": 8,
        "n_prompts": 8,
        "model_id": "gpt2-small",
        "measurement_record": _measurement_record(
            "a" * 64, "model_weights"),
    }
    result.update(overrides)
    return result


def test_a_stored_run_carries_the_weights_and_the_revision(wired):
    """The row names the model, the weight bytes and the code revision."""
    dispatcher, state = wired
    state["result"] = _live_result()

    body = dispatcher.run_benchmark({"benchmark_name": "IOI", "n_prompts": 8})
    assert body["status"] == "completed"
    assert body["recorded"] is True

    history = dispatcher.benchmark_history(benchmark="IOI", limit=5)
    assert history["count"] == 1
    row = history["runs"][0]

    assert row["model_id"] == "gpt2-small"
    assert row["weights_fingerprint"] == "a" * 64
    # The real commit, not a placeholder. The repository is a git checkout,
    # so git can be asked; if it could not be, the column stays empty and
    # that is asserted separately below.
    assert row["git_commit"], (
        "a stored run must record the commit it was produced at")
    assert row["git_commit"] != "unknown"
    assert len(row["git_commit"]) >= 7
    assert all(c in "0123456789abcdef" for c in row["git_commit"])


def test_a_name_derived_fingerprint_is_refused(wired):
    """A hash of the model NAME proves nothing about the weights.

    `provenance.py` falls back to hashing the model name when no weight
    bytes were bound. Storing that in the fingerprint column would present
    a name-hash as if it identified the weights, so the column stays empty.
    """
    dispatcher, state = wired
    state["result"] = _live_result(
        measurement_record=_measurement_record(
            "b" * 32, "model_name_fallback"))

    body = dispatcher.run_benchmark({"benchmark_name": "IOI", "n_prompts": 8})
    assert body["recorded"] is True

    row = dispatcher.benchmark_history(benchmark="IOI", limit=5)["runs"][0]
    assert row["weights_fingerprint"] is None
    # The field is still reported -- as unavailable, not as absent.
    assert body["field_provenance"]["weights_fingerprint"] == "unavailable"


def test_a_refusal_reports_that_nothing_was_recorded(wired):
    """The refusal branch answers with the same keys as the success branch.

    A consumer reading `body["recorded"]` must not have to know which branch
    produced the response.
    """
    dispatcher, state = wired
    state["result"] = {
        "status": "unavailable",
        "provenance": "unavailable",
        "reason": "no live weights",
    }

    body = dispatcher.run_benchmark({"benchmark_name": "IOI", "n_prompts": 8})

    assert body["status"] == "unavailable"
    assert body["recorded"] is False
    assert body["run_id"] is None
    # And nothing was written: a refusal is never a row with a zeroed score.
    assert dispatcher.benchmark_history(benchmark="IOI", limit=5)["count"] == 0


def test_a_non_numeric_score_reports_that_nothing_was_recorded(wired):
    """The same contract on the other refusal branch."""
    dispatcher, state = wired
    state["result"] = {
        "status": "completed",
        "provenance": "live",
        "accuracy": "not a number",
    }

    body = dispatcher.run_benchmark({"benchmark_name": "IOI", "n_prompts": 8})

    assert body["status"] == "unavailable"
    assert body["recorded"] is False
    assert body["run_id"] is None
    assert dispatcher.benchmark_history(benchmark="IOI", limit=5)["count"] == 0
