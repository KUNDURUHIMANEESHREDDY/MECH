"""Contract tests for the benchmark run-history surface (Lane L2 / MR-2).

The property under test is the project's non-negotiable rule: a missing
measurement is ``None``/NULL, never ``0.0``, and a refusal is never persisted
as a run. Each test below therefore asserts *what is stored*, not merely that a
call succeeded.

The database is a fresh file per test under ``tmp_path``, selected through
``MECH_STORAGE_DB`` -- the convention ``test_broken_and_leaking_methods.py``
uses -- because ``backend.storage.get_default_db_path()`` is read at call time
and the module-level wrappers build a store per call.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.benchmarking import run_history as RH
from backend.storage import DesktopStorage, StorageError

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _temp_database(tmp_path, monkeypatch):
    """Point every run-history call at a per-test database file."""
    monkeypatch.setenv("MECH_STORAGE_DB", str(tmp_path / "history.db"))
    yield


def _completed(score: float = 0.75, **extra) -> dict:
    """A minimal result shaped like ``POST /api/benchmarks/run``'s response."""
    result = {
        "status": "completed",
        "provenance": "live",
        "score": score,
        "pass_rate": 0.5,
        "eval_samples": 100,
    }
    result.update(extra)
    return result


def _row(run_id: str) -> dict:
    rows = [r for r in RH.history_payload(limit=500)["runs"] if r["run_id"] == run_id]
    assert len(rows) == 1, f"expected exactly one row for {run_id}, got {len(rows)}"
    return rows[0]


# -- round trip ------------------------------------------------------------- #

def test_a_recorded_run_round_trips_with_its_real_score():
    stored = RH.record_benchmark_run(
        benchmark_name="IOI",
        result=_completed(
            score=0.7325,
            model_id="gpt2-small",
            weights_fingerprint="sha256:deadbeef",
            git_commit="abc1234",
            n_prompts=100,
        ),
    )
    assert stored is not None, "a completed live run with a real score must be recorded"
    assert stored["primary_score"] == 0.7325
    assert stored["benchmark"] == "IOI"
    assert stored["primary_metric"] == "score"
    assert stored["status"] == "completed"
    assert stored["provenance"] == "live"
    assert stored["model_id"] == "gpt2-small"
    assert stored["weights_fingerprint"] == "sha256:deadbeef"
    assert stored["git_commit"] == "abc1234"
    assert stored["detail"]["pass_rate"] == 0.5
    assert stored["detail"]["eval_samples"] == 100
    assert stored["detail"]["n_prompts"] == 100

    # And the value read back through the read model is the same number.
    assert _row(stored["run_id"])["primary_score"] == 0.7325


def test_history_payload_reports_the_rows_it_read():
    RH.record_benchmark_run(benchmark_name="IOI", result=_completed(score=0.25))
    RH.record_benchmark_run(benchmark_name="SAE", result=_completed(score=0.5))

    everything = RH.history_payload()
    assert everything["count"] == 2
    assert len(everything["runs"]) == 2
    assert everything["benchmark"] is None
    assert everything["provenance"] == "live"
    assert {r["benchmark"] for r in everything["runs"]} == {"IOI", "SAE"}

    filtered = RH.history_payload("IOI")
    assert filtered["count"] == 1
    assert filtered["benchmark"] == "IOI"
    assert filtered["runs"][0]["benchmark"] == "IOI"
    assert filtered["runs"][0]["primary_score"] == 0.25


def test_field_provenance_never_claims_a_field_the_rows_do_not_carry():
    RH.record_benchmark_run(benchmark_name="IOI", result=_completed(score=0.4))

    payload = RH.history_payload("IOI")
    map_ = payload["field_provenance"]
    # The run has a score; a column nothing populated must not be called live.
    assert map_["primary_score"] == "live"
    assert map_["weights_fingerprint"] == "unavailable"
    assert map_["git_commit"] == "unavailable"
    assert set(map_.values()) <= {"live", "unavailable"}


# -- a missing measurement is never a zero ---------------------------------- #

def test_a_null_score_round_trips_as_none_never_zero():
    """The core rule: NULL comes back as None, not 0.0.

    Written straight through storage, because ``record_benchmark_run`` refuses
    a scoreless live result -- this pins the layer underneath that refusal.
    """
    DesktopStorage(Path(_db_path())).initialize()
    store_path = Path(_db_path())
    store = DesktopStorage(store_path)
    store.initialize()
    store.record_run(
        run_id="null-score",
        benchmark="IOI",
        primary_metric="score",
        primary_score=None,
        status="completed",
        provenance="live",
    )

    row = store.history("IOI", limit=5)[0]
    assert row["primary_score"] is None, (
        "a NULL primary_score came back as "
        f"{row['primary_score']!r}; a missing measurement must never be 0.0")
    assert row["primary_score"] != 0.0

    # It also reads back as NULL from SQLite itself, not as 0.
    with sqlite3.connect(store_path) as connection:
        raw = connection.execute(
            "SELECT primary_score FROM benchmark_runs WHERE run_id = ?",
            ("null-score",),
        ).fetchone()[0]
    assert raw is None, f"the stored column holds {raw!r}, not NULL"

    # And through the read model it survives as None.
    payload = RH.history_payload("IOI")
    assert payload["runs"][0]["primary_score"] is None


def test_a_scoreless_live_result_is_not_recorded():
    """No real score means no row -- and specifically not a zero-score row."""
    assert RH.record_benchmark_run(
        benchmark_name="IOI",
        result={"status": "completed", "provenance": "live"},
    ) is None
    assert RH.record_benchmark_run(
        benchmark_name="IOI",
        result=_completed(score=None),
    ) is None
    assert RH.history_payload()["runs"] == []


# -- negative controls: a refusal is never a run ----------------------------- #

@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "unavailable"),
        ("status", "error"),
        ("status", "failed"),
        ("status", "NOT_EXECUTED"),
        ("provenance", "unavailable"),
        ("provenance", "seeded"),
        ("provenance", "reference"),
    ],
)
def test_a_non_live_or_unfinished_result_is_not_recorded(field, value):
    result = _completed(score=0.9)
    result[field] = value
    assert RH.record_benchmark_run(benchmark_name="IOI", result=result) is None, (
        f"a {field}={value!r} result was persisted as a run")
    assert RH.history_payload()["runs"] == [], (
        "a refusal left a row in the history")


def test_a_malformed_result_is_refused_not_coerced():
    assert RH.record_benchmark_run(benchmark_name="IOI", result=None) is None  # type: ignore[arg-type]
    assert RH.record_benchmark_run(benchmark_name="IOI", result=["completed"]) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_score",
    [
        True,           # bool is an int subclass: 1.0 would be fabricated
        False,
        "0.75",         # a string is not a measurement
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_a_non_finite_or_non_numeric_score_is_refused(bad_score):
    result = _completed()
    result["score"] = bad_score
    assert RH.record_benchmark_run(benchmark_name="IOI", result=result) is None, (
        f"score={bad_score!r} was accepted as a measurement")
    assert RH.history_payload()["runs"] == []


def test_a_score_of_zero_from_a_real_measurement_is_recorded():
    """0.0 is rejected as a *placeholder*, not as a measurement.

    A live run that genuinely measured zero is a number and must be stored.
    """
    stored = RH.record_benchmark_run(
        benchmark_name="IOI", result=_completed(score=0.0))
    assert stored is not None
    assert stored["primary_score"] == 0.0


def test_recording_is_idempotent_on_run_id():
    first = RH.record_benchmark_run(
        benchmark_name="IOI", result=_completed(score=0.4, run_id="fixed-run"))
    second = RH.record_benchmark_run(
        benchmark_name="IOI", result=_completed(score=0.9, run_id="fixed-run"))

    assert first is not None and second is not None
    assert first["run_id"] == second["run_id"] == "fixed-run"
    runs = RH.history_payload()["runs"]
    assert len(runs) == 1, f"a re-record appended a duplicate: {len(runs)} rows"
    assert runs[0]["primary_score"] == 0.9, "the overwrite did not land"


# -- history clamps ---------------------------------------------------------- #

@pytest.mark.parametrize("limit,expected", [(0, 1), (-5, 1), (10_000, 3), (2, 2)])
def test_history_limit_is_clamped_not_honoured_blindly(limit, expected):
    for score in (0.1, 0.2, 0.3):
        RH.record_benchmark_run(benchmark_name="IOI", result=_completed(score=score))

    payload = RH.history_payload(limit=limit)
    assert payload["count"] == expected
    assert len(payload["runs"]) == expected


def test_a_non_integer_limit_is_a_client_error_not_a_500():
    """A malformed limit is a 400, and is never quietly answered with [].

    The wrapper raises its own typed error and carries the HTTP status the
    control plane should use, so the route does not have to read the message.
    """
    with pytest.raises(RH.RunHistoryError) as excinfo:
        RH.history_payload(limit="many")  # type: ignore[arg-type]

    assert excinfo.value.status_code == 400
    assert "limit must be an integer" in str(excinfo.value)


# -- compare ----------------------------------------------------------------- #

def test_comparing_two_runs_on_different_benchmarks_is_not_comparable():
    first = RH.record_benchmark_run(
        benchmark_name="IOI", result=_completed(score=0.2))
    second = RH.record_benchmark_run(
        benchmark_name="SAE", result=_completed(score=0.8))
    assert first is not None and second is not None

    payload = RH.compare_payload(first["run_id"], second["run_id"])
    assert payload["same_benchmark"] is False
    assert payload["score_delta"] is None, (
        "a delta was reported across two different benchmarks")
    assert payload["comparable"] is False
    assert payload["not_comparable"], "the reason a delta is absent was dropped"
    assert any("different benchmarks" in reason
               for reason in payload["not_comparable"]), payload["not_comparable"]
    assert payload["score_delta_reason"]


def test_comparing_two_runs_on_the_same_benchmark_reports_a_real_delta():
    first = RH.record_benchmark_run(
        benchmark_name="IOI", result=_completed(score=0.2))
    second = RH.record_benchmark_run(
        benchmark_name="IOI", result=_completed(score=0.5))
    assert first is not None and second is not None

    payload = RH.compare_payload(first["run_id"], second["run_id"])
    assert payload["comparable"] is True
    # delta is run_b - run_a, computed by storage, not by this wrapper.
    assert payload["score_delta"] == pytest.approx(0.3)
    assert payload["same_benchmark"] is True


def test_comparing_a_run_with_a_null_score_reports_no_delta():
    DesktopStorage(Path(_db_path())).initialize()
    store = DesktopStorage(Path(_db_path()))
    store.initialize()
    store.record_run(run_id="a", benchmark="IOI", primary_metric="score",
                     primary_score=0.5, status="completed", provenance="live")
    store.record_run(run_id="b", benchmark="IOI", primary_metric="score",
                     primary_score=None, status="completed", provenance="live")

    payload = RH.compare_payload("a", "b")
    assert payload["score_delta"] is None, (
        "a NULL score produced a numeric delta")
    assert payload["comparable"] is False
    assert any("primary score not measured" in reason
               for reason in payload["not_comparable"]), payload["not_comparable"]


@pytest.mark.parametrize("missing_a,missing_b", [("ghost-a", "ghost-b")])
def test_comparing_an_unknown_run_id_raises_an_addressable_error(missing_a, missing_b):
    RH.record_benchmark_run(benchmark_name="IOI", result=_completed(score=0.1))

    with pytest.raises(RH.RunHistoryError) as excinfo:
        RH.compare_payload(missing_a, missing_b)

    assert excinfo.value.run_id in {missing_a, missing_b}, (
        "the error does not say which run id was unknown, so the route cannot "
        "answer 404")
    assert excinfo.value.status_code == 404
    assert not isinstance(excinfo.value, StorageError)


# -- the module's own surface ------------------------------------------------ #

def test_run_history_does_not_build_a_second_store():
    """The history surface must read the one storage authority.

    A second SQLite handle or a copied schema here would make the history panel
    disagree with the rest of the platform.
    """
    source = (REPO_ROOT / "backend" / "benchmarking" / "run_history.py")
    text = source.read_text(encoding="utf-8")
    assert "sqlite3" not in text.replace("no second store", "")
    assert "CREATE TABLE" not in text
    assert "from backend.storage import" in text


def test_every_refusal_leaves_the_table_empty():
    """One sweep: no refusal path may persist anything."""
    refusals = [
        {"status": "unavailable", "provenance": "unavailable"},
        {"status": "error", "provenance": "live", "score": 0.5},
        {"status": "completed", "provenance": "seeded", "score": 0.5},
        {"status": "completed", "provenance": "live"},
        {"status": "completed", "provenance": "live", "score": "0.5"},
        {"status": "NOT_EXECUTED", "provenance": "unavailable"},
    ]
    for result in refusals:
        assert RH.record_benchmark_run(
            benchmark_name="IOI", result=result) is None, result
    assert RH.history_payload()["runs"] == []


def _db_path() -> str:
    """The database this test module pointed the store at."""
    from backend.storage import get_default_db_path

    return str(get_default_db_path())
