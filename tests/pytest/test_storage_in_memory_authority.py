"""`MECH_STORAGE_DB=":memory:"` must behave like a database, not like a fresh one
per connection.

The defect this pins
--------------------
Several test modules set the sentinel process-wide
(``os.environ.setdefault("MECH_STORAGE_DB", ":memory:")`` in
`test_api_exposure.py`, `test_runtime_controls.py`, `test_ipc_backend_contract.py`).
SQLite gives every connection its own private database for a bare ``:memory:``,
so `DesktopStorage.initialize()` created the schema on one connection and every
later query opened a different, empty database:

    sqlite3.OperationalError: no such table: settings

Every storage-backed route therefore returned 500 under the sentinel --
``/api/experiments``, ``/api/sessions``, ``/api/settings``, ``/api/recent-files``
and ``/api/benchmarks/history``. It went unnoticed because the control plane
resolves ``MECH_STORAGE_DB`` once at import, before the sentinel is set, while
``benchmarking.run_history`` resolves it at call time, so only the newest route
was exposed by the authorization matrix.

These assert the behaviour a caller bargains for, so they fail loudly if the
redirection in `DesktopStorage._connect` is ever removed.

Note on shared state: the in-memory database is deliberately one database per
process, so a few of these tests are written against membership rather than an
exact row count. Each uses a benchmark name of its own.
"""

from __future__ import annotations

import sqlite3

from backend.storage import DesktopStorage
from backend.storage.database import (
    MEMORY_DB_PATH,
    MEMORY_DB_URI,
    is_memory_path,
)


def test_the_memory_sentinel_is_recognised_and_not_mistaken_for_a_file(tmp_path):
    """The redirection must not capture ordinary paths, or every store in the
    process would silently become one shared in-memory database."""
    assert is_memory_path(MEMORY_DB_PATH) is True
    assert is_memory_path(":memory:") is True
    assert is_memory_path(tmp_path / "real.db") is False
    assert is_memory_path(":memoryx:") is False
    assert is_memory_path("C:/tmp/mech.db") is False


def test_two_memory_stores_address_one_database():
    """The point of the fix: a schema and its rows outlive one store instance.

    Two *separate* `DesktopStorage(":memory:")` instances stand in for the two
    authorities that disagreed in production -- the control plane's import-time
    store and `run_history`'s call-time store.
    """
    writer = DesktopStorage(MEMORY_DB_PATH)
    writer.initialize()
    written = writer.record_run(
        benchmark="memory-authority-probe",
        status="completed",
        provenance="live",
        primary_score=0.25,
    )

    reader = DesktopStorage(MEMORY_DB_PATH)
    rows = reader.history(benchmark="memory-authority-probe", limit=50)

    by_id = {row["run_id"]: row for row in rows}
    assert written["run_id"] in by_id, (
        "a row written through one in-memory store must be visible to another; "
        "if this fails each store got its own private database"
    )
    assert by_id[written["run_id"]]["primary_score"] == 0.25


def test_memory_storage_serves_every_table_the_routes_need(tmp_path):
    """The four routes that were already returning 500 under the sentinel.

    Before the redirection each of these raised ``no such table``.
    """
    store = DesktopStorage(MEMORY_DB_PATH)
    store.initialize()

    # backs GET /api/settings
    settings = store.get_settings()
    assert settings["theme"] in {"system", "light", "dark"}

    # backs GET /api/experiments  (the item payload requires a string 'id')
    store.add_experiment({"id": "mem-exp-1", "name": "probe"})
    experiment_ids = {
        item.get("id") or item.get("item_id") for item in store.list_experiments()
    }
    assert "mem-exp-1" in experiment_ids

    # backs GET /api/sessions
    store.add_session({"id": "mem-session-1", "name": "probe"})
    session_ids = {
        item.get("id") or item.get("item_id") for item in store.list_sessions()
    }
    assert "mem-session-1" in session_ids

    # backs GET /api/recent-files
    probe = str(tmp_path / "probe.txt")
    store.add_recent_file(path=probe)
    assert probe in {item["path"] for item in store.list_recent_files()}


def test_the_null_score_contract_holds_for_memory_storage_too():
    """The fix must not smuggle a placeholder into the newest table.

    A `:memory:` store is still a real store: a run with no score keeps None.
    """
    store = DesktopStorage(MEMORY_DB_PATH)
    store.initialize()
    stored = store.record_run(
        benchmark="memory-null-score-probe",
        status="completed",
        provenance="live",
        primary_score=None,
    )
    assert stored["primary_score"] is None, (
        "an absent measurement must round-trip as None, never as 0.0"
    )
    rows = store.history(benchmark="memory-null-score-probe", limit=1)
    assert rows[0]["primary_score"] is None


def test_the_memory_database_outlives_a_closed_connection():
    """A shared in-memory database dies with its last connection, so the module
    holds one open. Without that, the schema would vanish between two calls."""
    store = DesktopStorage(MEMORY_DB_PATH)
    store.initialize()

    # Open and explicitly close a connection, the way a short-lived caller would.
    connection = store._connect()
    connection.close()

    # A brand-new connection must still find the schema.
    fresh = sqlite3.connect(MEMORY_DB_URI, uri=True)
    try:
        names = {
            row[0]
            for row in fresh.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        fresh.close()

    assert "settings" in names
    assert "benchmark_runs" in names


def test_run_history_resolves_the_memory_sentinel_without_raising(monkeypatch):
    """The exact failure the authorization matrix caught.

    `run_history._store()` resolves `MECH_STORAGE_DB` at call time, so it saw the
    sentinel even when the control plane's import-time store did not, and every
    call raised ``no such table: benchmark_runs`` -> HTTP 500.
    """
    monkeypatch.setenv("MECH_STORAGE_DB", MEMORY_DB_PATH)

    from backend.benchmarking.run_history import history_payload, record_benchmark_run

    recorded = record_benchmark_run(
        benchmark_name="memory-run-history-probe",
        result={
            "status": "completed",
            "provenance": "live",
            "score": 0.75,
            "pass_rate": 0.5,
            "eval_samples": 4,
        },
    )
    assert recorded is not None, (
        "a completed live run with a finite score must be recorded"
    )
    assert recorded["primary_score"] == 0.75

    payload = history_payload(benchmark="memory-run-history-probe", limit=50)
    run_ids = {row["run_id"] for row in payload["runs"]}
    assert recorded["run_id"] in run_ids
    assert payload["provenance"] == "live"


def test_run_history_still_refuses_a_non_live_result_against_memory_storage(
    monkeypatch,
):
    """The fix must not weaken the recording contract on the way past.

    A refusal has no measurement, so it must still write nothing -- the
    difference between "recording works now" and "anything gets recorded".
    """
    monkeypatch.setenv("MECH_STORAGE_DB", MEMORY_DB_PATH)

    from backend.benchmarking.run_history import history_payload, record_benchmark_run

    assert record_benchmark_run(
        benchmark_name="memory-refusal-probe",
        result={"status": "unavailable", "provenance": "unavailable", "reason": "no"},
    ) is None
    assert record_benchmark_run(
        benchmark_name="memory-refusal-probe",
        result={"status": "completed", "provenance": "seeded", "score": 0.9},
    ) is None

    payload = history_payload(benchmark="memory-refusal-probe", limit=50)
    assert payload["count"] == 0, "a non-live run must leave the history empty"


def test_a_file_backed_store_is_untouched_by_the_memory_redirection(tmp_path):
    """The isolation guard.

    If the redirection ever captured file paths, every store in the process would
    silently share one in-memory database and both halves of this would fail.
    """
    file_store = DesktopStorage(tmp_path / "isolated.db")
    file_store.initialize()

    assert file_store.history(benchmark="memory-authority-probe", limit=50) == [], (
        "a file-backed store must not see rows written to the in-memory database"
    )

    file_store.record_run(
        benchmark="file-isolation-probe",
        status="completed",
        provenance="live",
        primary_score=0.1,
    )
    memory_store = DesktopStorage(MEMORY_DB_PATH)
    memory_store.initialize()
    assert memory_store.history(benchmark="file-isolation-probe", limit=50) == [], (
        "the in-memory store must not see rows written to a file-backed store"
    )
    assert [
        row["run_id"]
        for row in file_store.history(benchmark="file-isolation-probe", limit=50)
    ], "the file-backed store must still see its own row"
