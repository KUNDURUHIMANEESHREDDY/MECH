"""Persistence, migration and atomicity of the SQLite storage authority.

`backend/storage/database.py` is the project's only database handle
(`tests/pytest/test_storage_authority_consolidation.py` pins that). What it
does with the file it opens is covered here, because the interesting
properties are the ones a unit test on a single connection cannot see:

* a row written by one OS process is read by a different OS process;
* several OS processes writing the same file lose nothing;
* a migration that fails leaves the database exactly as it was, rather than
  half migrated;
* a migration that fails says which column it could not add.

The last two are the regression tests for a defect measured on this host: the
migration ran each `ALTER TABLE` in autocommit, so a database whose second
table could not be altered kept the first table's new column -- a file that
looked migrated and was not, and which no later run could tell apart from a
healthy one.

Nothing here touches `backend/storage/mech.db` or `backend/runtime.db`: every
test works on a file under pytest's own temporary directory.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.storage import DesktopStorage, StorageError  # noqa: E402

EXPECTED_TABLES = {
    "settings", "recent_projects", "recent_files",
    "experiments", "sessions", "plugins", "benchmark_runs",
}

#: A child process needs the repository on its path to import the authority.
CHILD_PREAMBLE = (
    f"import json\n"
    f"import sys\n"
    f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
    f"from backend.storage import DesktopStorage\n"
)


def _tables(path: Path) -> set[str]:
    connection = sqlite3.connect(str(path))
    try:
        names = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        connection.close()
    return {name for name in names if not name.startswith("sqlite_")}


def _columns(path: Path, table: str) -> list[str]:
    connection = sqlite3.connect(str(path))
    try:
        return [row[1] for row in connection.execute(
            f"PRAGMA table_info({table})")]
    finally:
        connection.close()


def _write_old_schema(path: Path) -> None:
    """A database as an older build left it: no `updated_at`, no
    `benchmark_runs`."""
    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(
            """
            CREATE TABLE experiments (
                item_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE sessions (
                item_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO experiments (item_id, payload, created_at) "
            "VALUES (?, ?, ?)",
            ("legacy-exp", json.dumps({"id": "legacy-exp"}),
             "2020-01-01T00:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO sessions (item_id, payload, created_at) "
            "VALUES (?, ?, ?)",
            ("legacy-ses", json.dumps({"id": "legacy-ses"}),
             "2020-01-01T00:00:00+00:00"),
        )
        connection.commit()
    finally:
        connection.close()


def _write_broken_schema(path: Path) -> None:
    """A database whose `sessions` is a view, so it cannot take a new
    column. `experiments` can, and it comes first in the migration."""
    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(
            """
            CREATE TABLE experiments (
                item_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE VIEW sessions AS
                SELECT item_id, payload, created_at FROM experiments;
            """
        )
        connection.commit()
    finally:
        connection.close()


def _run_child(tmp_path: Path, body: str, *args: str) -> str:
    """Run one statement block in a separate OS process."""
    script = tmp_path / "child.py"
    script.write_text(CHILD_PREAMBLE + textwrap.dedent(body),
                      encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True, timeout=120, cwd=str(tmp_path),
    )
    assert completed.returncode == 0, (
        f"child process failed ({completed.returncode}):\n"
        f"{completed.stdout}\n{completed.stderr}")
    return completed.stdout


# -- schema creation -------------------------------------------------------

def test_a_fresh_database_gets_the_whole_schema(tmp_path):
    store = DesktopStorage(tmp_path / "fresh.db")
    store.initialize()
    assert _tables(store.db_path) == EXPECTED_TABLES


def test_initialising_twice_changes_nothing(tmp_path):
    """The schema is created with `CREATE TABLE IF NOT EXISTS`, so the second
    run has to be a no-op -- and the data has to survive it."""
    store = DesktopStorage(tmp_path / "twice.db")
    store.initialize()
    store.add_experiment({"id": "kept", "name": "probe"})
    before = _tables(store.db_path)

    store.initialize()

    assert _tables(store.db_path) == before
    assert [item["id"] for item in store.list_experiments()] == ["kept"]


def test_the_database_is_in_wal_mode(tmp_path):
    """WAL is what lets a reader see a committed write while another process
    still holds the file open."""
    store = DesktopStorage(tmp_path / "wal.db")
    store.initialize()
    connection = sqlite3.connect(str(store.db_path))
    try:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        connection.close()
    assert mode == "wal"


# -- the migration path ----------------------------------------------------

def test_an_old_database_is_migrated_in_place_and_keeps_its_rows(tmp_path):
    old = tmp_path / "old.db"
    _write_old_schema(old)
    assert "updated_at" not in _columns(old, "experiments")

    DesktopStorage(old).initialize()

    assert "updated_at" in _columns(old, "experiments")
    assert "updated_at" in _columns(old, "sessions")
    assert "benchmark_runs" in _tables(old)
    # The rows that were there are still there...
    store = DesktopStorage(old)
    assert [item["id"] for item in store.list_experiments()] == ["legacy-exp"]
    # ...and their update time is seeded from their creation time, so the two
    # stamps stay ordered.
    connection = sqlite3.connect(str(old))
    try:
        row = connection.execute(
            "SELECT created_at, updated_at FROM experiments "
            "WHERE item_id = 'legacy-exp'").fetchone()
    finally:
        connection.close()
    assert row == ("2020-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00")


def test_migrating_twice_changes_nothing(tmp_path):
    old = tmp_path / "old-twice.db"
    _write_old_schema(old)
    DesktopStorage(old).initialize()
    before = {table: _columns(old, table) for table in _tables(old)}

    DesktopStorage(old).initialize()

    assert {table: _columns(old, table) for table in _tables(old)} == before


def test_a_failed_migration_leaves_the_database_untouched(tmp_path):
    """A migration that fails part way must not leave a half-migrated file.

    `sessions` is a view, so it cannot take a new column. `experiments` can,
    and it comes first in the migration. If the ALTERs run in autocommit --
    which is what Python's sqlite3 module does with DDL unless a transaction
    is opened by hand -- `experiments` keeps its new column and the file
    looks migrated while it is not.
    """
    broken = tmp_path / "broken.db"
    _write_broken_schema(broken)

    with pytest.raises(StorageError):
        DesktopStorage(broken).initialize()

    assert "updated_at" not in _columns(broken, "experiments"), (
        "the migration was not atomic: the first table was altered and "
        "committed even though the migration as a whole failed")


def test_a_failed_migration_says_which_column_failed(tmp_path):
    """The error has to name the table and the column, because 'cannot add a
    column to a view' and 'the disk is full' call for different repairs."""
    broken = tmp_path / "broken-reporting.db"
    _write_broken_schema(broken)

    with pytest.raises(StorageError) as caught:
        DesktopStorage(broken).initialize()

    message = str(caught.value)
    assert "sessions" in message, message
    assert "updated_at" in message, message


# -- persistence across a process boundary ---------------------------------

def test_a_row_written_by_one_process_is_read_by_another(tmp_path):
    """The headline property: the data outlives the process that wrote it.

    Two separate interpreters, not two calls in one process: a shared
    in-process object would prove nothing about the file.
    """
    db = tmp_path / "durable.db"

    _run_child(tmp_path, """
        store = DesktopStorage(sys.argv[1])
        store.initialize()
        store.record_run(
            run_id="durable-run",
            benchmark="durability-probe",
            primary_metric="accuracy",
            primary_score=0.5,
            status="completed",
            provenance="live",
            traces=[{"prompt": "p", "target": "t", "correct": True}],
        )
        store.add_experiment({"id": "durable-exp"})
        print("wrote")
        """, str(db))

    output = _run_child(tmp_path, """
        store = DesktopStorage(sys.argv[1])
        store.initialize()
        runs = store.history(benchmark="durability-probe")
        print(json.dumps({
            "runs": [
                {
                    "run_id": r["run_id"],
                    "score": r["primary_score"],
                    "traces": r["traces"],
                }
                for r in runs
            ],
            "experiments": [e["id"] for e in store.list_experiments()],
        }))
        """, str(db))

    payload = json.loads(output.strip().splitlines()[-1])
    assert [r["run_id"] for r in payload["runs"]] == ["durable-run"]
    assert payload["runs"][0]["score"] == 0.5
    assert payload["runs"][0]["traces"] == [
        {"prompt": "p", "target": "t", "correct": True}]
    assert payload["experiments"] == ["durable-exp"]


def test_concurrent_processes_do_not_lose_writes(tmp_path):
    """Four processes writing the same file at the same time: every write has
    to land."""
    db = tmp_path / "concurrent.db"
    DesktopStorage(db).initialize()

    script = tmp_path / "concurrent_child.py"
    script.write_text(
        CHILD_PREAMBLE
        + textwrap.dedent(
            """
            store = DesktopStorage(sys.argv[1])
            store.initialize()
            store.record_run(
                run_id="run-" + sys.argv[2],
                benchmark="concurrency-probe",
                primary_metric="accuracy",
                primary_score=float(sys.argv[2]),
                status="completed",
                provenance="live",
            )
            print("ok")
            """),
        encoding="utf-8")

    writers = [
        subprocess.Popen(
            [sys.executable, str(script), str(db), str(index)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(tmp_path),
        )
        for index in range(4)
    ]
    for writer in writers:
        stdout, stderr = writer.communicate(timeout=120)
        assert writer.returncode == 0, f"{stdout}\n{stderr}"

    store = DesktopStorage(db)
    rows = store.history(benchmark="concurrency-probe", limit=50)
    assert {row["run_id"] for row in rows} == {
        "run-0", "run-1", "run-2", "run-3"}
    assert {row["primary_score"] for row in rows} == {0.0, 1.0, 2.0, 3.0}


# -- a write that is refused -----------------------------------------------

def test_a_refused_write_leaves_no_row(tmp_path):
    """A measurement that is not a number is not a number. Nothing is
    written, and no placeholder score appears in its place."""
    store = DesktopStorage(tmp_path / "refused.db")
    store.initialize()

    with pytest.raises(StorageError):
        store.record_run(
            benchmark="refusal-probe",
            status="completed",
            provenance="live",
            primary_score="not-a-number",
        )

    assert store.history(benchmark="refusal-probe", limit=50) == []


def test_a_transaction_that_fails_leaves_no_partial_row(tmp_path):
    """The write itself is atomic: a transaction that raises rolls back the
    row it had already put in."""
    store = DesktopStorage(tmp_path / "atomic.db")
    store.initialize()

    connection = store._connect()
    try:
        with pytest.raises(RuntimeError):
            with connection:
                connection.execute(
                    "INSERT INTO benchmark_runs (run_id, benchmark, status, "
                    "provenance, created_at, traces) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("half-written", "atomic-probe", "completed", "live",
                     "2020-01-01T00:00:00+00:00", "[]"),
                )
                raise RuntimeError("simulated failure mid-transaction")
    finally:
        connection.close()

    assert store.history(benchmark="atomic-probe", limit=50) == []


# -- the databases that ship in the checkout -------------------------------

@pytest.mark.parametrize("relative", [
    "backend/storage/mech.db",
    "backend/runtime.db",
])
def test_the_shipped_databases_are_not_tracked_by_git(relative):
    """They are runtime state, not source: `.gitignore` excludes `*.db`, so
    a checkout has no database and the schema is built on first use."""
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert completed.returncode != 0, (
        f"{relative} is tracked by git, but it is runtime state")
