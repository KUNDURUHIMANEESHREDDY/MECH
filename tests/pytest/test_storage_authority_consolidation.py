"""One storage authority, and a `created_at` that means what it says.

`backend/storage/database.py` and `backend/core/database.py` both claimed to be
the project's database. Measured on this host before the fix:

    backend/core/database.py
      DATABASE_URL  sqlite:///./interp_research.db     <- CWD-relative
      file on disk  0 bytes
      tables        NONE
      init_db()     zero callers anywhere in the repo

So the second authority was a 0-byte file with no schema, and its one consumer
raised on every call:

    ProjectExporter().export_project("demo")
    -> sqlalchemy.exc.OperationalError: no such table: sessions

Worse, it declared `experiments` and `sessions` against the live `mech.db`
versions with **zero shared columns**:

    mech.db  experiments(item_id TEXT, payload TEXT, created_at TEXT)
    orm      experiments(id INTEGER, experiment_id, model_id, dataset_id,
                         status, metrics JSON, provenance JSON)

Same table names, incompatible shapes, two different files. Whichever one an
engine opened first would decide the schema, and the other's queries would
fail against it.

Separately, `_add_json_item` upserted `created_at` on conflict, so
`list_experiments()`/`list_sessions()` ordered by last-modified while claiming
to order by creation. Re-saving the oldest item moved it to the end:

    insert aaa, bbb, ccc   ->  ['aaa', 'bbb', 'ccc']
    re-save aaa            ->  ['bbb', 'ccc', 'aaa']
"""
import ast
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.storage import DesktopStorage  # noqa: E402


def _source_files():
    """Tracked Python sources, minus this test and build copies.

    Build output (`frontend/release/`, `MECH-standalone/`) is gitignored and
    carries a stale copy of the tree, so a scan that includes it reports the
    second authority forever after it is deleted from the source.
    """
    this_file = Path(__file__).resolve()
    for path in REPO_ROOT.rglob("*.py"):
        if path.resolve() == this_file:
            continue
        if any(part in (".git", "node_modules", "MECH-standalone",
                        ".pytest-tmp", "__pycache__", ".agents", "release")
               for part in path.parts):
            continue
        yield path


# ── one authority ────────────────────────────────────────────────────── #

def test_the_second_authority_is_gone():
    """`backend.core.database` must not exist as a competing definition."""
    assert not (REPO_ROOT / "backend" / "core" / "database.py").exists(), (
        "backend/core/database.py is a second storage authority: its own "
        "engine, its own CWD-relative file, and `experiments`/`sessions` "
        "tables that share no column with the live ones. Consolidate on "
        "backend/storage/database.py instead.")


def test_core_does_not_re_export_the_orm_records():
    import backend.core as core

    for name in ("ExperimentRecord", "SessionRecord", "ReportRecord"):
        assert not hasattr(core, name), (
            f"backend.core still re-exports {name}; the ORM it came from is "
            "gone and re-exporting it makes a deleted authority look live")
        assert name not in core.__all__


def test_nothing_creates_a_second_engine():
    """No module may open its own database engine or declare a db URL.

    Parsed with `ast` rather than grepped. A text scan matches the prose in
    the comments that explain why the second authority was removed -- which is
    exactly the text that must be allowed to mention `sqlite:///`.
    """
    engine_calls = {"create_engine", "sessionmaker", "declarative_base"}
    offenders = []
    for path in _source_files():
        try:
            # utf-8-sig, not utf-8: a file with a byte-order mark is valid
            # Python that plain `utf-8` refuses to parse.
            tree = ast.parse(
                path.read_text(encoding="utf-8-sig", errors="replace"))
        except SyntaxError as exc:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: unparseable ({exc})")
            continue
        relative = path.relative_to(REPO_ROOT)

        for node in ast.walk(tree):
            # A real call, not a mention in a comment or docstring.
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in engine_calls):
                offenders.append(
                    f"{relative}:{node.lineno} calls {node.func.id}()")
            # A db URL assigned to a name, rather than discussed in prose.
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if (isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                        and value.value.startswith("sqlite:///")):
                    offenders.append(
                        f"{relative}:{node.lineno} declares a sqlite URL")
    assert not offenders, (
        "a second database authority was declared:\n  "
        + "\n  ".join(offenders)
        + "\nbackend/storage/database.py is the only place allowed to hold a "
          "database handle.")


def test_importing_the_api_creates_no_stray_database_file(tmp_path, monkeypatch):
    """The CWD-relative URL meant the file landed wherever the app was launched.

    `sqlite:///./interp_research.db` resolved against the process working
    directory, so running the backend from two different folders produced two
    different empty databases and neither had a schema.
    """
    monkeypatch.chdir(tmp_path)
    importlib.import_module("backend.api.dispatcher")
    strays = [p.name for p in tmp_path.iterdir()]
    assert not strays, (
        f"importing the API wrote files into the working directory: {strays}")


# ── the live authority owns a known, non-colliding schema ────────────── #

def test_the_authority_owns_exactly_the_known_tables(tmp_path):
    store = DesktopStorage(tmp_path / "auth.db")
    store.initialize()
    with sqlite3.connect(store.db_path) as connection:
        names = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    names -= {n for n in names if n.startswith("sqlite_")}
    assert names == {
        "settings", "recent_projects", "recent_files",
        "experiments", "sessions", "plugins",
    }


def test_item_tables_carry_creation_and_update_times(tmp_path):
    store = DesktopStorage(tmp_path / "times.db")
    store.initialize()
    with sqlite3.connect(store.db_path) as connection:
        for table in ("experiments", "sessions"):
            columns = {row[1] for row in connection.execute(
                f"PRAGMA table_info({table})")}
            assert {"created_at", "updated_at"} <= columns, (
                f"{table} has no separate update time, so a save cannot be "
                "distinguished from a create")


# ── created_at means created ─────────────────────────────────────────── #

def test_updating_an_item_does_not_move_it_in_the_list(tmp_path):
    store = DesktopStorage(tmp_path / "order.db")
    store.initialize()
    for item_id in ("aaa", "bbb", "ccc"):
        store.add_experiment({"id": item_id, "n": item_id})

    assert [e["id"] for e in store.list_experiments()] == ["aaa", "bbb", "ccc"]

    # `aaa` was created first, so it must still be first.
    store.add_experiment({"id": "aaa", "n": "edited"})

    assert [e["id"] for e in store.list_experiments()] == ["aaa", "bbb", "ccc"], (
        "re-saving an item moved it in the list; the order is last-modified, "
        "not created")
    assert store.list_experiments()[0]["n"] == "edited", (
        "the update did not land")


def test_an_update_records_a_newer_update_time(tmp_path):
    store = DesktopStorage(tmp_path / "stamps.db")
    store.initialize()
    store.add_experiment({"id": "one", "n": 1})

    with sqlite3.connect(store.db_path) as connection:
        created, updated = connection.execute(
            "SELECT created_at, updated_at FROM experiments "
            "WHERE item_id = 'one'").fetchone()
    assert created == updated, "a fresh insert should have equal stamps"

    store.add_experiment({"id": "one", "n": 2})

    with sqlite3.connect(store.db_path) as connection:
        created_after, updated_after = connection.execute(
            "SELECT created_at, updated_at FROM experiments "
            "WHERE item_id = 'one'").fetchone()
    assert created_after == created, "created_at was rewritten by an update"
    assert updated_after >= updated, "updated_at did not advance"


def test_a_pre_existing_database_is_migrated(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` cannot add a column, so migration is explicit."""
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        # The exact pre-fix schema, written by an older build.
        connection.execute(
            "CREATE TABLE IF NOT EXISTS experiments ("
            "item_id TEXT PRIMARY KEY, payload TEXT NOT NULL, "
            "created_at TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO experiments VALUES ('legacy', '{\"id\": \"legacy\"}', "
            "'2020-01-01T00:00:00+00:00')")

    store = DesktopStorage(path)
    store.initialize()

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute(
            "PRAGMA table_info(experiments)")}
        count = connection.execute(
            "SELECT COUNT(*) FROM experiments").fetchone()[0]

    assert "updated_at" in columns, "the old table was never migrated"
    assert count == 1, "migration dropped the existing row"


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "twice.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE experiments (item_id TEXT PRIMARY KEY, "
            "payload TEXT NOT NULL, created_at TEXT NOT NULL)")
    store = DesktopStorage(path)
    store.initialize()
    store.initialize()  # must not raise on the second pass
    assert store.list_experiments() == []
