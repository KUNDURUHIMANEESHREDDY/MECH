from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StorageError(Exception):
    """Raised when desktop storage receives invalid data."""


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "mech.db"


def get_default_db_path() -> Path:
    """Resolve the SQLite path from MECH_STORAGE_DB, else backend/storage/mech.db."""
    env_path = os.environ.get("MECH_STORAGE_DB")
    return Path(env_path) if env_path else DEFAULT_DB_PATH


#: The path value that asks for an in-memory database.
MEMORY_DB_PATH = ":memory:"

#: A single shared in-memory database for this process.
#:
#: A bare ``:memory:`` makes every *connection* its own private database, so
#: `initialize()`'s schema is invisible to the next query -- the caller sees
#: ``no such table: settings``. A shared-cache URI makes every connection in the
#: process address one database, which is what a caller writing
#: ``MECH_STORAGE_DB=:memory:`` expects.
MEMORY_DB_URI = "file:mech-in-memory?mode=memory&cache=shared"

#: One connection held open for the process lifetime. A shared in-memory
#: database is destroyed when its last connection closes, so without this the
#: schema would vanish between two calls that each opened and closed their own
#: connection.
_MEMORY_KEEPALIVE: list[sqlite3.Connection] = []
_MEMORY_LOCK = threading.Lock()


def is_memory_path(db_path: Any) -> bool:
    """Whether ``db_path`` asks for the shared in-memory database."""
    return str(db_path) == MEMORY_DB_PATH


def _hold_memory_database_open() -> None:
    """Ensure the shared in-memory database exists and outlives any one query."""
    with _MEMORY_LOCK:
        if not _MEMORY_KEEPALIVE:
            _MEMORY_KEEPALIVE.append(sqlite3.connect(MEMORY_DB_URI, uri=True))


DEFAULT_SETTINGS: dict[str, Any] = {
    "theme": "system",
    "gpuEnabled": False,
    "acceleration": "auto",
    "cacheEnabled": True,
    "cacheMaxSizeMb": 1024,
    "cachePath": str(Path.home() / ".cache" / "neural-debugger"),
    "modelPath": "gpt2",
    "workspacePath": str(Path.home()),
    "pythonPath": "",
    "projectsPath": "",
}

SETTING_TYPES = {
    "theme": str,
    "gpuEnabled": bool,
    "acceleration": str,
    "cacheEnabled": bool,
    "cacheMaxSizeMb": int,
    "cachePath": str,
    "modelPath": str,
    "workspacePath": str,
    "pythonPath": str,
    "projectsPath": str,
}

VALID_THEMES = {"system", "light", "dark"}
VALID_ACCELERATION = {"auto", "on", "off"}

#: Windows `FILE_ATTRIBUTE_REPARSE_POINT`. A junction, a symlink and a mount point
#: all carry it. Present only on Windows; absent elsewhere, where `is_symlink()`
#: is sufficient to identify a link.
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _is_link_like(entry: os.DirEntry) -> bool:
    """Whether a directory entry is a link rather than a real directory.

    ``entry.is_dir(follow_symlinks=False)`` is not enough on Windows. A junction
    is a reparse point, so ``is_symlink()`` returns False and
    ``is_dir(follow_symlinks=False)`` returns True -- the junction looks like an
    ordinary directory to that call and was therefore traversed out of the
    authorised workspace root.

    Both signals are checked. ``st_file_attributes`` only exists on Windows, so
    the reparse-point test is guarded rather than assumed.
    """
    try:
        if entry.is_symlink():
            return True
    except OSError:
        return True
    try:
        attributes = getattr(entry.stat(follow_symlinks=False),
                             "st_file_attributes", None)
    except OSError:
        return True
    if attributes is None:
        return False
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _is_number(value: Any) -> bool:
    """Whether a stored score is a real number.

    ``bool`` is excluded on purpose: it is an ``int`` subclass in Python, so
    ``True`` would otherwise be written as the score 1.0.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _dumps_traces(traces: list[Any] | None) -> str:
    """Serialise per-sample traces, always as a JSON array."""
    if traces is None:
        traces = []
    if not isinstance(traces, list):
        raise StorageError("traces must be a list of per-sample trace objects")
    return json.dumps(traces)


def _dumps_detail(detail: dict[str, Any] | None) -> str | None:
    """Serialise a run detail object; None stays SQL NULL rather than '{}'."""
    if detail is None:
        return None
    if not isinstance(detail, dict):
        raise StorageError("detail must be an object or None")
    return json.dumps(detail)


def _loads_or(raw: str | None, fallback: Any) -> Any:
    """Parse a stored JSON column, falling back when the text is unusable."""
    if raw is None:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _trace_key(trace: Any) -> str | None:
    """The per-sample identity two traces are matched on.

    A trace carrying neither ``prompt`` nor ``target`` cannot be matched to its
    counterpart in the other run, so it yields None and the caller reports the
    comparison as not comparable instead of guessing an alignment.
    """
    if not isinstance(trace, dict):
        return None
    for key in ("prompt", "target"):
        value = trace.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _trace_outcome(trace: dict[str, Any]) -> str:
    """A stable string for whatever the trace records about the sample.

    The whole trace minus its identity is serialised with sorted keys so that
    two traces differing in any recorded field hash differently, and one that
    differs only in dict insertion order does not.
    """
    payload = {key: value for key, value in trace.items()
               if key not in ("prompt", "target")}
    return json.dumps(payload, sort_keys=True, default=repr)


def _traces_delta(
    traces_a: list[Any],
    traces_b: list[Any],
    not_comparable: list[str],
) -> dict[str, Any] | None:
    """Per-sample trace differences, or None with a reason when unmatched.

    ``changed`` holds prompts, so two different samples that happen to share a
    prompt collapse into one entry. That is acceptable -- such a prompt is
    already ambiguous by the caller's own key -- and is preferable to inventing
    an ordering the stored traces do not have.
    """
    if not isinstance(traces_a, list) or not isinstance(traces_b, list):
        not_comparable.append("traces are not stored as JSON arrays")
        return None
    if not traces_a or not traces_b:
        not_comparable.append(
            "one side has no recorded traces, so per-sample outcomes cannot "
            "be matched")
        return None
    keyed_a = [_trace_key(trace) for trace in traces_a]
    keyed_b = [_trace_key(trace) for trace in traces_b]
    if any(key is None for key in keyed_a + keyed_b):
        not_comparable.append(
            "traces carry no prompt/target key, so samples cannot be matched "
            "across runs")
        return None
    by_key_b = {key: trace for key, trace in zip(keyed_b, traces_b)}
    changed = [
        key for key, trace in zip(keyed_a, traces_a)
        if key in by_key_b
        and _trace_outcome(trace) != _trace_outcome(by_key_b[key])
    ]
    return {
        "n_traces_a": len(traces_a),
        "n_traces_b": len(traces_b),
        "changed": changed,
    }


class DesktopStorage:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recent_projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    opened_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recent_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    project_path TEXT,
                    opened_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    item_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    item_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plugins (
                    name TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    manifest TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    installed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    run_id TEXT PRIMARY KEY,
                    benchmark TEXT NOT NULL,
                    primary_metric TEXT,
                    primary_score REAL,
                    status TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    model_id TEXT,
                    weights_fingerprint TEXT,
                    git_commit TEXT,
                    created_at TEXT NOT NULL,
                    traces TEXT NOT NULL DEFAULT '[]',
                    detail TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_benchmark_runs_benchmark_created
                    ON benchmark_runs (benchmark, created_at DESC);
                """
            )
            self._migrate(connection)
            for key, value in DEFAULT_SETTINGS.items():
                connection.execute(
                    """
                    INSERT OR IGNORE INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (key, json.dumps(value), self._now()),
                )

    #: Columns added after the first schema shipped. `CREATE TABLE IF NOT
    #: EXISTS` is a no-op against an existing table, so a database written by an
    #: older build would otherwise be missing them permanently and every write
    #: naming them would fail.
    _ADDED_COLUMNS = {
        "experiments": {"updated_at": "TEXT"},
        "sessions": {"updated_at": "TEXT"},
    }

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """Bring an older database up to the current schema, in place.

        Existing rows get `updated_at` seeded from `created_at`, so an item's
        two stamps stay ordered rather than the update time reading as
        earlier than its own creation.
        """
        for table, columns in self._ADDED_COLUMNS.items():
            present = {row[1] for row in connection.execute(
                f"PRAGMA table_info({table})")}
            if not present:
                # The table does not exist yet; CREATE TABLE above made it with
                # every column already.
                continue
            for column, kind in columns.items():
                if column in present:
                    continue
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
                if column == "updated_at":
                    connection.execute(
                        f"UPDATE {table} SET updated_at = created_at")

    def get_settings(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value FROM settings").fetchall()
        settings = DEFAULT_SETTINGS.copy()
        for row in rows:
            if row["key"] in DEFAULT_SETTINGS:
                settings[row["key"]] = json.loads(row["value"])
        return settings

    def update_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(settings, dict):
            raise StorageError("settings must be an object")

        current = self.get_settings()
        for key, value in settings.items():
            if key not in DEFAULT_SETTINGS:
                raise StorageError(f"Unknown setting: {key}")
            expected_type = SETTING_TYPES[key]
            if expected_type is bool:
                valid = isinstance(value, bool)
            elif expected_type is int:
                valid = isinstance(value, int) and not isinstance(value, bool)
            else:
                valid = isinstance(value, expected_type)
            if not valid:
                raise StorageError(f"Invalid type for setting {key}")
            if key == "theme" and value not in VALID_THEMES:
                raise StorageError("theme must be system, light, or dark")
            if key == "acceleration" and value not in VALID_ACCELERATION:
                raise StorageError("acceleration must be auto, on, or off")
            if key == "cacheMaxSizeMb" and value < 0:
                raise StorageError("cacheMaxSizeMb must be a non-negative integer")
            current[key] = value

        with self._connect() as connection:
            for key, value in current.items():
                connection.execute(
                    """
                    INSERT INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(value), self._now()),
                )
        return current

    def list_recent_projects(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, path, name, opened_at
                FROM recent_projects
                ORDER BY opened_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def add_recent_project(self, path: str, name: str | None = None) -> dict[str, Any]:
        project_path = self._require_path(path, "project path")
        project_name = name or Path(project_path).name or project_path
        opened_at = self._now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recent_projects (path, name, opened_at)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    name = excluded.name,
                    opened_at = excluded.opened_at
                """,
                (project_path, project_name, opened_at),
            )
            row = connection.execute(
                """
                SELECT id, path, name, opened_at
                FROM recent_projects
                WHERE path = ?
                """,
                (project_path,),
            ).fetchone()
        return self._project_from_row(row)

    def list_recent_files(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, path, project_path, opened_at
                FROM recent_files
                ORDER BY opened_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._file_from_row(row) for row in rows]

    def add_recent_file(
        self,
        path: str,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        file_path = self._require_path(path, "file path")
        normalized_project_path = (
            str(Path(project_path).expanduser()) if project_path else None
        )
        opened_at = self._now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recent_files (path, project_path, opened_at)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    project_path = excluded.project_path,
                    opened_at = excluded.opened_at
                """,
                (file_path, normalized_project_path, opened_at),
            )
            row = connection.execute(
                """
                SELECT id, path, project_path, opened_at
                FROM recent_files
                WHERE path = ?
                """,
                (file_path,),
            ).fetchone()
        return self._file_from_row(row)

    def list_experiments(self) -> list[dict[str, Any]]:
        return self._list_json_items("experiments")

    def add_experiment(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._add_json_item("experiments", item)

    def delete_experiment(self, item_id: str) -> bool:
        return self._delete_json_item("experiments", item_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        return self._list_json_items("sessions")

    def add_session(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._add_json_item("sessions", item)

    def delete_session(self, item_id: str) -> bool:
        return self._delete_json_item("sessions", item_id)

    def list_plugins(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT name, path, manifest, enabled, installed_at
                FROM plugins
                ORDER BY name
                """
            ).fetchall()
        items = []
        for row in rows:
            try:
                manifest = json.loads(row["manifest"])
            except (TypeError, ValueError):
                manifest = {}
            items.append({
                "name": row["name"],
                "path": row["path"],
                "manifest": manifest if isinstance(manifest, dict) else {},
                "enabled": bool(row["enabled"]),
                "installed_at": row["installed_at"],
            })
        return items

    def upsert_plugin(self, name: str, path: str, manifest: dict[str, Any],
                      enabled: bool) -> dict[str, Any]:
        installed_at = self._now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO plugins (name, path, manifest, enabled, installed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    path = excluded.path,
                    manifest = excluded.manifest,
                    enabled = excluded.enabled,
                    installed_at = excluded.installed_at
                """,
                (name, path, json.dumps(manifest), int(bool(enabled)),
                 installed_at),
            )
        return {
            "name": name,
            "path": path,
            "manifest": manifest,
            "enabled": bool(enabled),
            "installed_at": installed_at,
        }

    def set_plugin_enabled(self, name: str, enabled: bool) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE plugins SET enabled = ? WHERE name = ?",
                (int(bool(enabled)), name),
            )
        return cursor.rowcount > 0

    def delete_plugin(self, name: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM plugins WHERE name = ?", (name,)
            )
        return cursor.rowcount > 0

    #: Hard ceiling on the walk, so a huge tree cannot stall the sidecar.
    MAX_WORKSPACE_FILES = 10_000
    #: Wall-clock budget for the walk. The file cap alone does not bound a
    #: deep tree of empty directories.
    MAX_WORKSPACE_SECONDS = 2.0

    def _workspace_roots(self) -> list[Path]:
        """Directories ``describe_workspace`` is allowed to inspect.

        Defaults to the user's home directory, which is where the default
        ``workspacePath`` setting points. Override with
        ``MECH_WORKSPACE_ROOTS`` (os.pathsep-separated) to scope it further.
        """
        configured = os.environ.get("MECH_WORKSPACE_ROOTS", "")
        roots = [entry.strip() for entry in configured.split(os.pathsep) if entry.strip()]
        if not roots:
            roots = [str(Path.home())]
        resolved = []
        for root in roots:
            try:
                resolved.append(Path(root).expanduser().resolve())
            except OSError:
                continue
        return resolved

    def describe_workspace(self, path: str) -> dict[str, Any]:
        """Summarise a workspace directory, bounded to an allowed root.

        The path arrives from the renderer, so it is contained before use:
        without this, ``workspace.describe`` with ``{"path": "/"}`` walks the
        whole filesystem and reports a file count for any directory, which is a
        usable enumeration oracle. Containment also means a legitimate
        ``..``-containing path is accepted only if it still lands in a root.
        """
        if not isinstance(path, str) or not path.strip():
            raise StorageError("workspace path is required")

        try:
            candidate = Path(path).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise StorageError(f"workspace path is not resolvable: {exc}") from exc

        roots = self._workspace_roots()
        if not any(candidate == root or root in candidate.parents for root in roots):
            raise StorageError(
                f"'{candidate}' is outside the allowed workspace roots"
            )

        exists = candidate.exists()
        is_directory = exists and candidate.is_dir()
        file_count = 0
        count_capped = False

        if is_directory:
            file_count, count_capped = self._count_files_bounded(candidate)

        return {
            "path": str(candidate),
            "exists": exists,
            "isDirectory": is_directory,
            "name": candidate.name or str(candidate),
            "fileCount": file_count,
            "countCapped": count_capped,
        }

    def _count_files_bounded(self, root: Path) -> tuple[int, bool]:
        """Count files under ``root`` without following links out of it.

        Uses an explicit stack rather than ``rglob``: rglob materialises a Path
        for every entry and follows directory symlinks, so a link pointing at
        ``C:\\`` would be traversed.

        Two link types have to be refused, and only one of them is a symlink.

        On POSIX, ``entry.is_dir(follow_symlinks=False)`` is False for a symlink
        to a directory, so the ``follow_symlinks=False`` call alone keeps the walk
        inside the root.

        On Windows that is not sufficient. A **junction** is a reparse point, not
        a symlink: ``is_symlink()`` returns False and ``is_dir(follow_symlinks=
        False)`` returns **True**, so the junction was pushed onto the stack and
        traversed. Measured on this host, with five files living outside the
        authorised root:

            file.txt   is_symlink=False  is_dir(follow=False)=False  attrs=32
            junction   is_symlink=False  is_dir(follow=False)=True   attrs=1040
            plain      is_symlink=False  is_dir(follow=False)=True   attrs=16

        1040 is 0x418 -- FILE_ATTRIBUTE_REPARSE_POINT (0x400) | DIRECTORY (0x10).
        So a junction escaped the workspace root and was counted, turning
        ``workspace.describe`` into the same enumeration oracle the containment
        check above exists to prevent. The test that would have caught this was
        skipped on Windows, because it built its escape with ``os.symlink``, which
        needs a privilege this account does not hold.

        Reparse points are therefore refused as well. That is deliberately broad:
        it also covers mount points and OneDrive placeholders, none of which
        should be traversed out of an authorised root either.
        """
        deadline = time.monotonic() + self.MAX_WORKSPACE_SECONDS
        stack = [root]
        count = 0
        while stack:
            if time.monotonic() > deadline:
                return count, True
            current = stack.pop()
            try:
                entries = list(os.scandir(current))
            except (OSError, PermissionError):
                continue
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if _is_link_like(entry):
                            # Counted as neither a file nor a directory: it is a
                            # pointer, and following it is what escapes the root.
                            continue
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        count += 1
                        if count >= self.MAX_WORKSPACE_FILES:
                            return count, True
                except OSError:
                    continue
        return count, False

    def _connect(self) -> sqlite3.Connection:
        """A new connection to this store's database.

        ``:memory:`` is redirected to the process-wide shared in-memory database
        (see `MEMORY_DB_URI`). Connecting a bare ``:memory:`` here would hand
        every caller a private empty database, which is the defect this
        redirection exists to remove.
        """
        if is_memory_path(self.db_path):
            _hold_memory_database_open()
            connection = sqlite3.connect(MEMORY_DB_URI, uri=True)
        else:
            connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def checkpoint_wal(self) -> tuple[int, int, int]:
        """Flush the WAL into the main DB and truncate the -wal file.

        Returns SQLite's ``(busy, log_pages, checkpointed)`` triple. ``busy``
        is non-zero when another connection holds a lock; that is a normal
        outcome, not an error, so callers get the numbers instead of an
        exception.
        """
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        except sqlite3.OperationalError:
            return (1, 0, 0)
        if row is None:
            return (0, 0, 0)
        return (int(row[0]), int(row[1]), int(row[2]))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    _JSON_TABLES = ("experiments", "sessions")

    def _check_table(self, table: str) -> None:
        if table not in self._JSON_TABLES:
            raise StorageError(f"unknown table: {table}")

    def _list_json_items(self, table: str) -> list[dict[str, Any]]:
        self._check_table(table)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT payload FROM {table} ORDER BY created_at, item_id"
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def _add_json_item(self, table: str, item: dict[str, Any]) -> dict[str, Any]:
        self._check_table(table)
        if not isinstance(item, dict):
            raise StorageError(f"{table} item must be an object")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise StorageError(f"{table} item requires a string 'id'")
        stamp = self._now()
        with self._connect() as connection:
            connection.execute(
                f"""
                INSERT INTO {table} (item_id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                # `created_at` is deliberately absent from the UPDATE clause.
                # Rewriting it made `list_*` order by last-modified while the
                # column name and the ORDER BY both said "created", so saving
                # an old item silently moved it to the end of the list.
                (item_id, json.dumps(item), stamp, stamp),
            )
        return item

    def _delete_json_item(self, table: str, item_id: str) -> bool:
        self._check_table(table)
        if not isinstance(item_id, str) or not item_id.strip():
            raise StorageError(f"{table} item id must be a string")
        with self._connect() as connection:
            cursor = connection.execute(
                f"DELETE FROM {table} WHERE item_id = ?", (item_id,)
            )
        return cursor.rowcount > 0

    def record_run(
        self,
        *,
        run_id: str | None = None,
        benchmark: str,
        primary_metric: str | None = None,
        primary_score: float | None = None,
        status: str,
        provenance: str,
        model_id: str | None = None,
        weights_fingerprint: str | None = None,
        git_commit: str | None = None,
        traces: list[Any] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert one benchmark run and return the row as it was stored.

        ``primary_score=None`` is written as SQL NULL and read back as None.
        Substituting 0.0 would turn "this run never measured the metric" into a
        measured zero, which is the one thing a benchmark record must not do:
        downstream comparisons and charts would treat the fabricated value as
        evidence.

        ``INSERT OR REPLACE`` is used so a caller re-recording the same
        ``run_id`` (a retry, or a corrected result) overwrites the earlier
        attempt instead of failing on the primary key.
        """
        if not isinstance(benchmark, str) or not benchmark.strip():
            raise StorageError("benchmark must be a non-empty string")
        if not isinstance(status, str) or not status.strip():
            raise StorageError("status must be a non-empty string")
        if not isinstance(provenance, str) or not provenance.strip():
            raise StorageError("provenance must be a non-empty string")
        if primary_score is not None and not _is_number(primary_score):
            raise StorageError(
                "primary_score must be a number or None; a missing "
                "measurement is None, never a placeholder number")
        stored_run_id = run_id if run_id else uuid.uuid4().hex
        if not isinstance(stored_run_id, str) or not stored_run_id.strip():
            raise StorageError("run_id must be a non-empty string")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO benchmark_runs (
                    run_id, benchmark, primary_metric, primary_score, status,
                    provenance, model_id, weights_fingerprint, git_commit,
                    created_at, traces, detail
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stored_run_id,
                    benchmark,
                    primary_metric,
                    primary_score,
                    status,
                    provenance,
                    model_id,
                    weights_fingerprint,
                    git_commit,
                    self._now(),
                    _dumps_traces(traces),
                    _dumps_detail(detail),
                ),
            )
            row = connection.execute(
                "SELECT * FROM benchmark_runs WHERE run_id = ?",
                (stored_run_id,),
            ).fetchone()
        # Read back rather than echoing the arguments: the row is the record of
        # what was actually stored, and it is what makes a NULL round-trip
        # visible to the caller.
        return self._run_from_row(row)

    def history(
        self, benchmark: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Recent runs, newest first, optionally for one benchmark.

        ``limit`` is clamped to 1..500. A history view that can be asked for
        zero rows, or for an unbounded scan, would either render as empty when
        the caller meant "all" or pull the whole table into memory.
        """
        try:
            bounded = int(limit)
        except (TypeError, ValueError):
            raise StorageError("limit must be an integer") from None
        bounded = max(1, min(500, bounded))
        sql = """
            SELECT * FROM benchmark_runs
        """
        parameters: list[Any] = []
        if benchmark is not None:
            sql += " WHERE benchmark = ?"
            parameters.append(benchmark)
        # run_id breaks ties: two runs recorded inside the same clock tick
        # share a created_at string, and an unstable order makes paging and
        # diffs look like the data changed when it did not.
        sql += " ORDER BY created_at DESC, run_id DESC LIMIT ?"
        parameters.append(bounded)
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._run_from_row(row) for row in rows]

    def compare(self, run_a: str, run_b: str) -> dict[str, Any]:
        """Compare two stored runs without ever inventing a score change.

        ``score_delta`` is None unless the two rows measure the same metric on
        the same benchmark and both actually carry a score. Returning 0.0 for a
        missing measurement would read as "no change", which is a claim about
        the model that the stored data does not support; the reason string
        carries the explanation instead, and ``not_comparable`` lists every
        reason so a caller cannot act on a delta without seeing them.
        """
        stored_a = self._get_run(run_a)
        stored_b = self._get_run(run_b)
        not_comparable: list[str] = []

        same_benchmark = stored_a["benchmark"] == stored_b["benchmark"]
        if not same_benchmark:
            not_comparable.append(
                f"different benchmarks: {stored_a['benchmark']!r} vs "
                f"{stored_b['benchmark']!r}")

        metrics = (stored_a["primary_metric"], stored_b["primary_metric"])
        # A score only means something against a named metric, so two unnamed
        # scores are not comparable: treating "no metric recorded" as a match
        # would fabricate a delta between possibly different measurements. A
        # one-sided name is not a match either.
        same_metric = bool(metrics[0]) and metrics[0] == metrics[1]
        primary_metric = metrics[0] if same_metric else None
        if not same_metric:
            not_comparable.append(
                f"different primary metrics: {metrics[0]!r} vs {metrics[1]!r}")

        scores = (stored_a["primary_score"], stored_b["primary_score"])
        if scores[0] is None or scores[1] is None:
            score_delta = None
            missing = [
                run_id for run_id, score in zip((run_a, run_b), scores)
                if score is None
            ]
            score_delta_reason = (
                "a primary score is NULL for run(s) " + ", ".join(missing)
                + "; a missing measurement is not a zero, so no delta is "
                "reported")
            not_comparable.append(
                "primary score not measured for run(s) " + ", ".join(missing))
        elif not same_benchmark:
            score_delta = None
            score_delta_reason = (
                "not comparable across different benchmarks: "
                f"{stored_a['benchmark']!r} vs {stored_b['benchmark']!r}")
        elif not same_metric:
            score_delta = None
            score_delta_reason = (
                "different primary metrics "
                f"({metrics[0]!r} vs {metrics[1]!r}); subtracting them would "
                "be a meaningless number")
        else:
            # Both scores are real numbers, so this subtraction is a measured
            # change rather than a substitute for one.
            score_delta = float(scores[1]) - float(scores[0])
            score_delta_reason = None

        fingerprints = (
            stored_a["weights_fingerprint"], stored_b["weights_fingerprint"])
        if fingerprints[0] is None or fingerprints[1] is None:
            weights_fingerprint_match: bool | None = None
        else:
            weights_fingerprint_match = fingerprints[0] == fingerprints[1]

        traces_delta = _traces_delta(
            stored_a["traces"], stored_b["traces"], not_comparable)

        return {
            "run_a": stored_a,
            "run_b": stored_b,
            "same_benchmark": same_benchmark,
            "primary_metric": primary_metric,
            "score_delta": score_delta,
            "score_delta_reason": score_delta_reason,
            "traces_delta": traces_delta,
            "not_comparable": not_comparable,
            "weights_fingerprint_match": weights_fingerprint_match,
        }

    def delete_run(self, run_id: str) -> bool:
        """Delete one run; True when a row was actually removed."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM benchmark_runs WHERE run_id = ?", (run_id,)
            )
        return cursor.rowcount > 0

    def _get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise StorageError(f"no such benchmark run: {run_id}")
        return self._run_from_row(row)

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "benchmark": row["benchmark"],
            "primary_metric": row["primary_metric"],
            "primary_score": row["primary_score"],
            "status": row["status"],
            "provenance": row["provenance"],
            "model_id": row["model_id"],
            "weights_fingerprint": row["weights_fingerprint"],
            "git_commit": row["git_commit"],
            "created_at": row["created_at"],
            "traces": _loads_or(row["traces"], []),
            "detail": _loads_or(row["detail"], None),
        }

    @staticmethod
    def _require_path(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise StorageError(f"{label} is required")
        return str(Path(value).expanduser())

    @staticmethod
    def _project_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "path": row["path"],
            "name": row["name"],
            "openedAt": row["opened_at"],
        }

    @staticmethod
    def _file_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "path": row["path"],
            "projectPath": row["project_path"],
            "openedAt": row["opened_at"],
        }


def checkpoint_wal(db_path: Path | str | None = None) -> tuple[int, int, int]:
    """Checkpoint the WAL for the default (or given) database.

    Returns ``(busy, log_pages, checkpointed)``. Never raises for lock
    contention -- a busy result is a legitimate outcome that shutdown should
    tolerate rather than crash on.
    """
    return DesktopStorage(db_path or get_default_db_path()).checkpoint_wal()


def record_run(**kwargs: Any) -> dict[str, Any]:
    """Record a benchmark run in the default database (module-level form)."""
    storage = DesktopStorage(get_default_db_path())
    storage.initialize()
    return storage.record_run(**kwargs)


def history(benchmark: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """Benchmark-run history from the default database."""
    storage = DesktopStorage(get_default_db_path())
    storage.initialize()
    return storage.history(benchmark=benchmark, limit=limit)


def compare(run_a: str, run_b: str) -> dict[str, Any]:
    """Compare two runs in the default database."""
    storage = DesktopStorage(get_default_db_path())
    storage.initialize()
    return storage.compare(run_a, run_b)
