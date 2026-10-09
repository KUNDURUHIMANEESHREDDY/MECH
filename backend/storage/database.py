from __future__ import annotations

import json
import os
import sqlite3
import time
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
