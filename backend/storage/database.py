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
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    item_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
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
            for key, value in DEFAULT_SETTINGS.items():
                connection.execute(
                    """
                    INSERT OR IGNORE INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (key, json.dumps(value), self._now()),
                )

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
        """Count files under ``root`` without following symlinks out of it.

        Uses an explicit stack rather than ``rglob``: rglob materialises a Path
        for every entry and follows directory symlinks, so a link pointing at
        ``C:\\`` would be traversed. ``followlinks=False`` on the ``stat`` call
        keeps the count inside the root the caller already authorised.
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

    def _list_json_items(self, table: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT payload FROM {table} ORDER BY created_at, item_id"
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def _add_json_item(self, table: str, item: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise StorageError(f"{table} item must be an object")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise StorageError(f"{table} item requires a string 'id'")
        with self._connect() as connection:
            connection.execute(
                f"""
                INSERT INTO {table} (item_id, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (item_id, json.dumps(item), self._now()),
            )
        return item

    def _delete_json_item(self, table: str, item_id: str) -> bool:
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
