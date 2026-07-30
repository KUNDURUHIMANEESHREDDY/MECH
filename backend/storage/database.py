from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StorageError(Exception):
    """Raised when desktop storage receives invalid data."""


DEFAULT_SETTINGS: dict[str, Any] = {
    "theme": "system",
    "gpuEnabled": False,
    "cachePath": str(Path.home() / ".cache" / "neural-debugger"),
    "modelPath": "gpt2",
    "workspacePath": str(Path.home()),
}

SETTING_TYPES = {
    "theme": str,
    "gpuEnabled": bool,
    "cachePath": str,
    "modelPath": str,
    "workspacePath": str,
}

VALID_THEMES = {"system", "light", "dark"}


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
            if not isinstance(value, expected_type):
                raise StorageError(f"Invalid type for setting {key}")
            if key == "theme" and value not in VALID_THEMES:
                raise StorageError("theme must be system, light, or dark")
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

    def describe_workspace(self, path: str) -> dict[str, Any]:
        workspace_path = self._require_path(path, "workspace path")
        candidate = Path(workspace_path)
        file_count = 0
        if candidate.exists() and candidate.is_dir():
            for child in candidate.rglob("*"):
                if child.is_file():
                    file_count += 1
                    if file_count >= 10_000:
                        break

        return {
            "path": workspace_path,
            "exists": candidate.exists(),
            "name": candidate.name or workspace_path,
            "fileCount": file_count,
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

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
