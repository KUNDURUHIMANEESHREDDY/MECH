from __future__ import annotations

import json
import logging
import random
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("MECH.storage")

try:
    from backend.storage.migrations import migrate, MIGRATIONS, backup_db
except ImportError:
    migrate = MIGRATIONS = backup_db = None


class StorageError(Exception):
    """Raised when desktop storage receives invalid data or encounters a database error."""


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
    """Thread-safe SQLite persistent storage engine with automatic lock retry handlers."""

    ALLOWED_TABLES = {
        "settings",
        "recent_projects",
        "recent_files",
        "experiments",
        "sessions",
        "logs",
        "build_logs",
        "projects",
        "investigations",
        "hypotheses",
        "experiment_runs",
        "evidence_records",
        "mechanisms",
        "artifacts",
        "jobs",
    }

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".cache" / "neural-debugger" / "mech.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        """Creates an optimized SQLite connection with 30s timeout and WAL journal mode."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,  # autocommit mode
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 30000;")
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database."""
        return self._connect()

    def _execute_with_retry(self, fn: Callable[[sqlite3.Connection], Any], max_retries: int = 5) -> Any:
        """Executes a database transaction with exponential backoff on SQLite lock contention."""
        with self._lock:
            last_err = None
            for attempt in range(max_retries):
                try:
                    with self._connect() as conn:
                        return fn(conn)
                except sqlite3.OperationalError as e:
                    last_err = e
                    if "locked" in str(e).lower() or "busy" in str(e).lower():
                        sleep_time = (0.02 * (2 ** attempt)) + random.uniform(0.005, 0.02)
                        time.sleep(sleep_time)
                    else:
                        raise StorageError(f"SQLite Operational Error: {e}") from e
                except Exception as e:
                    raise StorageError(f"Storage operation failed: {e}") from e
            raise StorageError(f"SQLite transaction failed after {max_retries} attempts: {last_err}") from last_err

    def initialize(self) -> None:
        """Creates database tables and default configuration settings."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if migrate and MIGRATIONS and backup_db:
            backup_db(self.db_path)
            migrate(str(self.db_path), MIGRATIONS["mech.db"])

        def _init_schema(conn: sqlite3.Connection) -> None:
            conn.executescript(
                """
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;
                PRAGMA busy_timeout = 30000;

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

                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    module TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS build_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    build_id TEXT NOT NULL,
                    log_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    config TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS investigations (
                    item_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS hypotheses (
                    item_id TEXT PRIMARY KEY,
                    investigation_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiment_runs (
                    item_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    investigation_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence_records (
                    item_id TEXT PRIMARY KEY,
                    investigation_id TEXT NOT NULL,
                    hypothesis_id TEXT,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mechanisms (
                    item_id TEXT PRIMARY KEY,
                    investigation_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    item_id TEXT PRIMARY KEY,
                    investigation_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    item_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            for key, value in DEFAULT_SETTINGS.items():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (key, json.dumps(value), self._now()),
                )

        self._execute_with_retry(_init_schema)
        self._initialized = True

    # ── Settings ─────────────────────────────────────────────────────────────

    def get_settings(self) -> dict[str, Any]:
        def _get(conn: sqlite3.Connection) -> dict[str, Any]:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
            settings = DEFAULT_SETTINGS.copy()
            for row in rows:
                if row["key"] in DEFAULT_SETTINGS:
                    settings[row["key"]] = json.loads(row["value"])
            return settings

        return self._execute_with_retry(_get)

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

        def _update(conn: sqlite3.Connection) -> None:
            for key, value in current.items():
                conn.execute(
                    """
                    INSERT INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(value), self._now()),
                )

        self._execute_with_retry(_update)
        return current

    # ── Recent Projects & Files ──────────────────────────────────────────────

    def list_recent_projects(self, limit: int = 10) -> list[dict[str, Any]]:
        def _list(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                """
                SELECT id, path, name, opened_at
                FROM recent_projects
                ORDER BY opened_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._project_from_row(row) for row in rows]

        return self._execute_with_retry(_list)

    def add_recent_project(self, path: str, name: str | None = None) -> dict[str, Any]:
        project_path = self._require_path(path, "project path")
        project_name = name or Path(project_path).name or project_path
        opened_at = self._now()

        def _add(conn: sqlite3.Connection) -> dict[str, Any]:
            conn.execute(
                """
                INSERT INTO recent_projects (path, name, opened_at)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    name = excluded.name,
                    opened_at = excluded.opened_at
                """,
                (project_path, project_name, opened_at),
            )
            row = conn.execute(
                "SELECT id, path, name, opened_at FROM recent_projects WHERE path = ?",
                (project_path,),
            ).fetchone()
            return self._project_from_row(row)

        return self._execute_with_retry(_add)

    def list_recent_files(self, limit: int = 20) -> list[dict[str, Any]]:
        def _list(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                """
                SELECT id, path, project_path, opened_at
                FROM recent_files
                ORDER BY opened_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._file_from_row(row) for row in rows]

        return self._execute_with_retry(_list)

    def add_recent_file(self, path: str, project_path: str | None = None) -> dict[str, Any]:
        file_path = self._require_path(path, "file path")
        normalized_project_path = str(Path(project_path).expanduser()) if project_path else None
        opened_at = self._now()

        def _add(conn: sqlite3.Connection) -> dict[str, Any]:
            conn.execute(
                """
                INSERT INTO recent_files (path, project_path, opened_at)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    project_path = excluded.project_path,
                    opened_at = excluded.opened_at
                """,
                (file_path, normalized_project_path, opened_at),
            )
            row = conn.execute(
                "SELECT id, path, project_path, opened_at FROM recent_files WHERE path = ?",
                (file_path,),
            ).fetchone()
            return self._file_from_row(row)

        return self._execute_with_retry(_add)

    def clear_recent_files(self) -> int:
        def _clear(conn: sqlite3.Connection) -> int:
            cursor = conn.execute("DELETE FROM recent_files")
            return cursor.rowcount

        return self._execute_with_retry(_clear)

    # ── Experiments & Sessions ───────────────────────────────────────────────

    def list_experiments(self) -> list[dict[str, Any]]:
        return self._list_json_items("experiments")

    def get_experiment(self, item_id: str) -> Optional[dict[str, Any]]:
        return self._get_json_item("experiments", item_id)

    def add_experiment(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._add_json_item("experiments", item)

    def save_experiment(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._add_json_item("experiments", item)

    def delete_experiment(self, item_id: str) -> bool:
        return self._delete_json_item("experiments", item_id)

    def list_hypotheses(self, investigation_id: Optional[str] = None) -> list[dict[str, Any]]:
        items = self._list_json_items("hypotheses")
        if investigation_id:
            return [i for i in items if i.get("investigation_id") == investigation_id]
        return items

    def get_hypothesis(self, item_id: str) -> Optional[dict[str, Any]]:
        return self._get_json_item("hypotheses", item_id)

    def save_hypothesis(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._add_json_item("hypotheses", item)

    def delete_hypothesis(self, item_id: str) -> bool:
        return self._delete_json_item("hypotheses", item_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        return self._list_json_items("sessions")

    def add_session(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._add_json_item("sessions", item)

    def delete_session(self, item_id: str) -> bool:
        return self._delete_json_item("sessions", item_id)

    # ── Logs & Build Logs ────────────────────────────────────────────────────

    def add_log(
        self,
        level: str,
        message: str,
        module: str = "app",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Records an application log entry into persistent storage."""
        now = self._now()
        meta_json = json.dumps(metadata) if metadata else None

        def _add(conn: sqlite3.Connection) -> Dict[str, Any]:
            cursor = conn.execute(
                """
                INSERT INTO logs (level, message, module, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (level.upper(), message, module, meta_json, now),
            )
            return {
                "id": cursor.lastrowid,
                "level": level.upper(),
                "message": message,
                "module": module,
                "metadata": metadata or {},
                "created_at": now,
            }

        return self._execute_with_retry(_add)

    def list_logs(self, limit: int = 100, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Queries stored application logs."""
        def _list(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            if level:
                rows = conn.execute(
                    "SELECT id, level, message, module, metadata, created_at FROM logs WHERE level = ? ORDER BY id DESC LIMIT ?",
                    (level.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, level, message, module, metadata, created_at FROM logs ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            out = []
            for r in rows:
                meta = json.loads(r["metadata"]) if r["metadata"] else {}
                out.append({
                    "id": r["id"],
                    "level": r["level"],
                    "message": r["message"],
                    "module": r["module"],
                    "metadata": meta,
                    "created_at": r["created_at"],
                })
            return out

        return self._execute_with_retry(_list)

    def clear_logs(self) -> int:
        """Clears all stored application logs."""
        def _clear(conn: sqlite3.Connection) -> int:
            cursor = conn.execute("DELETE FROM logs")
            return cursor.rowcount

        return self._execute_with_retry(_clear)

    def add_build_log(self, build_id: str, log_text: str, status: str = "running") -> Dict[str, Any]:
        """Appends a build log output to persistent storage."""
        now = self._now()

        def _add(conn: sqlite3.Connection) -> Dict[str, Any]:
            cursor = conn.execute(
                """
                INSERT INTO build_logs (build_id, log_text, status, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (build_id, log_text, status, now),
            )
            return {
                "id": cursor.lastrowid,
                "build_id": build_id,
                "log_text": log_text,
                "status": status,
                "created_at": now,
            }

        return self._execute_with_retry(_add)

    def list_build_logs(self, build_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Queries stored build logs."""
        def _list(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            if build_id:
                rows = conn.execute(
                    "SELECT id, build_id, log_text, status, created_at FROM build_logs WHERE build_id = ? ORDER BY id ASC LIMIT ?",
                    (build_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, build_id, log_text, status, created_at FROM build_logs ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            return [
                {
                    "id": r["id"],
                    "build_id": r["build_id"],
                    "log_text": r["log_text"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

        return self._execute_with_retry(_list)

    def clear_build_logs(self) -> int:
        """Clears stored build logs."""
        def _clear(conn: sqlite3.Connection) -> int:
            cursor = conn.execute("DELETE FROM build_logs")
            return cursor.rowcount

        return self._execute_with_retry(_clear)

    # ── Projects ─────────────────────────────────────────────────────────────

    def list_projects(self) -> List[Dict[str, Any]]:
        """Lists all registered projects from SQLite."""
        def _list(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            rows = conn.execute(
                "SELECT id, name, path, config, created_at, updated_at FROM projects ORDER BY updated_at DESC"
            ).fetchall()
            out = []
            for r in rows:
                cfg = json.loads(r["config"]) if r["config"] else {}
                out.append({
                    "id": r["id"],
                    "name": r["name"],
                    "path": r["path"],
                    "config": cfg,
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                })
            return out

        return self._execute_with_retry(_list)

    def add_project(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """Saves or updates a project definition in SQLite."""
        proj_id = project.get("id") or f"proj_{int(time.time() * 1000)}"
        name = project.get("name") or "Untitled Project"
        path = self._require_path(project.get("path", str(Path.home() / name)), "project path")
        cfg_json = json.dumps(project.get("config", {}))
        now = self._now()

        def _add(conn: sqlite3.Connection) -> Dict[str, Any]:
            conn.execute(
                """
                INSERT INTO projects (id, name, path, config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    path = excluded.path,
                    config = excluded.config,
                    updated_at = excluded.updated_at
                ON CONFLICT(path) DO UPDATE SET
                    name = excluded.name,
                    config = excluded.config,
                    updated_at = excluded.updated_at
                """,
                (proj_id, name, path, cfg_json, now, now),
            )
            return {
                "id": proj_id,
                "name": name,
                "path": path,
                "config": project.get("config", {}),
                "created_at": now,
                "updated_at": now,
            }

        return self._execute_with_retry(_add)

    def delete_project(self, project_id: str) -> bool:
        """Deletes a project by ID or Path from SQLite."""
        def _delete(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute("DELETE FROM projects WHERE id = ? OR path = ?", (project_id, project_id))
            return cursor.rowcount > 0

        return self._execute_with_retry(_delete)

    # ── Scientific Research Entities ──────────────────────────────────────────
    
    # Investigations
    def list_investigations(self) -> List[Dict[str, Any]]:
        return self._list_json_items("investigations")

    def get_investigation(self, inv_id: str) -> Optional[Dict[str, Any]]:
        def _get(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
            row = conn.execute("SELECT payload FROM investigations WHERE item_id = ?", (inv_id,)).fetchone()
            return json.loads(row["payload"]) if row else None
        return self._execute_with_retry(_get)

    def save_investigation(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self._add_json_item("investigations", item)

    def delete_investigation(self, inv_id: str) -> bool:
        return self._delete_json_item("investigations", inv_id)

    # Hypotheses
    def list_hypotheses(self, investigation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        def _list(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            if investigation_id:
                rows = conn.execute(
                    "SELECT payload FROM hypotheses WHERE investigation_id = ? ORDER BY created_at DESC",
                    (investigation_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT payload FROM hypotheses ORDER BY created_at DESC").fetchall()
            return [json.loads(r["payload"]) for r in rows]
        return self._execute_with_retry(_list)

    def get_hypothesis(self, hyp_id: str) -> Optional[Dict[str, Any]]:
        def _get(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
            row = conn.execute("SELECT payload FROM hypotheses WHERE item_id = ?", (hyp_id,)).fetchone()
            return json.loads(row["payload"]) if row else None
        return self._execute_with_retry(_get)

    def save_hypothesis(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item_id = item.get("id")
        inv_id = item.get("investigation_id", "default")
        now = self._now()
        def _save(conn: sqlite3.Connection) -> Dict[str, Any]:
            conn.execute(
                """
                INSERT INTO hypotheses (item_id, investigation_id, payload, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    investigation_id = excluded.investigation_id,
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (item_id, inv_id, json.dumps(item), now),
            )
            return item
        return self._execute_with_retry(_save)

    def delete_hypothesis(self, hyp_id: str) -> bool:
        return self._delete_json_item("hypotheses", hyp_id)

    # Experiment Runs
    def list_experiment_runs(self, investigation_id: Optional[str] = None, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        def _list(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            if experiment_id and investigation_id:
                rows = conn.execute(
                    "SELECT payload FROM experiment_runs WHERE investigation_id = ? AND experiment_id = ? ORDER BY created_at DESC",
                    (investigation_id, experiment_id),
                ).fetchall()
            elif investigation_id:
                rows = conn.execute(
                    "SELECT payload FROM experiment_runs WHERE investigation_id = ? ORDER BY created_at DESC",
                    (investigation_id,),
                ).fetchall()
            elif experiment_id:
                rows = conn.execute(
                    "SELECT payload FROM experiment_runs WHERE experiment_id = ? ORDER BY created_at DESC",
                    (experiment_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT payload FROM experiment_runs ORDER BY created_at DESC").fetchall()
            return [json.loads(r["payload"]) for r in rows]
        return self._execute_with_retry(_list)

    def get_experiment_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        def _get(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
            row = conn.execute("SELECT payload FROM experiment_runs WHERE item_id = ?", (run_id,)).fetchone()
            return json.loads(row["payload"]) if row else None
        return self._execute_with_retry(_get)

    def save_experiment_run(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item_id = item.get("id")
        exp_id = item.get("experiment_id", "")
        inv_id = item.get("investigation_id", "")
        now = self._now()
        def _save(conn: sqlite3.Connection) -> Dict[str, Any]:
            conn.execute(
                """
                INSERT INTO experiment_runs (item_id, experiment_id, investigation_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    experiment_id = excluded.experiment_id,
                    investigation_id = excluded.investigation_id,
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (item_id, exp_id, inv_id, json.dumps(item), now),
            )
            return item
        return self._execute_with_retry(_save)

    list_runs = list_experiment_runs
    save_run = save_experiment_run

    # Evidence Records
    def list_evidence_records(self, investigation_id: Optional[str] = None, hypothesis_id: Optional[str] = None) -> List[Dict[str, Any]]:
        def _list(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            if hypothesis_id:
                rows = conn.execute(
                    "SELECT payload FROM evidence_records WHERE hypothesis_id = ? ORDER BY created_at DESC",
                    (hypothesis_id,),
                ).fetchall()
            elif investigation_id:
                rows = conn.execute(
                    "SELECT payload FROM evidence_records WHERE investigation_id = ? ORDER BY created_at DESC",
                    (investigation_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT payload FROM evidence_records ORDER BY created_at DESC").fetchall()
            return [json.loads(r["payload"]) for r in rows]
        return self._execute_with_retry(_list)

    def save_evidence_record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item_id = item.get("id")
        inv_id = item.get("investigation_id", "")
        hyp_id = item.get("hypothesis_id")
        now = self._now()
        def _save(conn: sqlite3.Connection) -> Dict[str, Any]:
            conn.execute(
                """
                INSERT INTO evidence_records (item_id, investigation_id, hypothesis_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    investigation_id = excluded.investigation_id,
                    hypothesis_id = excluded.hypothesis_id,
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (item_id, inv_id, hyp_id, json.dumps(item), now),
            )
            return item
        return self._execute_with_retry(_save)

    def delete_evidence_record(self, evidence_id: str) -> bool:
        return self._delete_json_item("evidence_records", evidence_id)

    # Mechanisms
    def list_mechanisms(self, investigation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        def _list(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            if investigation_id:
                rows = conn.execute(
                    "SELECT payload FROM mechanisms WHERE investigation_id = ? ORDER BY created_at DESC",
                    (investigation_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT payload FROM mechanisms ORDER BY created_at DESC").fetchall()
            return [json.loads(r["payload"]) for r in rows]
        return self._execute_with_retry(_list)

    def get_mechanism(self, mech_id: str) -> Optional[Dict[str, Any]]:
        def _get(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
            row = conn.execute("SELECT payload FROM mechanisms WHERE item_id = ?", (mech_id,)).fetchone()
            return json.loads(row["payload"]) if row else None
        return self._execute_with_retry(_get)

    def save_mechanism(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item_id = item.get("id")
        inv_id = item.get("investigation_id", "")
        now = self._now()
        def _save(conn: sqlite3.Connection) -> Dict[str, Any]:
            conn.execute(
                """
                INSERT INTO mechanisms (item_id, investigation_id, payload, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    investigation_id = excluded.investigation_id,
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (item_id, inv_id, json.dumps(item), now),
            )
            return item
        return self._execute_with_retry(_save)

    def delete_mechanism(self, mech_id: str) -> bool:
        return self._delete_json_item("mechanisms", mech_id)

    # Evidence aliases
    list_evidence = list_evidence_records
    save_evidence = save_evidence_record

    # Artifacts
    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        def _get(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
            row = conn.execute("SELECT payload FROM artifacts WHERE item_id = ?", (artifact_id,)).fetchone()
            return json.loads(row["payload"]) if row else None
        return self._execute_with_retry(_get)

    def list_artifacts(self, investigation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        def _list(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            if investigation_id:
                rows = conn.execute(
                    "SELECT payload FROM artifacts WHERE investigation_id = ? ORDER BY created_at DESC",
                    (investigation_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT payload FROM artifacts ORDER BY created_at DESC").fetchall()
            return [json.loads(r["payload"]) for r in rows]
        return self._execute_with_retry(_list)

    def save_artifact(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item_id = item.get("id")
        inv_id = item.get("investigation_id", "")
        now = self._now()
        def _save(conn: sqlite3.Connection) -> Dict[str, Any]:
            conn.execute(
                """
                INSERT INTO artifacts (item_id, investigation_id, payload, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    investigation_id = excluded.investigation_id,
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (item_id, inv_id, json.dumps(item), now),
            )
            return item
        return self._execute_with_retry(_save)

    # Jobs
    def list_jobs(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        def _list(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            rows = conn.execute("SELECT payload FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            items = [json.loads(r["payload"]) for r in rows]
            if status:
                items = [j for j in items if j.get("status") == status]
            return items
        return self._execute_with_retry(_list)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        def _get(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
            row = conn.execute("SELECT payload FROM jobs WHERE item_id = ?", (job_id,)).fetchone()
            return json.loads(row["payload"]) if row else None
        return self._execute_with_retry(_get)

    def save_job(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self._add_json_item("jobs", item)

    def update_job_status(self, job_id: str, status: str, progress: Optional[float] = None, error: Optional[str] = None) -> Optional[Dict[str, Any]]:
        job = self.get_job(job_id)
        if not job:
            return None
        job["status"] = status
        if progress is not None:
            job["progress"] = progress
        if error is not None:
            job["error"] = error
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            job["completed_at"] = time.time()
        return self.save_job(job)

    # ── Private JSON Item Helpers ────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _list_json_items(self, table: str) -> list[dict[str, Any]]:
        if table not in self.ALLOWED_TABLES:
            raise StorageError(f"Invalid table name: {table}")

        def _list(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                f"SELECT payload FROM {table} ORDER BY created_at, item_id"
            ).fetchall()
            return [json.loads(row["payload"]) for row in rows]

        return self._execute_with_retry(_list)

    def _get_json_item(self, table: str, item_id: str) -> Optional[dict[str, Any]]:
        if table not in self.ALLOWED_TABLES:
            raise StorageError(f"Invalid table name: {table}")

        def _get(conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
            row = conn.execute(
                f"SELECT payload FROM {table} WHERE item_id = ?", (item_id,)
            ).fetchone()
            return json.loads(row["payload"]) if row else None

        return self._execute_with_retry(_get)

    def _add_json_item(self, table: str, item: dict[str, Any]) -> dict[str, Any]:
        if table not in self.ALLOWED_TABLES:
            raise StorageError(f"Invalid table name: {table}")
        if not isinstance(item, dict):
            raise StorageError(f"{table} item must be an object")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise StorageError(f"{table} item requires a string 'id'")

        def _add(conn: sqlite3.Connection) -> dict[str, Any]:
            conn.execute(
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

        return self._execute_with_retry(_add)

    def _delete_json_item(self, table: str, item_id: str) -> bool:
        if table not in self.ALLOWED_TABLES:
            raise StorageError(f"Invalid table name: {table}")
        if not isinstance(item_id, str) or not item_id.strip():
            raise StorageError(f"{table} item id must be a string")

        def _del(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(f"DELETE FROM {table} WHERE item_id = ?", (item_id,))
            return cursor.rowcount > 0

        return self._execute_with_retry(_del)

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

