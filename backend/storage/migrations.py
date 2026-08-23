"""Simple migration framework for SQLite databases."""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List


MIGRATIONS: Dict[str, List[Callable[[sqlite3.Connection], None]]] = {
    "mech.db": [
        # v1: Initial schema (already exists) + version table
        lambda c: c.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                db_name TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            )
        """),
    ],
    "warm_model.db": [
        # v1: Initial warm model schema
        lambda c: c.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (
                db_name TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS warm_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                layer INTEGER,
                feature_idx INTEGER,
                semantic_label TEXT,
                description TEXT,
                confidence REAL,
                activation_freq REAL,
                top_tokens_json TEXT,
                discovered_by TEXT,
                created_at TEXT,
                UNIQUE(model_id, layer, feature_idx)
            );
            CREATE TABLE IF NOT EXISTS warm_circuits (
                circuit_id TEXT PRIMARY KEY,
                model_id TEXT,
                name TEXT,
                task_name TEXT,
                nodes_json TEXT,
                edges_json TEXT,
                faithfulness_score REAL,
                recovery_score REAL,
                discovered_by TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_wf_model_layer ON warm_features(model_id, layer);
            CREATE INDEX IF NOT EXISTS idx_wc_model ON warm_circuits(model_id);
        """),
    ],
    "artifacts.db": [
        # v1: Initial artifacts schema (already exists) + version table
        lambda c: c.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                db_name TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            )
        """),
    ],
}


def migrate(db_path: str, migrations: List[Callable[[sqlite3.Connection], None]]) -> int:
    """Applies pending migrations. Returns new version number."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    # Ensure version table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            db_name TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            applied_at TEXT NOT NULL
        )
    """)

    db_name = Path(db_path).name
    cursor = conn.execute("SELECT version FROM schema_version WHERE db_name = ?", (db_name,))
    row = cursor.fetchone()
    current_version = row[0] if row else 0

    for version, migration in enumerate(migrations, start=1):
        if version > current_version:
            migration(conn)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (db_name, version, applied_at) VALUES (?, ?, ?)",
                (db_name, version, datetime.now().isoformat())
            )
            conn.commit()
            print(f"Applied migration v{version} to {db_name}")

    conn.close()
    return max(current_version, len(migrations))


def backup_db(db_path: str, backup_dir: str = "backend/storage/backups") -> str:
    """Creates a timestamped backup using SQLite backup API."""
    Path(backup_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(backup_dir) / f"{Path(db_path).stem}_{timestamp}.db"

    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(str(backup_path))
    src.backup(dst)
    src.close()
    dst.close()

    # Keep only last 10 backups
    backups = sorted(Path(backup_dir).glob(f"{Path(db_path).stem}_*.db"))
    for old in backups[:-10]:
        old.unlink()

    return str(backup_path)


def get_schema_version(db_path: str) -> int:
    """Returns current schema version for a database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT version FROM schema_version WHERE db_name = ?", (Path(db_path).name,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0