import sqlite3
import tempfile
from pathlib import Path
import pytest

from backend.storage.migrations import migrate, backup_db
from backend.storage.database import DesktopStorage


def test_fresh_database_migration():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "fresh.db"
        storage = DesktopStorage(db_path)
        storage.initialize()

        # Check all tables were created
        with storage._connect() as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            assert "investigations" in tables
            assert "hypotheses" in tables
            assert "experiment_runs" in tables
            assert "evidence_records" in tables
            assert "mechanisms" in tables
            assert "artifacts" in tables
            assert "jobs" in tables


def test_database_upgrade_progression():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = str(Path(tmpdir) / "upgrade_test.db")

        # Step 1: Migration v1
        mig_v1 = [
            lambda c: c.execute("CREATE TABLE test_table_v1 (id TEXT PRIMARY KEY, val TEXT)"),
        ]
        v = migrate(db_path, mig_v1)
        assert v == 1

        # Step 2: Migration v2
        mig_v2 = [
            lambda c: c.execute("CREATE TABLE test_table_v1 (id TEXT PRIMARY KEY, val TEXT)"),
            lambda c: c.execute("CREATE TABLE test_table_v2 (id TEXT PRIMARY KEY, num INTEGER)"),
        ]
        v2 = migrate(db_path, mig_v2)
        assert v2 == 2

        # Verify both tables exist
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "test_table_v1" in tables
        assert "test_table_v2" in tables
        conn.close()


def test_database_backup():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = str(Path(tmpdir) / "original.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE sample (id INT)")
        conn.execute("INSERT INTO sample VALUES (42)")
        conn.commit()
        conn.close()

        backup_file = backup_db(db_path)
        assert backup_file is not None
        assert Path(backup_file).exists()

        # Read from backup
        conn_b = sqlite3.connect(backup_file)
        row = conn_b.execute("SELECT id FROM sample").fetchone()
        assert row[0] == 42
        conn_b.close()
