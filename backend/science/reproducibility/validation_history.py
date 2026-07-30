"""Validation History Logger — Reference Promotion & Timeline Support.

Tracks Milestone A execution histories across versions to detect regressions.
Supports "Reference Baseline" promotion to freeze stable local results.
"""
import sqlite3
import json
import csv
import shutil
import os
from typing import Dict, Any, List, Optional
import datetime as _dt

class ValidationHistoryLogger:
    def __init__(self, db_path: str = "validation_history.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    timestamp TEXT,
                    git_sha TEXT,
                    model TEXT,
                    model_revision TEXT,
                    benchmark TEXT,
                    status TEXT,
                    expected_metric REAL,
                    observed_metric REAL,
                    difference REAL,
                    runtime_sec REAL,
                    peak_vram_gb REAL,
                    seed INTEGER,
                    is_reference_baseline INTEGER DEFAULT 0,
                    reference_tag TEXT
                )
            """)
            conn.commit()
            
    def record_run(self, metadata: Dict[str, Any]):
        """Records a single benchmark run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO validation_runs (
                    run_id, timestamp, git_sha, model, model_revision, benchmark, 
                    status, expected_metric, observed_metric, difference, 
                    runtime_sec, peak_vram_gb, seed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.get("run_id"),
                metadata.get("timestamp"),
                metadata.get("git_sha"),
                metadata.get("model"),
                metadata.get("model_revision"),
                metadata.get("benchmark"),
                metadata.get("status"),
                metadata.get("expected_metric"),
                metadata.get("observed_metric"),
                metadata.get("difference"),
                metadata.get("runtime_sec"),
                metadata.get("peak_vram_gb"),
                metadata.get("seed")
            ))
            conn.commit()

    def promote_to_reference(self, run_id: str, tag: str = "v1.0"):
        """Promotes a successful run to be the official Reference Baseline."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Unset previous reference for the same benchmark/model
            cursor.execute("SELECT benchmark, model FROM validation_runs WHERE run_id = ?", (run_id,))
            res = cursor.fetchone()
            if not res: return

            cursor.execute("""
                UPDATE validation_runs SET is_reference_baseline = 0
                WHERE benchmark = ? AND model = ?
            """, (res[0], res[1]))

            # Set new reference
            cursor.execute("""
                UPDATE validation_runs SET is_reference_baseline = 1, reference_tag = ?
                WHERE run_id = ?
            """, (tag, run_id))
            conn.commit()

    def get_reference_baseline(self, benchmark: str, model: str) -> Optional[Dict[str, Any]]:
        """Fetch the official Reference Baseline for a given benchmark and model."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM validation_runs 
                WHERE benchmark = ? AND model = ? AND is_reference_baseline = 1
                LIMIT 1
            """, (benchmark, model))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_benchmark_timeline(self, benchmark: str, model: str) -> List[Dict[str, Any]]:
        """Returns all historic runs for trend analysis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM validation_runs
                WHERE benchmark = ? AND model = ?
                ORDER BY timestamp ASC
            """, (benchmark, model))
            return [dict(r) for r in cursor.fetchall()]
