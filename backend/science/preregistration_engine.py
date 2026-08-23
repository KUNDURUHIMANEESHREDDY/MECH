"""Experiment Preregistration, Protocol Freezing, and Immutable Ledger for MECH.

Guarantees scientific integrity by allowing researchers to preregister predictions,
metrics, and falsification conditions before execution, preventing HARKing
(Hypothesizing After Results are Known).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.storage.database import DesktopStorage

logger = logging.getLogger("MECH.science.preregistration")


class PreregistrationLockedError(Exception):
    """Raised when an execution or modification violates a locked scientific preregistration."""
    pass


class PreregistrationRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"prereg_{uuid.uuid4().hex[:10]}")
    investigation_id: str
    hypothesis_id: str
    hypothesis_title: str
    prediction_statement: str
    target_component: str
    negative_control_component: str
    primary_metric: str
    falsification_condition: str
    falsification_threshold: float
    sample_size: int = 100
    is_locked: bool = False
    locked_at: Optional[float] = None
    created_at: float = Field(default_factory=time.time)


class LedgerEntry(BaseModel):
    id: str = Field(default_factory=lambda: f"led_{uuid.uuid4().hex[:10]}")
    investigation_id: str
    preregistration_id: Optional[str] = None
    run_id: str
    mode: str  # EXPLORATORY or CONFIRMATORY_LOCKED
    prediction_matched: bool
    empirical_result_summary: str
    falsification_evaluated: bool
    is_falsified: bool
    timestamp: float = Field(default_factory=time.time)


class PreregistrationEngine:
    """Manages protocol freezing, preregistrations, and immutable research ledgers."""

    def __init__(self, storage: Optional[DesktopStorage] = None) -> None:
        from pathlib import Path
        db_path = Path.home() / ".cache" / "neural-debugger" / "mech.db"
        self.storage = storage or DesktopStorage(db_path)
        self.storage.initialize()
        self._init_tables()

    def _init_tables(self) -> None:
        with self.storage.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preregistrations (
                    id TEXT PRIMARY KEY,
                    investigation_id TEXT NOT NULL,
                    hypothesis_id TEXT NOT NULL,
                    hypothesis_title TEXT,
                    prediction_statement TEXT,
                    target_component TEXT,
                    negative_control_component TEXT,
                    primary_metric TEXT,
                    falsification_condition TEXT,
                    falsification_threshold REAL,
                    sample_size INTEGER,
                    is_locked INTEGER DEFAULT 0,
                    locked_at REAL,
                    created_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_ledger (
                    id TEXT PRIMARY KEY,
                    investigation_id TEXT NOT NULL,
                    preregistration_id TEXT,
                    run_id TEXT NOT NULL,
                    mode TEXT,
                    prediction_matched INTEGER,
                    empirical_result_summary TEXT,
                    falsification_evaluated INTEGER,
                    is_falsified INTEGER,
                    timestamp REAL
                )
            """)
            conn.commit()

    def create_preregistration(self, record: PreregistrationRecord) -> PreregistrationRecord:
        """Registers a new experimental protocol proposal or rejects mutation if locked."""
        existing = self.get_preregistration(record.id)
        if existing and existing.is_locked:
            raise PreregistrationLockedError(
                f"HARKing Protection: Cannot modify preregistration {record.id} because it has been permanently locked."
            )

        with self.storage.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO preregistrations 
                (id, investigation_id, hypothesis_id, hypothesis_title, prediction_statement,
                 target_component, negative_control_component, primary_metric, falsification_condition,
                 falsification_threshold, sample_size, is_locked, locked_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id, record.investigation_id, record.hypothesis_id, record.hypothesis_title,
                record.prediction_statement, record.target_component, record.negative_control_component,
                record.primary_metric, record.falsification_condition, record.falsification_threshold,
                record.sample_size, 1 if record.is_locked else 0, record.locked_at, record.created_at,
            ))
            conn.commit()
        return record

    def lock_preregistration(self, prereg_id: str) -> PreregistrationRecord:
        """Permanently freezes an experimental protocol prior to confirmatory execution."""
        rec = self.get_preregistration(prereg_id)
        if not rec:
            raise ValueError(f"Preregistration {prereg_id} not found.")

        rec.is_locked = True
        rec.locked_at = time.time()

        with self.storage.get_connection() as conn:
            conn.execute("""
                UPDATE preregistrations SET is_locked = 1, locked_at = ? WHERE id = ?
            """, (rec.locked_at, prereg_id))
            conn.commit()

        logger.info("Permanently locked experimental preregistration %s", prereg_id)
        return rec

    def get_preregistration(self, prereg_id: str) -> Optional[PreregistrationRecord]:
        with self.storage.get_connection() as conn:
            row = conn.execute("SELECT * FROM preregistrations WHERE id = ?", (prereg_id,)).fetchone()
            if not row:
                return None
            return PreregistrationRecord(
                id=row[0], investigation_id=row[1], hypothesis_id=row[2], hypothesis_title=row[3],
                prediction_statement=row[4], target_component=row[5], negative_control_component=row[6],
                primary_metric=row[7], falsification_condition=row[8], falsification_threshold=row[9],
                sample_size=row[10], is_locked=bool(row[11]), locked_at=row[12], created_at=row[13],
            )

    def record_ledger_entry(self, entry: LedgerEntry) -> LedgerEntry:
        """Appends an immutable execution outcome to the research ledger."""
        with self.storage.get_connection() as conn:
            conn.execute("""
                INSERT INTO experiment_ledger 
                (id, investigation_id, preregistration_id, run_id, mode, prediction_matched,
                 empirical_result_summary, falsification_evaluated, is_falsified, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id, entry.investigation_id, entry.preregistration_id, entry.run_id,
                entry.mode, 1 if entry.prediction_matched else 0, entry.empirical_result_summary,
                1 if entry.falsification_evaluated else 0, 1 if entry.is_falsified else 0, entry.timestamp,
            ))
            conn.commit()
        return entry

    def list_ledger_entries(self, investigation_id: str) -> List[LedgerEntry]:
        with self.storage.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM experiment_ledger WHERE investigation_id = ? ORDER BY timestamp ASC",
                (investigation_id,)
            ).fetchall()
            return [
                LedgerEntry(
                    id=r[0], investigation_id=r[1], preregistration_id=r[2], run_id=r[3],
                    mode=r[4], prediction_matched=bool(r[5]), empirical_result_summary=r[6],
                    falsification_evaluated=bool(r[7]), is_falsified=bool(r[8]), timestamp=r[9],
                )
                for r in rows
            ]


# Global preregistration engine singleton
preregistration_engine = PreregistrationEngine()
