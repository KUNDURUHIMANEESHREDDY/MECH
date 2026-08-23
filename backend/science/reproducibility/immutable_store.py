"""Immutable SQLite Store for Scientific Experiment Runs and Provenance Lineage.

Enforces append-only storage and cryptographic SHA-256 manifest verification.
Once an experiment run is saved, its record is immutable.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    ExecutionEnvironment,
    ExperimentSpecification,
    ImmutableExperimentRun,
    ModelIdentity,
    ProvenanceChain,
)

import logging
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "experiment_runs.db"


class ImmutableExperimentStore:
    """Thread-safe SQLite store for immutable experiment runs with lineage."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        if db_path is None:
            self._db_path = str(DEFAULT_DB_PATH)
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        else:
            self._db_path = str(db_path)
            if self._db_path != ":memory:":
                os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    parent_run_id TEXT,
                    experiment_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    model_json TEXT NOT NULL,
                    environment_json TEXT NOT NULL,
                    specification_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    measurements_json TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    canonical_spec_hash TEXT,
                    strategy_json TEXT,
                    telemetry_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Migration check for existing databases
            try:
                conn.execute("ALTER TABLE experiment_runs ADD COLUMN strategy_json TEXT;")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Swallowed exception: %s", exc)
            try:
                conn.execute("ALTER TABLE experiment_runs ADD COLUMN telemetry_json TEXT;")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Swallowed exception: %s", exc)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_run_id ON experiment_runs(parent_run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiment_type ON experiment_runs(experiment_type);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_canonical_spec_hash ON experiment_runs(canonical_spec_hash);")
            conn.commit()

    def _row_to_run(self, row: sqlite3.Row) -> ImmutableExperimentRun:
        model = ModelIdentity(**json.loads(row["model_json"]))
        env = ExecutionEnvironment(**json.loads(row["environment_json"]))
        spec_dict = json.loads(row["specification_json"])
        prov_dict = json.loads(row["provenance_json"])
        meas_dict = json.loads(row["measurements_json"])

        # Handle intervention_impl in specification
        if "intervention_impl" in spec_dict and isinstance(spec_dict["intervention_impl"], dict):
            from .types import InterventionImplementation
            spec_dict["intervention_impl"] = InterventionImplementation(**spec_dict["intervention_impl"])
        spec = ExperimentSpecification(**spec_dict)
        prov = ProvenanceChain(**prov_dict)

        # Parse strategy and telemetry
        from .types import ExecutionTelemetry, RuntimeStrategy
        strat = RuntimeStrategy()
        if "strategy_json" in row.keys() and row["strategy_json"]:
            strat = RuntimeStrategy(**json.loads(row["strategy_json"]))

        telemetry = None
        if "telemetry_json" in row.keys() and row["telemetry_json"]:
            telemetry = ExecutionTelemetry(**json.loads(row["telemetry_json"]))

        spec_hash = row["canonical_spec_hash"] if "canonical_spec_hash" in row.keys() else ""

        return ImmutableExperimentRun(
            run_id=row["run_id"],
            parent_run_id=row["parent_run_id"],
            experiment_type=row["experiment_type"],
            title=row["title"],
            timestamp_utc=row["timestamp_utc"],
            model=model,
            environment=env,
            specification=spec,
            provenance_chain=prov,
            measurements=meas_dict,
            verdict=row["verdict"],
            runtime_strategy=strat,
            execution_telemetry=telemetry,
            manifest_sha256=row["manifest_sha256"],
            canonical_spec_hash=spec_hash or "",
        )

    def save_run(self, run: ImmutableExperimentRun) -> ImmutableExperimentRun:
        """Saves an immutable experiment run. Enforces manifest hash calculation."""
        spec_hash = run.canonical_spec_hash or run.calculate_canonical_spec_hash()
        computed_hash = run.calculate_manifest_hash()
        signed_run = ImmutableExperimentRun(
            run_id=run.run_id,
            parent_run_id=run.parent_run_id,
            experiment_type=run.experiment_type,
            title=run.title,
            timestamp_utc=run.timestamp_utc,
            model=run.model,
            environment=run.environment,
            specification=run.specification,
            provenance_chain=run.provenance_chain,
            measurements=run.measurements,
            verdict=run.verdict,
            runtime_strategy=run.runtime_strategy,
            execution_telemetry=run.execution_telemetry,
            manifest_sha256=computed_hash,
            canonical_spec_hash=spec_hash,
        )

        with self._get_connection() as conn:
            # Enforce that existing run_ids cannot be mutated/overwritten
            cursor = conn.execute("SELECT run_id FROM experiment_runs WHERE run_id = ?", (signed_run.run_id,))
            if cursor.fetchone() is not None:
                raise ValueError(
                    f"Experiment run '{signed_run.run_id}' already exists and is immutable. "
                    "Create a child run with parent_run_id instead of mutating."
                )

            conn.execute(
                """
                INSERT INTO experiment_runs (
                    run_id, parent_run_id, experiment_type, title, timestamp_utc,
                    model_json, environment_json, specification_json, provenance_json,
                    measurements_json, verdict, manifest_sha256, canonical_spec_hash,
                    strategy_json, telemetry_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signed_run.run_id,
                    signed_run.parent_run_id,
                    signed_run.experiment_type,
                    signed_run.title,
                    signed_run.timestamp_utc,
                    json.dumps(signed_run.model.to_dict()),
                    json.dumps(signed_run.environment.to_dict()),
                    json.dumps(signed_run.specification.to_dict()),
                    json.dumps(signed_run.provenance_chain.to_dict()),
                    json.dumps(signed_run.measurements),
                    signed_run.verdict,
                    signed_run.manifest_sha256,
                    signed_run.canonical_spec_hash,
                    json.dumps(signed_run.runtime_strategy.to_dict()),
                    json.dumps(signed_run.execution_telemetry.to_dict()) if signed_run.execution_telemetry else None,
                ),
            )
            conn.commit()

        return signed_run

    def find_by_spec_hash(self, spec_hash: str) -> List[ImmutableExperimentRun]:
        """Finds runs matching the exact canonical experiment specification hash."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM experiment_runs WHERE canonical_spec_hash = ?", (spec_hash,))
            return [self._row_to_run(r) for r in cursor.fetchall()]


    def get_run(self, run_id: str) -> Optional[ImmutableExperimentRun]:
        """Retrieves a single experiment run by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_run(row)

    def list_runs(
        self, limit: int = 50, experiment_type: Optional[str] = None
    ) -> List[ImmutableExperimentRun]:
        """Lists runs sorted by timestamp descending."""
        with self._get_connection() as conn:
            if experiment_type:
                cursor = conn.execute(
                    "SELECT * FROM experiment_runs WHERE experiment_type = ? ORDER BY created_at DESC LIMIT ?",
                    (experiment_type, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM experiment_runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            return [self._row_to_run(r) for r in cursor.fetchall()]

    def get_lineage(self, run_id: str) -> List[ImmutableExperimentRun]:
        """Traces the complete ancestry and immediate child lineage of a run."""
        target = self.get_run(run_id)
        if not target:
            return []

        lineage: List[ImmutableExperimentRun] = []
        # Walk up ancestry
        current = target
        while current.parent_run_id:
            parent = self.get_run(current.parent_run_id)
            if not parent:
                break
            lineage.insert(0, parent)
            current = parent

        # Add target
        lineage.append(target)

        # Add direct children
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM experiment_runs WHERE parent_run_id = ?", (run_id,))
            children = [self._row_to_run(r) for r in cursor.fetchall()]
            lineage.extend(children)

        return lineage

    def verify_integrity(self, run_id: str) -> Tuple[bool, str]:
        """Re-computes the SHA-256 manifest hash and verifies against stored manifest."""
        run = self.get_run(run_id)
        if not run:
            return False, f"Run '{run_id}' not found"

        expected = run.manifest_sha256
        actual = run.calculate_manifest_hash()
        if expected == actual:
            return True, f"Integrity verified (SHA-256: {actual})"
        return False, f"Integrity violation: expected {expected}, calculated {actual}"
