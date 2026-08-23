"""Mechanism Versioning, Lineage Tracking, and Diff Analysis for MECH.

Maintains an immutable historical graph of mechanism circuit definitions as research
evolves from initial hypothesis to multi-component validated subgraphs.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.storage.database import DesktopStorage

logger = logging.getLogger("MECH.science.mechanism_versioning")


class MechanismVersion(BaseModel):
    id: str = Field(default_factory=lambda: f"mver_{uuid.uuid4().hex[:10]}")
    investigation_id: str
    version_number: int = 1
    parent_version_id: Optional[str] = None
    title: str
    components: List[str] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    scientific_rationale: str = ""
    created_at: float = Field(default_factory=time.time)


class MechanismDiff(BaseModel):
    v1_id: str
    v2_id: str
    v1_number: int
    v2_number: int
    added_components: List[str]
    removed_components: List[str]
    added_edges: List[Dict[str, Any]]
    removed_edges: List[Dict[str, Any]]
    new_evidence_citations: List[str]
    rationale: str
    timestamp: float = Field(default_factory=time.time)


def compute_mechanism_diff(v1: MechanismVersion, v2: MechanismVersion) -> MechanismDiff:
    """Calculates structural and empirical differential between two mechanism versions."""
    s1_comp = set(v1.components)
    s2_comp = set(v2.components)

    added_comp = sorted(list(s2_comp - s1_comp))
    removed_comp = sorted(list(s1_comp - s2_comp))

    e1_keys = {f"{e.get('source')}->{e.get('target')}" for e in v1.edges}
    e2_keys = {f"{e.get('source')}->{e.get('target')}" for e in v2.edges}

    added_edges = [e for e in v2.edges if f"{e.get('source')}->{e.get('target')}" not in e1_keys]
    removed_edges = [e for e in v1.edges if f"{e.get('source')}->{e.get('target')}" not in e2_keys]

    new_evi = sorted(list(set(v2.evidence_ids) - set(v1.evidence_ids)))

    return MechanismDiff(
        v1_id=v1.id,
        v2_id=v2.id,
        v1_number=v1.version_number,
        v2_number=v2.version_number,
        added_components=added_comp,
        removed_components=removed_comp,
        added_edges=added_edges,
        removed_edges=removed_edges,
        new_evidence_citations=new_evi,
        rationale=v2.scientific_rationale or f"Progression from v{v1.version_number} to v{v2.version_number}",
    )


class MechanismVersioningEngine:
    """Manages mechanism version history and evolution diffs."""

    def __init__(self, storage: Optional[DesktopStorage] = None) -> None:
        from pathlib import Path
        db_path = Path.home() / ".cache" / "neural-debugger" / "mech.db"
        self.storage = storage or DesktopStorage(db_path)
        self.storage.initialize()
        self._init_tables()

    def _init_tables(self) -> None:
        import json
        with self.storage.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mechanism_versions (
                    id TEXT PRIMARY KEY,
                    investigation_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    parent_version_id TEXT,
                    title TEXT NOT NULL,
                    components_json TEXT,
                    edges_json TEXT,
                    evidence_ids_json TEXT,
                    scientific_rationale TEXT,
                    created_at REAL
                )
            """)
            conn.commit()

    def save_version(self, version: MechanismVersion) -> MechanismVersion:
        import json
        with self.storage.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO mechanism_versions
                (id, investigation_id, version_number, parent_version_id, title,
                 components_json, edges_json, evidence_ids_json, scientific_rationale, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.id, version.investigation_id, version.version_number, version.parent_version_id,
                version.title, json.dumps(version.components), json.dumps(version.edges),
                json.dumps(version.evidence_ids), version.scientific_rationale, version.created_at,
            ))
            conn.commit()
        return version

    def get_version(self, version_id: str) -> Optional[MechanismVersion]:
        import json
        with self.storage.get_connection() as conn:
            row = conn.execute("SELECT * FROM mechanism_versions WHERE id = ?", (version_id,)).fetchone()
            if not row:
                return None
            return MechanismVersion(
                id=row[0], investigation_id=row[1], version_number=row[2], parent_version_id=row[3],
                title=row[4], components=json.loads(row[5] or "[]"), edges=json.loads(row[6] or "[]"),
                evidence_ids=json.loads(row[7] or "[]"), scientific_rationale=row[8] or "", created_at=row[9],
            )

    def list_versions(self, investigation_id: str) -> List[MechanismVersion]:
        import json
        with self.storage.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM mechanism_versions WHERE investigation_id = ? ORDER BY version_number ASC",
                (investigation_id,)
            ).fetchall()
            return [
                MechanismVersion(
                    id=r[0], investigation_id=r[1], version_number=r[2], parent_version_id=r[3],
                    title=r[4], components=json.loads(r[5] or "[]"), edges=json.loads(r[6] or "[]"),
                    evidence_ids=json.loads(r[7] or "[]"), scientific_rationale=r[8] or "", created_at=r[9],
                )
                for r in rows
            ]


# Global mechanism versioning engine singleton
mechanism_versioning_engine = MechanismVersioningEngine()
