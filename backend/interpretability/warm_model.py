"""Warm Model Knowledge Base.

Maintains persistent, cumulative interpretability knowledge across research sessions:
- Verified Circuit Candidates & Subgraphs
- Semantic SAE Feature Dictionaries & Behavioral Labels
- Causal Intervention Profiles & Faithfulness Records
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MECH.warm_model")

try:
    from backend.storage.migrations import migrate, MIGRATIONS, backup_db
except ImportError:
    migrate = MIGRATIONS = backup_db = None


class EdgeEvidenceState:
    OBSERVED = "OBSERVED"
    CANDIDATE = "CANDIDATE"
    SUPPORTED = "SUPPORTED"
    CAUSALLY_VERIFIED = "CAUSALLY_VERIFIED"


@dataclass
class WarmFeatureRecord:
    """Discovered candidate computational feature definition."""
    model_id: str
    layer: int
    feature_idx: int
    semantic_label: str
    description: str = ""
    confidence: float = 0.9
    activation_freq: float = 0.0
    specificity: float = 0.85
    consistency: float = 0.88
    stability: float = 0.82
    causal_effect: float = 0.75
    is_substrate_reference: bool = False
    top_activating_tokens: List[str] = field(default_factory=list)
    discovered_by: str = "auto_feature_discovery"
    created_at: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WarmCircuitRecord:
    """Verified causal circuit subgraph connecting candidate SAE features and components."""
    circuit_id: str
    model_id: str
    name: str
    task_name: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    faithfulness_score: float = 0.0
    recovery_score: float = 0.0
    evidence_state: str = EdgeEvidenceState.SUPPORTED
    discovered_by: str = "acdc_search"
    created_at: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WarmModelKnowledgeBase:
    """Persistent database for accumulated model interpretability discoveries."""

    def __init__(self, db_path: str = "backend/storage/warm_model.db") -> None:
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        if migrate and MIGRATIONS and backup_db:
            backup_db(self.db_path)
            migrate(self.db_path, MIGRATIONS["warm_model.db"])
        
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
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
                )
            """)
            conn.execute("""
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
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_wf_model_layer ON warm_features(model_id, layer)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_wc_model ON warm_circuits(model_id)")

    def register_feature(self, record: WarmFeatureRecord) -> int:
        """Saves or updates a discovered semantic feature."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO warm_features (
                    model_id, layer, feature_idx, semantic_label, description,
                    confidence, activation_freq, top_tokens_json, discovered_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id, layer, feature_idx) DO UPDATE SET
                    semantic_label = excluded.semantic_label,
                    description = excluded.description,
                    confidence = excluded.confidence,
                    activation_freq = excluded.activation_freq,
                    top_tokens_json = excluded.top_tokens_json
            """, (
                record.model_id,
                record.layer,
                record.feature_idx,
                record.semantic_label,
                record.description,
                record.confidence,
                record.activation_freq,
                json.dumps(record.top_activating_tokens),
                record.discovered_by,
                record.created_at,
            ))
            return cursor.lastrowid or 0

    def get_features(self, model_id: str, layer: Optional[int] = None) -> List[WarmFeatureRecord]:
        """Retrieves learned features for a model."""
        query = "SELECT model_id, layer, feature_idx, semantic_label, description, confidence, activation_freq, top_tokens_json, discovered_by, created_at FROM warm_features WHERE model_id = ?"
        params: List[Any] = [model_id]
        if layer is not None:
            query += " AND layer = ?"
            params.append(layer)

        features = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            for row in cursor.fetchall():
                features.append(WarmFeatureRecord(
                    model_id=row[0],
                    layer=row[1],
                    feature_idx=row[2],
                    semantic_label=row[3],
                    description=row[4],
                    confidence=row[5],
                    activation_freq=row[6],
                    top_activating_tokens=json.loads(row[7]) if row[7] else [],
                    discovered_by=row[8],
                    created_at=row[9],
                ))
        return features

    def register_circuit(self, circuit: WarmCircuitRecord) -> str:
        """Saves a discovered causal circuit."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO warm_circuits (
                    circuit_id, model_id, name, task_name, nodes_json,
                    edges_json, faithfulness_score, recovery_score, discovered_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                circuit.circuit_id,
                circuit.model_id,
                circuit.name,
                circuit.task_name,
                json.dumps(circuit.nodes),
                json.dumps(circuit.edges),
                circuit.faithfulness_score,
                circuit.recovery_score,
                circuit.discovered_by,
                circuit.created_at,
            ))
        return circuit.circuit_id

    def get_circuits(self, model_id: Optional[str] = None, task_name: Optional[str] = None) -> List[WarmCircuitRecord]:
        """Retrieves verified circuits."""
        query = "SELECT circuit_id, model_id, name, task_name, nodes_json, edges_json, faithfulness_score, recovery_score, discovered_by, created_at FROM warm_circuits WHERE 1=1"
        params: List[Any] = []
        if model_id is not None:
            query += " AND model_id = ?"
            params.append(model_id)
        if task_name is not None:
            query += " AND task_name = ?"
            params.append(task_name)

        circuits = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            for row in cursor.fetchall():
                circuits.append(WarmCircuitRecord(
                    circuit_id=row[0],
                    model_id=row[1],
                    name=row[2],
                    task_name=row[3],
                    nodes=json.loads(row[4]) if row[4] else [],
                    edges=json.loads(row[5]) if row[5] else [],
                    faithfulness_score=row[6],
                    recovery_score=row[7],
                    discovered_by=row[8],
                    created_at=row[9],
                ))
        return circuits

    def get_model_profile(self, model_id: str) -> Dict[str, Any]:
        """Returns statistical overview of accumulated knowledge for a model."""
        features = self.get_features(model_id)
        circuits = self.get_circuits(model_id)
        layers_covered = sorted(list({f.layer for f in features}))
        return {
            "model_id": model_id,
            "total_features_labeled": len(features),
            "total_verified_circuits": len(circuits),
            "layers_analyzed": layers_covered,
            "circuits": [c.name for c in circuits],
        }


_global_warm_model_kb: Optional[WarmModelKnowledgeBase] = None


def get_warm_model_kb() -> WarmModelKnowledgeBase:
    global _global_warm_model_kb
    if _global_warm_model_kb is None:
        _global_warm_model_kb = WarmModelKnowledgeBase()
    return _global_warm_model_kb
