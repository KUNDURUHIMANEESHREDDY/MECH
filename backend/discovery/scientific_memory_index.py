r"""Scientific Memory Index & Content-Addressed Knowledge Graph for MECH.

Associative memory tracking past inquiries, experiments, residuals, and falsifications:
    Query: "Have I already run an experiment for this uncertainty?" -> Prevents redundant experiments (RER <= 10.0%)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set


@dataclass
class MemoryRecord:
    memory_key: str
    question_statement: str
    experiment_id: str
    observed_outcome: float
    residual_recorded: float
    theories_falsified: List[str]
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_key": self.memory_key,
            "question_statement": self.question_statement,
            "experiment_id": self.experiment_id,
            "observed_outcome": round(self.observed_outcome, 4),
            "residual_recorded": round(self.residual_recorded, 4),
            "theories_falsified": self.theories_falsified,
            "timestamp_utc": self.timestamp_utc,
        }


class ScientificMemoryIndex:
    """Provides content-addressed retrieval of past scientific investigations."""

    def __init__(self) -> None:
        self.memories: Dict[str, MemoryRecord] = {}
        self.experiment_signatures: Set[str] = set()

    def record_investigation(
        self,
        question_statement: str,
        experiment_id: str,
        observed_outcome: float,
        residual_recorded: float,
        theories_falsified: List[str],
    ) -> MemoryRecord:
        """Stores a completed investigation into content-addressed memory."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        key_raw = f"{question_statement}_{experiment_id}"
        mem_key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:12]

        record = MemoryRecord(
            memory_key=mem_key,
            question_statement=question_statement,
            experiment_id=experiment_id,
            observed_outcome=observed_outcome,
            residual_recorded=residual_recorded,
            theories_falsified=theories_falsified,
            timestamp_utc=ts,
        )
        self.memories[mem_key] = record
        self.experiment_signatures.add(experiment_id)
        return record

    def is_experiment_redundant(self, experiment_id: str) -> bool:
        """Checks if the proposed experiment has already been executed."""
        return experiment_id in self.experiment_signatures
