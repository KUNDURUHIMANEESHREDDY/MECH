r"""Prospective Generalization Protocol for Combinatorial Held-Out Challenges.

Defines immutable prospective prediction contracts, challenge suite export formats,
and zero-knowledge observation ingestion structures.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TransferClass(str, Enum):
    TRANSFER = "TRANSFER"
    NEGATIVE_TRANSFER = "NEGATIVE_TRANSFER"
    ABSTAIN = "ABSTAIN"


@dataclass
class ProspectivePrediction:
    prediction_id: str
    source_model: str
    target_model: str
    task_name: str
    predicted_transfer_class: TransferClass
    predicted_delta_z: float
    predicted_delta_p: float
    predicted_rescue: float
    predictive_interval: Tuple[float, float]
    abstention_probability: float
    timestamp_utc: str
    source_claim_hash: str
    predictive_model_hash: str
    model_hash: str
    tokenizer_hash: str
    task_hash: str
    analysis_plan_hash: str
    preregistration_hash: str

    def compute_seal_hash(self) -> str:
        payload = {
            "prediction_id": self.prediction_id,
            "source_model": self.source_model,
            "target_model": self.target_model,
            "task_name": self.task_name,
            "class": self.predicted_transfer_class.value,
            "predicted_delta_z": round(self.predicted_delta_z, 4),
            "predicted_delta_p": round(self.predicted_delta_p, 4),
            "predicted_rescue": round(self.predicted_rescue, 4),
            "interval": [round(x, 4) for x in self.predictive_interval],
            "abstention_prob": round(self.abstention_probability, 4),
            "prereg_hash": self.preregistration_hash,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["predicted_transfer_class"] = self.predicted_transfer_class.value
        d["seal_hash"] = self.compute_seal_hash()
        return d


@dataclass
class HeldOutChallengeSuite:
    suite_id: str
    predictions: List[ProspectivePrediction]
    sealed_manifest_hash: str
    timestamp_utc: str

    def export_blind_bundle(self) -> Dict[str, Any]:
        """Exports challenge bundle for external researchers containing zero empirical targets."""
        items = []
        for p in self.predictions:
            items.append({
                "prediction_id": p.prediction_id,
                "source_model": p.source_model,
                "target_model": p.target_model,
                "task_name": p.task_name,
                "model_hash": p.model_hash,
                "task_hash": p.task_hash,
                "probes": [f"PROBE_{p.task_name.upper()}_X", f"PROBE_{p.task_name.upper()}_Y"],
            })
        return {
            "suite_id": self.suite_id,
            "sealed_manifest_hash": self.sealed_manifest_hash,
            "challenges": items,
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class HeldOutObservation:
    prediction_id: str
    investigator_id: str
    observed_transfer_class: TransferClass
    empirical_delta_z: float
    empirical_delta_p: float
    empirical_rescue: float
    raw_tensors: Dict[str, List[float]]
    hardware_metadata: Dict[str, Any]
    investigator_signature: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["observed_transfer_class"] = self.observed_transfer_class.value
        return d
