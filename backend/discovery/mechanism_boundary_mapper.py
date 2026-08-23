r"""Mechanism Boundary Mapper & Scientific Revision Agenda Generator for MECH.

Transforms adversarial failures and abstentions into structured boundary objects:
    Adversarial Challenge -> Boundary Detection -> Structured Causal Agenda
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .adversarial_replication_oracle import AdversarialChallengeCase
from .failure_budget_controller import FailureEvaluationRecord, FailureRegime


@dataclass
class ClaimBoundaryRecord:
    boundary_id: str
    case_id: str
    architecture_family: str
    model_scale: str
    task_family: str
    mechanism_type: str
    transfer_distance: float
    predicted_uncertainty: float
    observed_error: float
    causal_failure_mode: str
    recommended_calibration_experiment: str
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "case_id": self.case_id,
            "architecture_family": self.architecture_family,
            "model_scale": self.model_scale,
            "task_family": self.task_family,
            "mechanism_type": self.mechanism_type,
            "transfer_distance": round(self.transfer_distance, 3),
            "predicted_uncertainty": round(self.predicted_uncertainty, 4),
            "observed_error": round(self.observed_error, 4),
            "causal_failure_mode": self.causal_failure_mode,
            "recommended_calibration_experiment": self.recommended_calibration_experiment,
            "timestamp_utc": self.timestamp_utc,
        }


class MechanismBoundaryMapper:
    """Extracts and registers structured scientific boundary definitions."""

    def map_boundary_case(
        self,
        case: AdversarialChallengeCase,
        eval_record: FailureEvaluationRecord,
    ) -> ClaimBoundaryRecord:
        """Constructs an actionable scientific boundary and revision agenda."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        boundary_id = f"BOUNDARY_{case.case_id}_{hashlib.sha256(ts.encode('utf-8')).hexdigest()[:6]}"

        if case.is_out_of_domain:
            rec_exp = f"Design cross-paradigm calibration bridge between Transformer self-attention and {case.target_model} recurrent state kernel."
        elif eval_record.regime == FailureRegime.CALIBRATION_FAILURE:
            rec_exp = f"Isolate polysemantic superposition subspaces via Sparse Autoencoder dictionary expansion for {case.task_name}."
        else:
            rec_exp = f"Extend capacity-scaling regression terms for asymmetric {case.source_model} -> {case.target_model} transfer."

        return ClaimBoundaryRecord(
            boundary_id=boundary_id,
            case_id=case.case_id,
            architecture_family="Transformer-to-SSM" if case.is_out_of_domain else "Cross-Transformer",
            model_scale="Dense-to-MoE" if "Mixtral" in case.target_model else "Dense-to-Dense",
            task_family=case.task_name,
            mechanism_type=case.stress_dimension.value,
            transfer_distance=case.adversarial_difficulty_d_adv,
            predicted_uncertainty=eval_record.abstention_probability,
            observed_error=eval_record.prediction_error,
            causal_failure_mode=case.expected_failure_mode,
            recommended_calibration_experiment=rec_exp,
            timestamp_utc=ts,
        )
