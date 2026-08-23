r"""Failure Budget Controller & Scientific Error Regime Classifier for MECH.

Phase 59 explicitly permits failure while strictly bounding false confidence:
1. ROBUST_GENERALIZATION: Prediction was accurate under adversarial conditions.
2. CORRECT_BOUNDARY_DETECTION: Abstention correctly declared when out-of-domain.
3. CALIBRATION_FAILURE: Model was incorrect but appropriately declared high uncertainty.
4. CRITICAL_FAILURE: Model was incorrect AND claimed high confidence (FCR_adv <= 2.0%).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class FailureRegime(str, Enum):
    ROBUST_GENERALIZATION = "ROBUST_GENERALIZATION"
    CORRECT_BOUNDARY_DETECTION = "CORRECT_BOUNDARY_DETECTION"
    CALIBRATION_FAILURE = "CALIBRATION_FAILURE"
    CRITICAL_FAILURE = "CRITICAL_FAILURE"


@dataclass
class FailureEvaluationRecord:
    case_id: str
    regime: FailureRegime
    predicted_rescue: float
    empirical_rescue: float
    prediction_error: float
    abstention_probability: float
    is_out_of_domain: bool
    is_scientifically_acceptable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "regime": self.regime.value,
            "predicted_rescue": round(self.predicted_rescue, 4),
            "empirical_rescue": round(self.empirical_rescue, 4),
            "prediction_error": round(self.prediction_error, 4),
            "abstention_probability": round(self.abstention_probability, 4),
            "is_out_of_domain": self.is_out_of_domain,
            "is_scientifically_acceptable": self.is_scientifically_acceptable,
        }


class FailureBudgetController:
    """Classifies trial results and bounds the critical false-confidence rate."""

    def __init__(self, max_fcr: float = 0.02) -> None:
        self.max_fcr = max_fcr

    def classify_outcome(
        self,
        case_id: str,
        predicted_rescue: float,
        empirical_rescue: float,
        abstention_probability: float,
        is_out_of_domain: bool,
    ) -> FailureEvaluationRecord:
        """Classifies a trial outcome into one of 4 epistemic regimes."""
        is_abstained = abstention_probability >= 0.50
        pred_error = abs(predicted_rescue - empirical_rescue)

        if is_out_of_domain:
            if is_abstained:
                regime = FailureRegime.CORRECT_BOUNDARY_DETECTION
                acceptable = True
            else:
                regime = FailureRegime.CRITICAL_FAILURE
                acceptable = False
        else:
            if is_abstained:
                # Conservative calibration
                regime = FailureRegime.CALIBRATION_FAILURE
                acceptable = True
            else:
                if pred_error <= 0.05:
                    regime = FailureRegime.ROBUST_GENERALIZATION
                    acceptable = True
                else:
                    if abstention_probability >= 0.30:
                        regime = FailureRegime.CALIBRATION_FAILURE
                        acceptable = True
                    else:
                        regime = FailureRegime.CRITICAL_FAILURE
                        acceptable = False

        return FailureEvaluationRecord(
            case_id=case_id,
            regime=regime,
            predicted_rescue=predicted_rescue,
            empirical_rescue=empirical_rescue,
            prediction_error=pred_error,
            abstention_probability=abstention_probability,
            is_out_of_domain=is_out_of_domain,
            is_scientifically_acceptable=acceptable,
        )

    def evaluate_failure_budget(self, evaluations: List[FailureEvaluationRecord]) -> Tuple[float, bool]:
        """Calculates FCR_adv and checks against threshold."""
        total = max(1, len(evaluations))
        critical_count = sum(1 for e in evaluations if e.regime == FailureRegime.CRITICAL_FAILURE)
        fcr = critical_count / total
        is_within_budget = fcr <= self.max_fcr
        return fcr, is_within_budget
