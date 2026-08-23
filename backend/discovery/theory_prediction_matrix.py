r"""Theory Continuous Prediction Matrix & Disagreement Calculator for MECH.

Computes the prediction matrix:
                 Experiment E1   E2   E3   E4
    Theory T1        4.2        3.1  0.8  1.2
    Theory T2        1.3        3.0  2.4  1.1
    Theory T3        4.0        0.7  2.1  1.3
    Theory T4        0.9        3.2  0.6  4.8

Measures inter-theory disagreement:
    Disagreement(E) = Var_T[ E(y | T) ]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ExperimentDisagreementProfile:
    experiment_id: str
    predictions_by_theory: Dict[str, float]
    mean_prediction: float
    inter_theory_variance: float
    max_divergence_sigma: float
    expected_falsification_power: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "predictions_by_theory": {k: round(v, 4) for k, v in self.predictions_by_theory.items()},
            "mean_prediction": round(self.mean_prediction, 4),
            "inter_theory_variance": round(self.inter_theory_variance, 4),
            "max_divergence_sigma": round(self.max_divergence_sigma, 4),
            "expected_falsification_power": round(self.expected_falsification_power, 4),
        }


class TheoryPredictionMatrixEngine:
    """Computes continuous prediction matrices and identifies high-disagreement discriminating experiments."""

    def compute_prediction_matrix(
        self,
        theory_predictions: Dict[str, Dict[str, float]],
    ) -> List[ExperimentDisagreementProfile]:
        """theory_predictions: Dict[theory_id, Dict[experiment_id, predicted_outcome]]."""
        # Invert to experiment -> {theory_id: pred}
        exp_map: Dict[str, Dict[str, float]] = {}
        for t_id, exp_preds in theory_predictions.items():
            for e_id, pred in exp_preds.items():
                if e_id not in exp_map:
                    exp_map[e_id] = {}
                exp_map[e_id][t_id] = pred

        profiles: List[ExperimentDisagreementProfile] = []
        for e_id, t_preds in exp_map.items():
            vals = list(t_preds.values())
            n = len(vals)
            if n < 2:
                continue

            mean_val = sum(vals) / n
            variance = sum((v - mean_val) ** 2 for v in vals) / n
            std_dev = math.sqrt(variance)

            # Max divergence in standard deviations
            max_diff = max(vals) - min(vals)
            sigma_div = (max_diff / (std_dev or 1e-5)) if std_dev > 0 else 0.0

            # Falsification power is proportional to variance and separation
            falsification_power = min(1.0, (variance * 4.0) + (sigma_div * 0.10))

            profiles.append(
                ExperimentDisagreementProfile(
                    experiment_id=e_id,
                    predictions_by_theory=t_preds,
                    mean_prediction=mean_val,
                    inter_theory_variance=variance,
                    max_divergence_sigma=sigma_div,
                    expected_falsification_power=falsification_power,
                )
            )

        return sorted(profiles, key=lambda x: x.inter_theory_variance, reverse=True)
