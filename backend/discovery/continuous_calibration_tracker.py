"""Continuous Calibration & Multi-Variate Regret Tracker for MECH.

Tracks:
1. Continuous Log-Likelihood Calibration Error: -ln P(y | H_true, E)
2. Mahalanobis Distance Prediction Error: sqrt( sum_k (y_k - mu_k)^2 / var_k )
3. Continuous Observed Information Gain (OIG) & Selection Regret
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .continuous_outcome_likelihood_engine import ContinuousMeasurementVector, ContinuousOutcomeLikelihoodEngine, GaussianOutcomeParam


@dataclass
class ContinuousCalibrationRecord:
    """Audit record evaluating predicted vs observed continuous multi-variate metrics."""
    experiment_id: str
    experiment_category: str
    observed_measurements: ContinuousMeasurementVector
    predicted_eig_bits: float
    observed_oig_bits: float
    calibration_error_bits: float
    log_likelihood_loss: float
    mahalanobis_distance_error: float
    selection_regret_bits: float
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_category": self.experiment_category,
            "observed_measurements": self.observed_measurements.to_dict(),
            "predicted_eig_bits": round(self.predicted_eig_bits, 4),
            "observed_oig_bits": round(self.observed_oig_bits, 4),
            "calibration_error_bits": round(self.calibration_error_bits, 4),
            "log_likelihood_loss": round(self.log_likelihood_loss, 4),
            "mahalanobis_distance_error": round(self.mahalanobis_distance_error, 4),
            "selection_regret_bits": round(self.selection_regret_bits, 4),
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class ContinuousCalibrationSummary:
    """Summary of continuous multi-metric calibration and regret across all executed experiments."""
    total_continuous_experiments_evaluated: int
    mean_calibration_error_bits: float
    mean_log_likelihood_loss: float
    mean_mahalanobis_error: float
    mean_selection_regret_bits: float
    continuous_calibration_efficiency_pct: float
    records: List[ContinuousCalibrationRecord]
    last_updated_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_continuous_experiments_evaluated": self.total_continuous_experiments_evaluated,
            "mean_calibration_error_bits": round(self.mean_calibration_error_bits, 4),
            "mean_log_likelihood_loss": round(self.mean_log_likelihood_loss, 4),
            "mean_mahalanobis_error": round(self.mean_mahalanobis_error, 4),
            "mean_selection_regret_bits": round(self.mean_selection_regret_bits, 4),
            "continuous_calibration_efficiency_pct": round(self.continuous_calibration_efficiency_pct, 2),
            "records": [r.to_dict() for r in self.records],
            "last_updated_utc": self.last_updated_utc,
        }


class ContinuousCalibrationTracker:
    """Tracks continuous multi-metric calibration accuracy, Mahalanobis distances, and selection regret."""

    def __init__(self, likelihood_engine: Optional[ContinuousOutcomeLikelihoodEngine] = None) -> None:
        self.likelihood_engine = likelihood_engine or ContinuousOutcomeLikelihoodEngine()
        self.history: List[ContinuousCalibrationRecord] = []

    def record_continuous_experiment_outcome(
        self,
        experiment_id: str,
        experiment_category: str,
        true_hypothesis_id: str,
        observed_vector: ContinuousMeasurementVector,
        predicted_eig_bits: float,
        prior_entropy_bits: float,
        posterior_entropy_bits: float,
        oracle_max_oig: Optional[float] = None,
    ) -> ContinuousCalibrationRecord:
        """Computes OIG, NLL, Mahalanobis distance, and selection regret for a continuous experiment."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        oig = max(0.0, prior_entropy_bits - posterior_entropy_bits)
        cal_err = abs(predicted_eig_bits - oig)

        max_oig = oracle_max_oig if oracle_max_oig is not None else max(predicted_eig_bits, oig)
        regret = max(0.0, max_oig - oig)

        # Compute continuous likelihood under true hypothesis
        density = self.likelihood_engine.compute_continuous_density(
            hypothesis_id=true_hypothesis_id,
            experiment_category=experiment_category,
            observed_vector=observed_vector,
        )
        nll = -math.log(max(1e-12, density))

        # Compute Mahalanobis distance error
        param = self.likelihood_engine.likelihood_matrix.get((true_hypothesis_id, experiment_category))
        if param:
            d_dz = (observed_vector.delta_z - param.mu_delta_z)**2 / max(1e-6, param.var_delta_z)
            d_res = (observed_vector.mediation_rescue_fraction - param.mu_rescue)**2 / max(1e-6, param.var_rescue)
            d_spec = (observed_vector.control_specificity_ratio - param.mu_specificity)**2 / max(1e-6, param.var_specificity)
            mahalanobis = math.sqrt(d_dz + d_res + d_spec)
        else:
            mahalanobis = 1.0

        rec = ContinuousCalibrationRecord(
            experiment_id=experiment_id,
            experiment_category=experiment_category,
            observed_measurements=observed_vector,
            predicted_eig_bits=round(predicted_eig_bits, 4),
            observed_oig_bits=round(oig, 4),
            calibration_error_bits=round(cal_err, 4),
            log_likelihood_loss=round(nll, 4),
            mahalanobis_distance_error=round(mahalanobis, 4),
            selection_regret_bits=round(regret, 4),
            timestamp_utc=ts,
        )
        self.history.append(rec)
        return rec

    def get_continuous_telemetry_summary(self) -> ContinuousCalibrationSummary:
        """Computes aggregate continuous calibration metrics."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        n = len(self.history)
        if n == 0:
            return ContinuousCalibrationSummary(
                total_continuous_experiments_evaluated=0,
                mean_calibration_error_bits=0.0,
                mean_log_likelihood_loss=0.0,
                mean_mahalanobis_error=0.0,
                mean_selection_regret_bits=0.0,
                continuous_calibration_efficiency_pct=100.0,
                records=[],
                last_updated_utc=ts,
            )

        mean_cal = sum(r.calibration_error_bits for r in self.history) / n
        mean_nll = sum(r.log_likelihood_loss for r in self.history) / n
        mean_mah = sum(r.mahalanobis_distance_error for r in self.history) / n
        mean_reg = sum(r.selection_regret_bits for r in self.history) / n

        efficiency = min(100.0, max(0.0, (1.0 - (mean_cal / max(0.1, sum(r.predicted_eig_bits for r in self.history) / n))) * 100.0))

        return ContinuousCalibrationSummary(
            total_continuous_experiments_evaluated=n,
            mean_calibration_error_bits=mean_cal,
            mean_log_likelihood_loss=mean_nll,
            mean_mahalanobis_error=mean_mah,
            mean_selection_regret_bits=mean_reg,
            continuous_calibration_efficiency_pct=efficiency,
            records=self.history,
            last_updated_utc=ts,
        )
