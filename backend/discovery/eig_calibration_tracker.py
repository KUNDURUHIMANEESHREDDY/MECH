"""EIG Calibration & Experiment Selection Regret Tracker for MECH.

Tracks empirical calibration error:
| EIG_hat - Observed_Information_Gain |

And selection regret:
Regret = OIG(E_oracle) - OIG(E_chosen)
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EIGCalibrationRecord:
    """Audit record evaluating predicted vs observed information gain for an experiment."""
    experiment_id: str
    experiment_category: str
    predicted_eig_bits: float
    observed_oig_bits: float
    calibration_error_bits: float
    oracle_max_oig_bits: float
    selection_regret_bits: float
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationTelemetrySummary:
    """Summary of EIG calibration and regret metrics across all executed experiments."""
    total_experiments_evaluated: int
    mean_calibration_error_bits: float
    mean_selection_regret_bits: float
    max_selection_regret_bits: float
    calibration_efficiency_pct: float
    records: List[EIGCalibrationRecord]
    last_updated_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_experiments_evaluated": self.total_experiments_evaluated,
            "mean_calibration_error_bits": round(self.mean_calibration_error_bits, 4),
            "mean_selection_regret_bits": round(self.mean_selection_regret_bits, 4),
            "max_selection_regret_bits": round(self.max_selection_regret_bits, 4),
            "calibration_efficiency_pct": round(self.calibration_efficiency_pct, 2),
            "records": [r.to_dict() for r in self.records],
            "last_updated_utc": self.last_updated_utc,
        }


class EIGCalibrationTracker:
    """Tracks calibration accuracy and decision regret for autonomous experiment selection."""

    def __init__(self) -> None:
        self.history: List[EIGCalibrationRecord] = []

    def record_experiment_outcome(
        self,
        experiment_id: str,
        experiment_category: str,
        predicted_eig: float,
        prior_entropy_bits: float,
        posterior_entropy_bits: float,
        oracle_max_oig: Optional[float] = None,
    ) -> EIGCalibrationRecord:
        """Calculates observed information gain, calibration error, and selection regret."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        oig = max(0.0, prior_entropy_bits - posterior_entropy_bits)
        cal_error = abs(predicted_eig - oig)

        max_oig = oracle_max_oig if oracle_max_oig is not None else max(predicted_eig, oig)
        regret = max(0.0, max_oig - oig)

        rec = EIGCalibrationRecord(
            experiment_id=experiment_id,
            experiment_category=experiment_category,
            predicted_eig_bits=round(predicted_eig, 4),
            observed_oig_bits=round(oig, 4),
            calibration_error_bits=round(cal_error, 4),
            oracle_max_oig_bits=round(max_oig, 4),
            selection_regret_bits=round(regret, 4),
            timestamp_utc=ts,
        )
        self.history.append(rec)
        return rec

    def get_telemetry_summary(self) -> CalibrationTelemetrySummary:
        """Computes aggregate calibration error, selection regret, and efficiency percentage."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        n = len(self.history)
        if n == 0:
            return CalibrationTelemetrySummary(
                total_experiments_evaluated=0,
                mean_calibration_error_bits=0.0,
                mean_selection_regret_bits=0.0,
                max_selection_regret_bits=0.0,
                calibration_efficiency_pct=100.0,
                records=[],
                last_updated_utc=ts,
            )

        mean_err = sum(r.calibration_error_bits for r in self.history) / n
        mean_reg = sum(r.selection_regret_bits for r in self.history) / n
        max_reg = max(r.selection_regret_bits for r in self.history)

        total_predicted = sum(r.predicted_eig_bits for r in self.history)
        total_observed = sum(r.observed_oig_bits for r in self.history)
        efficiency = min(100.0, max(0.0, (1.0 - (mean_err / max(0.1, (total_predicted / n)))) * 100.0))

        return CalibrationTelemetrySummary(
            total_experiments_evaluated=n,
            mean_calibration_error_bits=mean_err,
            mean_selection_regret_bits=mean_reg,
            max_selection_regret_bits=max_reg,
            calibration_efficiency_pct=efficiency,
            records=self.history,
            last_updated_utc=ts,
        )
