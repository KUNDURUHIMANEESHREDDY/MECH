import numpy as np
from typing import Dict, Any, Tuple

class CalibrationMetrics:
    """
    Computes calibration metrics: ECE, MCE, Brier Score.
    """

    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0., 1., n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        ece = 0.0
        for i in range(n_bins):
            mask = binids == i
            if np.sum(mask) > 0:
                prob_pred = np.mean(y_prob[mask])
                prob_true = np.mean(y_true[mask])
                ece += (np.sum(mask) / len(y_prob)) * np.abs(prob_pred - prob_true)
        return float(ece)

    @staticmethod
    def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0., 1., n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        mce = 0.0
        for i in range(n_bins):
            mask = binids == i
            if np.sum(mask) > 0:
                prob_pred = np.mean(y_prob[mask])
                prob_true = np.mean(y_true[mask])
                mce = max(mce, np.abs(prob_pred - prob_true))
        return float(mce)

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return float(np.mean((y_prob - y_true)**2))

    @classmethod
    def compute_all(cls, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
        return {
            "expected_calibration_error": cls.expected_calibration_error(y_true, y_prob, n_bins),
            "maximum_calibration_error": cls.maximum_calibration_error(y_true, y_prob, n_bins),
            "brier_score": cls.brier_score(y_true, y_prob)
        }
