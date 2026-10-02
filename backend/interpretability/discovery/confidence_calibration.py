"""Confidence Calibration Engine.

NOT IMPLEMENTED as a calibration. A calibration requires comparing predicted
confidences against observed frequencies (e.g. a reliability diagram); this
only applies a shrinkage heuristic that pulls every score toward zero by
``1/sqrt(n)`` and then declared ``is_calibrated: True`` unconditionally.

Claiming calibration means nothing more than that a score means what it says.
That claim is removed; the shrinkage is kept and relabelled for what it is.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ConfidenceCalibrationEngine:
    """Applies a shrinkage heuristic. This is not a calibration."""

    def calibrate_confidence(
        self,
        raw_confidence: Optional[float] = None,
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        assumed = raw_confidence is None or sample_size is None
        raw = 0.95 if raw_confidence is None else raw_confidence
        n = 500 if sample_size is None else sample_size

        if n <= 0:
            return {
                "status": "error",
                "provenance": "unavailable",
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": "sample_size must be positive.",
            }

        calibrated = round(min(0.99, raw * (1.0 - (1.0 / (n ** 0.5)))), 4)
        return {
            "raw_confidence": raw,
            "calibrated_confidence": calibrated,
            "sample_size": n,
            "shrinkage_delta": round(abs(raw - calibrated), 4),
            "method": "sqrt_shrinkage",
            "is_calibrated": False,
            "calibration_performed": False,
            "status": "unavailable",
            "provenance": "unavailable" if assumed else "reference",
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Calibration is not implemented. No predicted-vs-observed "
                "comparison was performed, so a confidence score is not known "
                "to mean what it claims; the returned value is a sqrt "
                "shrinkage of the input."
            ),
        }
