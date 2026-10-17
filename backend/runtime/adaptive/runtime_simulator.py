"""Epic 1 — Runtime Simulator Engine with Dynamic Calibration Auto-Correction.

Pre-executes simulated workloads to predict hardware memory, execution time, contention, and failure risk,
applying dynamic calibration scaling factors derived from observed telemetry diffs.

The output is an ESTIMATE. The arithmetic is real, but its inputs are assumed
constants -- 8.4 GB base VRAM, 0.048 s per prompt, plus the calibration factor
-- so nothing here measures the target hardware. The "uncertainty intervals"
are a fixed +/-8% band around the point estimate, not a distribution over
samples. `confidence_pct` previously returned the literal 94.0 and
`simulation_passed` the literal True for every input.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from .calibration_engine import PredictionCalibrationEngine


class RuntimeSimulatorEngine:
    """Simulates job execution prior to submission with statistical uncertainty intervals and dynamic calibration factors."""

    def __init__(self) -> None:
        self.calibration_engine = PredictionCalibrationEngine()

    def simulate_execution(
        self,
        model_name: str = "GPT-2 Small",
        prompts_count: int = 1000,
        target_backend: str = "Ray",
        precision: str = "FP16",
    ) -> Dict[str, Any]:
        cal_factor = self.calibration_engine.get_calibration_factor(model_name, target_backend)

        precision_multiplier = 0.5 if precision == "INT8" else (1.0 if precision == "FP16" else 2.0)
        base_vram_gb = 8.4 * precision_multiplier
        vram_margin = round(base_vram_gb * 0.08, 2)
        vram_interval = [round(base_vram_gb - vram_margin, 2), round(base_vram_gb + vram_margin, 2)]

        raw_runtime_sec = prompts_count * 0.048 * precision_multiplier
        calibrated_runtime_sec = round(raw_runtime_sec * cal_factor, 2)
        runtime_margin_sec = round(calibrated_runtime_sec * 0.08, 2)
        runtime_interval = [round(calibrated_runtime_sec - runtime_margin_sec, 2), round(calibrated_runtime_sec + runtime_margin_sec, 2)]

        contention_factor = round(min(0.95, 0.10 + 0.15 * math.log1p(prompts_count / 100)), 2)
        failure_prob = round(max(0.001, 0.005 + (base_vram_gb / 80.0) * 0.02), 4)

        # A plan "fits" on the same comparison that already selects the safety
        # margin, so the verdict and the advice cannot disagree.
        fits_vram = base_vram_gb < 16.0

        return {
            "model_name": model_name,
            "prompts_count": prompts_count,
            "target_backend": target_backend,
            "precision": precision,
            "simulated_vram_gb": round(base_vram_gb, 2),
            "vram_margin_gb": vram_margin,
            "vram_interval": vram_interval,
            "simulated_runtime_sec": calibrated_runtime_sec,
            "calibration_factor_applied": cal_factor,
            "runtime_margin_sec": runtime_margin_sec,
            "runtime_interval": runtime_interval,
            # No confidence is estimated. The model is analytic arithmetic over
            # assumed constants and nothing here measures the target hardware,
            # so there is no basis for a percentage. This was the literal 94.0.
            "confidence_pct": None,
            "simulated_contention_factor": contention_factor,
            "simulated_failure_probability": failure_prob,
            "recommended_safety_margin": ("Sufficient VRAM" if fits_vram
                                          else "Consider INT8 Quantization"),
            # Derived from the same VRAM comparison that chooses the safety
            # margin, so the verdict and the advice cannot disagree. The
            # previous literal `True` meant the field could not report a
            # failure for any input.
            "simulation_passed": fits_vram,
            "simulation_is_estimate": True,
            "provenance": "reference",
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Simulated estimate. Values come from a fixed analytic model "
                "over assumed constants (8.4 GB base VRAM, 0.048 s per prompt, "
                "and a per-model calibration factor), not from a measurement of "
                "the target hardware or of a real run."
            ),
        }
