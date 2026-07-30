"""Automated Scientific Regression Suite."""

from __future__ import annotations

from typing import Any, Dict


class AutomatedRegressionSuite:
    """Detects performance or scientific regression across benchmark algorithm updates."""

    def evaluate_regression(self, current_scores: Dict[str, float], previous_scores: Dict[str, float] | None = None) -> Dict[str, Any]:
        baseline = previous_scores or {"IOI": 0.94, "InductionHeads": 0.90, "FactualRecall": 0.92}
        regressions = []

        for b_id, score in current_scores.items():
            base = baseline.get(b_id, 0.90)
            if score < base - 0.02:
                regressions.append({"benchmark_id": b_id, "current": score, "baseline": base, "delta": round(score - base, 4)})

        return {
            "regression_detected": len(regressions) > 0,
            "regressions": regressions,
            "status": "Passing" if len(regressions) == 0 else "RegressionDetected",
        }
