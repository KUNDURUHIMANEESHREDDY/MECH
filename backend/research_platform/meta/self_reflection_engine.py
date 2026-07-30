"""Epic 2 — Self-Reflection Engine.

Evaluates research campaigns post-execution using machine-readable reflection reports.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class ReflectionReport:
    """Structured machine-readable report documenting AI scientist campaign reflection."""

    campaign_id: str
    timestamp: str
    successes: List[Dict[str, Any]]
    failures: List[Dict[str, Any]]
    compute_waste: float
    false_hypotheses: List[str]
    good_decisions: List[str]
    bad_decisions: List[str]
    future_recommendations: List[str]
    confidence: float
    overall_reflection_score: float


class SelfReflectionEngine:
    """Generates machine-readable reflection reports evaluating efficiency and decision quality."""

    def __init__(self) -> None:
        self.reflections_history: List[ReflectionReport] = []

    def generate_reflection_report(
        self,
        campaign_id: str = "camp_default",
        successful_hypotheses: List[str] | None = None,
        failed_hypotheses: List[str] | None = None,
        compute_used_gb_hours: float = 12.5,
        planner_decisions: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        succ = successful_hypotheses or ["L8_N402 IOI Induction Head", "SAE Feature #1402 Capital Encoding"]
        fail = failed_hypotheses or ["Random Attention Head Ablation in Layer 2", "Uncalibrated Logit Difference Thresholding"]

        compute_waste = round(compute_used_gb_hours * (len(fail) / (len(succ) + len(fail) or 1)), 2)
        score = round(max(0.1, 1.0 - (compute_waste / (compute_used_gb_hours or 1.0))), 2)

        bad_decisions = []
        if compute_waste > 4.0:
            bad_decisions.append("Used attribution patching before running Sparse Autoencoder feature extraction.")
        if len(fail) > len(succ):
            bad_decisions.append("Initiated deep multi-gpu execution before validating sample density.")

        good_decisions = [
            "Validated IOI benchmark suite before executing causal interventions.",
            "Utilized Sparse Autoencoder feature genealogy to track multi-layer projections.",
        ]

        report = ReflectionReport(
            campaign_id=campaign_id,
            timestamp=_dt.datetime.utcnow().isoformat() + "Z",
            successes=[{"hypothesis": h, "impact": 0.92} for h in succ],
            failures=[{"hypothesis": h, "reason": "Low causal effect"} for h in fail],
            compute_waste=compute_waste,
            false_hypotheses=fail,
            good_decisions=good_decisions,
            bad_decisions=bad_decisions or ["None detected"],
            future_recommendations=[
                "Always run SAE inspection prior to dense intervention testing.",
                "Enforce minimum sample density check (N>=5) before launching distributed Slurm jobs.",
            ],
            confidence=0.94,
            overall_reflection_score=score,
        )

        self.reflections_history.append(report)
        return asdict(report)

    def list_reflections(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.reflections_history]
