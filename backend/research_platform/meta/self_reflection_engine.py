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
        # With no hypotheses supplied there is nothing to reflect on. This used
        # to invent two successes (including "SAE Feature #1402 Capital
        # Encoding" at impact 0.92) and two failures attributed to "Low causal
        # effect", then credited good decisions such as having "utilized
        # Sparse Autoencoder feature genealogy" -- work that cannot have
        # happened, because SAE encoding is not implemented here.
        supplied = bool(successful_hypotheses or failed_hypotheses)
        succ: List[str] = list(successful_hypotheses or [])
        fail: List[str] = list(failed_hypotheses or [])

        compute_waste = round(compute_used_gb_hours * (len(fail) / (len(succ) + len(fail) or 1)), 2)
        score = round(max(0.1, 1.0 - (compute_waste / (compute_used_gb_hours or 1.0))), 2)

        bad_decisions = []
        if compute_waste > 4.0:
            bad_decisions.append("Compute was spent on hypotheses that did not succeed.")
        if len(fail) > len(succ):
            bad_decisions.append("Initiated deep multi-gpu execution before validating sample density.")

        # Only credit decisions that were actually recorded.
        good_decisions = [str(d.get("rationale")) for d in (planner_decisions or [])
                          if isinstance(d, dict) and d.get("rationale")]

        report = ReflectionReport(
            campaign_id=campaign_id,
            timestamp=_dt.datetime.utcnow().isoformat() + "Z",
            # Per-hypothesis impact is not measurable here, so it is not claimed.
            successes=[{"hypothesis": h, "impact": None} for h in succ],
            failures=[{"hypothesis": h, "reason": None} for h in fail],
            compute_waste=compute_waste,
            false_hypotheses=fail,
            good_decisions=good_decisions,
            bad_decisions=bad_decisions,
            future_recommendations=[],
            confidence=0.0,
            overall_reflection_score=score,
        )

        payload = asdict(report)
        payload["hypotheses_supplied"] = supplied
        payload["provenance"] = "live" if supplied else "unavailable"
        payload["validation_eligible"] = False
        payload["publication_eligible"] = False
        payload["reason"] = (
            None if supplied else
            "No successful or failed hypotheses were supplied, so this report "
            "reflects on nothing. It previously invented hypotheses, assigned "
            "them impact scores, attributed failures to 'Low causal effect', "
            "and credited a Sparse Autoencoder workflow that is not "
            "implemented."
        )

        self.reflections_history.append(report)
        return payload

    def list_reflections(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.reflections_history]
