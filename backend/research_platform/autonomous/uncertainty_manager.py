"""Uncertainty Manager Engine with Configurable Policy & Closed-Loop Planner Integration."""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class UncertaintyPolicy:
    """Configurable and versioned threshold policy for research decisions."""

    policy_id: str = "pol_v1.0"
    version: str = "1.0.0"
    name: str = "Publication Mode Standard"
    created_at: str = "2026-07-26T00:00:00Z"
    created_by: str = "AI Scientist Architect"
    publication_confidence: float = 0.85
    max_interval_width: float = 0.15
    debate_variance: float = 0.05
    rejection_confidence: float = 0.60
    min_samples: int = 2

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> UncertaintyPolicy:
        if not data:
            return cls()
        return cls(
            policy_id=data.get("policy_id", "pol_v1.0"),
            version=data.get("version", "1.0.0"),
            name=data.get("name", "Publication Mode Standard"),
            created_at=data.get("created_at", "2026-07-26T00:00:00Z"),
            created_by=data.get("created_by", "AI Scientist Architect"),
            publication_confidence=data.get("publication_confidence", 0.85),
            max_interval_width=data.get("max_interval_width", 0.15),
            debate_variance=data.get("debate_variance", 0.05),
            rejection_confidence=data.get("rejection_confidence", 0.60),
            min_samples=data.get("min_samples", 2),
        )


class UncertaintyManagerEngine:
    """Evaluates evidence uncertainty bounds using versioned policies and drives planning feedback loops."""

    def __init__(self, policy: UncertaintyPolicy | Dict[str, Any] | None = None) -> None:
        self.policy = policy if isinstance(policy, UncertaintyPolicy) else UncertaintyPolicy.from_dict(policy)

    def set_policy(self, policy: UncertaintyPolicy | Dict[str, Any]) -> None:
        self.policy = policy if isinstance(policy, UncertaintyPolicy) else UncertaintyPolicy.from_dict(policy)

    def evaluate_uncertainty(
        self,
        confidence_score: float | None = None,
        uncertainty_interval: List[float] | None = None,
        sample_size: int | None = None,
        variance: float | None = None,
        custom_policy: UncertaintyPolicy | Dict[str, Any] | None = None,
        evidence_missing: bool = False,
    ) -> Dict[str, Any]:
        """Apply the threshold policy to a measurement.

        Every input defaults to None, meaning *not supplied*. They previously
        defaulted to plausible values -- `confidence_score=0.92`,
        `uncertainty_interval=[0.88, 0.95]`, `sample_size=5`, `variance=0.02` --
        and with all four defaults the branch chain reached:

            0.92 >= publication_confidence (0.85)
            interval width 0.07 <= max_interval_width (0.15)
            -> "Enough evidence" / "Publish"

        So calling this method with no arguments at all returned a publish
        decision on four invented numbers. That is the exact outcome the
        `evidence_missing` branch exists to prevent, reachable simply by omitting
        arguments -- and the four defaults were individually too small to look
        wrong in review while composing into a confident publication.

        Omitting any input now routes to the same missing-evidence branch, so
        absence cannot be read as confidence. Note the policy thresholds
        themselves (`publication_confidence=0.85`, `rejection_confidence=0.60`)
        are deliberately left alone: those are decision rules in a class
        documented as "threshold policy", not claims about the world.

        Callers must pass what they measured. The production caller
        (`ai_scientist_engine`) already passes all four explicitly.
        """
        pol = custom_policy if isinstance(custom_policy, UncertaintyPolicy) else (UncertaintyPolicy.from_dict(custom_policy) if custom_policy else self.policy)
        interval = list(uncertainty_interval) if uncertainty_interval else None
        interval_width = (
            round(interval[1] - interval[0], 4) if interval and len(interval) >= 2 else None
        )
        missing = (
            evidence_missing
            or confidence_score is None
            or interval_width is None
            or sample_size is None
            or variance is None
        )

        if missing:
            # Validation returned nothing usable. Absence of evidence is not
            # evidence of absence, so neither reject nor publish: the correct
            # response is to go and gather evidence.
            decision = "Validation unavailable"
            action = "More experiments"
            priority = 5
            info_gain = 0.85
            est_compute = 1.50
            est_duration = 600
            planner_request = {
                "target": "planner",
                "reason": "Validation returned no usable confidence evidence",
                "recommended_additional_samples": max(1, pol.min_samples - (sample_size or 0)),
                "priority": priority,
                "expected_information_gain": info_gain,
                "estimated_compute_gpu_hours": est_compute,
            }
        # Everything below this point is only reachable when `missing` is False,
        # which guarantees confidence_score, interval_width, sample_size and
        # variance are all supplied. The comparisons are therefore safe even
        # though the annotations admit None.
        elif confidence_score < pol.rejection_confidence:
            decision = "Reject hypothesis"
            action = "Reject"
            priority = 1
            info_gain = 0.10
            est_compute = 0.10
            est_duration = 30
            planner_request = None
        elif sample_size < pol.min_samples:
            decision = "Weak evidence"
            action = "More experiments"
            priority = 5
            info_gain = 0.85
            est_compute = 1.50
            est_duration = 600
            planner_request = {
                "target": "planner",
                "reason": "Insufficient samples",
                "recommended_additional_samples": pol.min_samples - sample_size + 2,
                "priority": priority,
                "expected_information_gain": info_gain,
                "estimated_compute_gpu_hours": est_compute,
            }
        elif variance > pol.debate_variance:
            decision = "Conflicting evidence"
            action = "Debate"
            priority = 4
            info_gain = 0.75
            est_compute = 0.50
            est_duration = 180
            planner_request = {
                "target": "debate_engine",
                "reason": "High variance among experimental runs",
                "variance": variance,
                "priority": priority,
                "expected_information_gain": info_gain,
            }
        elif confidence_score >= pol.publication_confidence and interval_width <= pol.max_interval_width:
            decision = "Enough evidence"
            action = "Publish"
            priority = 2
            info_gain = 0.95
            est_compute = 0.20
            est_duration = 60
            planner_request = None
        else:
            decision = "Moderate evidence"
            action = "More experiments"
            priority = 3
            info_gain = 0.60
            est_compute = 1.00
            est_duration = 450
            planner_request = {
                "target": "planner",
                "reason": "Moderate confidence requires higher sample density",
                "recommended_additional_samples": 3,
                "priority": priority,
                "expected_information_gain": info_gain,
                "estimated_compute_gpu_hours": est_compute,
            }

        return {
            "confidence_score": confidence_score,
            "uncertainty_interval": interval,
            "interval_width": interval_width,
            "evidence_missing": bool(missing),
            "decision": decision,
            "action": action,
            "priority": priority,
            "expected_information_gain": info_gain,
            "estimated_compute_gpu_hours": est_compute,
            "estimated_duration_sec": est_duration,
            "ready_for_publication": action == "Publish",
            "policy": asdict(pol),
            "planner_request": planner_request,
        }
