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
        confidence_score: float = 0.92,
        uncertainty_interval: List[float] | None = None,
        sample_size: int = 5,
        variance: float = 0.02,
        custom_policy: UncertaintyPolicy | Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        pol = custom_policy if isinstance(custom_policy, UncertaintyPolicy) else (UncertaintyPolicy.from_dict(custom_policy) if custom_policy else self.policy)
        interval = uncertainty_interval or [0.88, 0.95]
        interval_width = round(interval[1] - interval[0], 4)

        if confidence_score < pol.rejection_confidence:
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
