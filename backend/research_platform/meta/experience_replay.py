"""Epic 8 — Research Experience Replay Engine with "Why" Decision Provenance.

Stores complete research trajectories and replays historical campaigns to identify decision divergence points and optimize future policies.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class ExperienceStep:
    """Single step in a recorded research trajectory with full "Why" decision provenance."""

    step_index: int
    action_type: str
    decision_reasoning: str
    why: List[str]
    confidence: float
    alternatives: List[str]
    compute_cost_sec: float
    confidence_delta: float
    outcome: str


class ResearchExperienceReplay:
    """Replays historical research trajectories to analyze decision divergence points and 'Why' provenance."""

    def __init__(self) -> None:
        self.trajectories: Dict[str, List[ExperienceStep]] = {
            "camp_s6_ioi": [
                ExperienceStep(
                    step_index=1,
                    action_type="Run SAE Inspection",
                    decision_reasoning="Extract candidate sparse feature directions",
                    why=["SAE L8 checkpoint available", "Polysemanticity score low", "Target token active"],
                    confidence=0.94,
                    alternatives=["Dense Activation Search", "Random Head Sweep"],
                    compute_cost_sec=12.0,
                    confidence_delta=0.25,
                    outcome="Success",
                ),
                ExperienceStep(
                    step_index=2,
                    action_type="Run Causal Tracing",
                    decision_reasoning="Verify layer 8 activation mediation",
                    why=["IOI benchmark score high (0.94)", "SAE feature 1402 active", "Patch confidence 0.91"],
                    confidence=0.91,
                    alternatives=["Attribution Patching", "Activation Search"],
                    compute_cost_sec=45.0,
                    confidence_delta=0.40,
                    outcome="Success",
                ),
                ExperienceStep(
                    step_index=3,
                    action_type="Path Patching",
                    decision_reasoning="Localize precise attention head edges",
                    why=["Direct logit boost confirmed", "Name mover head active"],
                    confidence=0.88,
                    alternatives=["Full Matrix Fine-Tuning"],
                    compute_cost_sec=30.0,
                    confidence_delta=0.20,
                    outcome="Success",
                ),
                ExperienceStep(
                    step_index=4,
                    action_type="Random Attention Sweep",
                    decision_reasoning="Unfocused sweep over non-candidate heads",
                    why=["Exploratory check"],
                    confidence=0.45,
                    alternatives=["Cross-Model Circuit Comparison"],
                    compute_cost_sec=120.0,
                    confidence_delta=-0.05,
                    outcome="Wasted Compute",
                ),
            ]
        }

    def record_trajectory_step(
        self,
        campaign_id: str,
        action_type: str,
        decision_reasoning: str,
        why: List[str] | None = None,
        confidence: float = 0.90,
        alternatives: List[str] | None = None,
        compute_cost_sec: float = 15.0,
        confidence_delta: float = 0.15,
        outcome: str = "Success",
    ) -> Dict[str, Any]:
        if campaign_id not in self.trajectories:
            self.trajectories[campaign_id] = []

        step_index = len(self.trajectories[campaign_id]) + 1
        step = ExperienceStep(
            step_index=step_index,
            action_type=action_type,
            decision_reasoning=decision_reasoning,
            why=why or ["Default reasoning step"],
            confidence=confidence,
            alternatives=alternatives or ["None"],
            compute_cost_sec=compute_cost_sec,
            confidence_delta=confidence_delta,
            outcome=outcome,
        )
        self.trajectories[campaign_id].append(step)
        return asdict(step)

    def replay_campaign(self, campaign_id: str = "camp_s6_ioi") -> Dict[str, Any]:
        trajectory = self.trajectories.get(campaign_id, [])
        if not trajectory:
            return {"status": "TrajectoryNotFound", "campaign_id": campaign_id}

        divergence_points = []
        wasted_compute_sec = 0.0

        for step in trajectory:
            if step.outcome.lower() == "wasted compute" or step.confidence_delta < 0:
                divergence_points.append({
                    "step_index": step.step_index,
                    "suboptimal_action": step.action_type,
                    "reason": step.decision_reasoning,
                    "why_provenance": step.why,
                    "rejected_alternatives": step.alternatives,
                    "recommended_alternative": "Skip random sweep; proceed directly to cross-model circuit comparison.",
                })
                wasted_compute_sec += step.compute_cost_sec

        return {
            "status": "ReplayCompleted",
            "campaign_id": campaign_id,
            "total_steps": len(trajectory),
            "trajectory": [asdict(s) for s in trajectory],
            "divergence_points": divergence_points,
            "wasted_compute_sec": wasted_compute_sec,
            "policy_insight": "Experience replay recommends placing SAE inspection before causal interventions to reduce compute waste.",
        }
