"""Epic 8 — Research Experience Replay Engine with "Why" Decision Provenance.

Stores complete research trajectories and replays historical campaigns to identify decision divergence points and optimize future policies.

The replay *logic* is real: given recorded steps it finds divergence points and
totals wasted compute. The seeded `camp_s6_ioi` trajectory is not real. It
previously asserted, as remembered history, that an "SAE L8 checkpoint" was
available (no SAE weights exist in this repository), that "IOI benchmark score
high (0.94)" held (that number was a stub), that "SAE feature 1402" was active,
and that a "direct logit boost" was confirmed -- then drew a policy
recommendation from all of it. A replay engine is exactly the kind of component
that feeds an agent confident-looking conclusions, so the seed is labelled
reference data rather than history.
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
    #: Whether this step came from a real recorded run. The seeded trajectory
    #: is illustrative, not history.
    recorded: bool = True
    provenance: str = "live"
    validation_eligible: bool = False
    publication_eligible: bool = False


class ResearchExperienceReplay:
    """Replays historical research trajectories to analyze decision divergence points and 'Why' provenance."""

    #: The seed is an illustration of the trajectory shape, not a record of
    #: experiments. Several of its original "why" entries asserted
    #: measurements this repository cannot make, so they are stated as
    #: reference notes instead.
    SEED_NOTICE = (
        "Illustrative trajectory. These steps were not recorded from real "
        "runs: no SAE encoder is loaded, the IOI benchmark figure was a stub, "
        "and no logit boost was measured. Use this shape as an example only."
    )

    def __init__(self) -> None:
        ref = "reference"
        self.trajectories: Dict[str, List[ExperienceStep]] = {
            "camp_s6_ioi": [
                ExperienceStep(
                    step_index=1,
                    action_type="Run SAE Inspection",
                    decision_reasoning="Extract candidate sparse feature directions",
                    why=["Reference note: no SAE encoder weights are loaded, "
                         "so no candidate directions exist yet."],
                    confidence=0.94,
                    alternatives=["Dense Activation Search", "Random Head Sweep"],
                    compute_cost_sec=12.0,
                    confidence_delta=0.25,
                    outcome="Illustrative",
                    recorded=False,
                    provenance=ref,
                ),
                ExperienceStep(
                    step_index=2,
                    action_type="Run Causal Tracing",
                    decision_reasoning="Verify layer 8 activation mediation",
                    why=["Reference note: the 0.94 IOI figure this step used "
                         "to cite was a stub, not a measurement."],
                    confidence=0.91,
                    alternatives=["Attribution Patching", "Activation Search"],
                    compute_cost_sec=45.0,
                    confidence_delta=0.40,
                    outcome="Illustrative",
                    recorded=False,
                    provenance=ref,
                ),
                ExperienceStep(
                    step_index=3,
                    action_type="Path Patching",
                    decision_reasoning="Localize precise attention head edges",
                    why=["Reference note: no logit boost was confirmed; real "
                         "path patching is not implemented here."],
                    confidence=0.88,
                    alternatives=["Full Matrix Fine-Tuning"],
                    compute_cost_sec=30.0,
                    confidence_delta=0.20,
                    outcome="Illustrative",
                    recorded=False,
                    provenance=ref,
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
                    recorded=False,
                    provenance=ref,
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

        recorded = any(s.recorded for s in trajectory)
        return {
            "status": "ReplayCompleted",
            "campaign_id": campaign_id,
            "total_steps": len(trajectory),
            "trajectory": [asdict(s) for s in trajectory],
            "divergence_points": divergence_points,
            "wasted_compute_sec": wasted_compute_sec,
            # A policy recommendation derived from an illustrative trajectory is
            # a fabricated conclusion. The divergence arithmetic is real; the
            # advice about what to do about it is not evidence.
            "policy_insight": (
                "Experience replay recommends placing SAE inspection before "
                "causal interventions to reduce compute waste."
                if recorded else None
            ),
            "recorded": recorded,
            "provenance": "live" if recorded else "reference",
            "validation_eligible": bool(recorded),
            "publication_eligible": False,
            "reason": None if recorded else self.SEED_NOTICE,
        }

