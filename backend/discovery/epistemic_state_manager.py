r"""Global Epistemic State Manager for MECH.

Unifies global autonomous decision routing:
    EXPLOIT_THEORY | CALIBRATE_UNCERTAINTY | DISCRIMINATE_RIVALS | SYNTHESIZE_PRIMITIVE | ABSTAIN_BOUNDARY | HALT_EXHAUSTION
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class EpistemicOperationalState(str, Enum):
    EXPLOIT_ESTABLISHED_THEORY = "EXPLOIT_ESTABLISHED_THEORY"
    CALIBRATE_UNCERTAINTY = "CALIBRATE_UNCERTAINTY"
    DISCRIMINATE_RIVAL_THEORIES = "DISCRIMINATE_RIVAL_THEORIES"
    SYNTHESIZE_PRIMITIVE_GAP = "SYNTHESIZE_PRIMITIVE_GAP"
    ABSTAIN_BOUNDARY_MAP = "ABSTAIN_BOUNDARY_MAP"
    HALT_BUDGET_EXHAUSTION = "HALT_BUDGET_EXHAUSTION"


@dataclass
class EpistemicDecisionRecord:
    state: EpistemicOperationalState
    justification: str
    target_action: str
    confidence_level: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "justification": self.justification,
            "target_action": self.target_action,
            "confidence_level": round(self.confidence_level, 4),
        }


class EpistemicStateManager:
    """Evaluates the global scientific context and selects the next epistemic action."""

    def determine_next_action(
        self,
        theory_posterior_entropy: float,
        residual_magnitude: float,
        is_out_of_domain: bool,
        remaining_budget: float,
        max_eig: float,
    ) -> EpistemicDecisionRecord:
        """Determines the appropriate scientific action with calibrated stopping & abstention."""
        if remaining_budget <= 0.1 or max_eig < 0.05:
            return EpistemicDecisionRecord(
                state=EpistemicOperationalState.HALT_BUDGET_EXHAUSTION,
                justification="Marginal expected information gain is below cost threshold or compute budget is exhausted.",
                target_action="Terminate autonomous loop safely and preserve Claim DAG certificates.",
                confidence_level=1.00,
            )

        if is_out_of_domain:
            return EpistemicDecisionRecord(
                state=EpistemicOperationalState.ABSTAIN_BOUNDARY_MAP,
                justification="Target model/task is outside validated validity domain of active theory population.",
                target_action="Abstain from causal prediction and register structured boundary agenda.",
                confidence_level=0.98,
            )

        if abs(residual_magnitude) >= 0.20:
            return EpistemicDecisionRecord(
                state=EpistemicOperationalState.SYNTHESIZE_PRIMITIVE_GAP,
                justification=f"Persistent residual (delta={residual_magnitude:.2f}) indicates an expressive primitive gap.",
                target_action="Trigger novel mechanistic primitive discovery and causal verification.",
                confidence_level=0.95,
            )

        if theory_posterior_entropy >= 0.50:
            return EpistemicDecisionRecord(
                state=EpistemicOperationalState.DISCRIMINATE_RIVAL_THEORIES,
                justification=f"Multiple competing theories have high posterior ambiguity (entropy={theory_posterior_entropy:.2f}).",
                target_action="Schedule Max-EIG discriminating experiment to eliminate rival hypotheses.",
                confidence_level=0.92,
            )

        return EpistemicDecisionRecord(
            state=EpistemicOperationalState.EXPLOIT_ESTABLISHED_THEORY,
            justification="Active theory population is decisive and calibrated on in-domain tasks.",
            target_action="Execute prospective replication on new held-out validation tasks.",
            confidence_level=0.96,
        )
