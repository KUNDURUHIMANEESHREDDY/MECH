"""Behavioral Validation Gate for MECH.

Separates two orthogonal dimensions that are commonly conflated:

    execution_pass  — the infrastructure ran without error
    behavioral_pass — the model actually produced the expected token

Enforces the upstream gate:

    execution_pass = True
    AND
    behavioral_pass = True

    ──────────────────────────────────────────
    BEFORE any mechanistic result is promoted
    into the causal discovery pipeline.
    ──────────────────────────────────────────

Without this gate MECH could discover a perfectly causal circuit for
the *wrong* behavior (the model never produces the target token, so
any "explanation" is explaining noise, not the mechanism of interest).

The top_k list is sourced from the live model output — never hardcoded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class BehavioralGateDecision(str, Enum):
    """Decision made by the behavioral validation gate."""
    PROMOTE   = "PROMOTE"     # execution_pass AND behavioral_pass — safe to investigate mechanistically
    BLOCK     = "BLOCK"       # execution_pass but behavioral_pass FAILED — investigate setup, not mechanism
    ERROR     = "ERROR"       # execution itself failed — nothing to promote


@dataclass
class BehavioralValidation:
    """
    Full behavioral diagnosis for a single probe execution.

    Fields
    ------
    probe_id              : identifier from StandardMechanisticProbe
    expected_token        : the token the probe *declares* as correct (e.g., " Paris")
    observed_token        : the token the model *actually* predicted (top-1 argmax)
    expected_rank         : rank of expected_token in the model's output distribution
                            (0-indexed; 0 = top prediction, -1 = not in vocab / not measured)
    expected_probability  : softmax probability assigned to expected_token
    top_k                 : list of (token_string, probability) for the actual top-k predictions
                            sourced directly from the model — never hardcoded
    behavioral_pass       : True only when observed_token == expected_token
    execution_pass        : True when infrastructure ran without exception
    gate_decision         : PROMOTE | BLOCK | ERROR
    block_reason          : human-readable explanation when gate_decision != PROMOTE
    """
    probe_id              : str
    expected_token        : str
    observed_token        : str
    expected_rank         : int                              # 0-indexed; -1 = not measured
    expected_probability  : float
    top_k                 : List[Tuple[str, float]]          # [(token, prob), ...]  live from model
    behavioral_pass       : bool
    execution_pass        : bool
    gate_decision         : BehavioralGateDecision
    block_reason          : str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gate_decision"] = self.gate_decision.value
        d["top_k"] = [{"token": t, "probability": round(p, 6)} for t, p in self.top_k]
        return d


def build_behavioral_validation(
    probe_id: str,
    expected_token: str,
    observed_token: str,
    expected_rank: int,
    expected_probability: float,
    top_k: List[Tuple[str, float]],
    execution_pass: bool,
) -> BehavioralValidation:
    """
    Constructs a BehavioralValidation from live forward-pass measurements.

    Parameters
    ----------
    probe_id            : identifier for this probe
    expected_token      : the probe's declared target token (e.g., " Paris")
    observed_token      : `RuntimeForwardOutput.top_predicted_token` — the actual argmax
    expected_rank       : `RuntimeForwardOutput.target_rank` — live from model
    expected_probability: `RuntimeForwardOutput.target_probability` — live from model
    top_k               : from `ModelRuntimeInterface.project_to_vocabulary(top_k=10)`
                          — a list of (token_str, probability) pairs, never hardcoded
    execution_pass      : True when the runtime completed without raising an exception

    Notes
    -----
    behavioral_pass is True iff observed_token == expected_token.
    The gate is strict: both flags must be True to PROMOTE.
    """
    if not execution_pass:
        return BehavioralValidation(
            probe_id=probe_id,
            expected_token=expected_token,
            observed_token=observed_token,
            expected_rank=expected_rank,
            expected_probability=expected_probability,
            top_k=top_k,
            behavioral_pass=False,
            execution_pass=False,
            gate_decision=BehavioralGateDecision.ERROR,
            block_reason="Execution failed — infrastructure error; no behavior to validate.",
        )

    behavioral_pass = (observed_token.strip() == expected_token.strip())

    if behavioral_pass:
        gate_decision = BehavioralGateDecision.PROMOTE
        block_reason = ""
    else:
        gate_decision = BehavioralGateDecision.BLOCK
        block_reason = (
            f"Behavioral FAIL: model top-1 = '{observed_token}' "
            f"(rank={expected_rank}, p={expected_probability:.4f}); "
            f"expected '{expected_token}'. "
            f"Investigate prompt/model configuration before mechanistic analysis."
        )

    return BehavioralValidation(
        probe_id=probe_id,
        expected_token=expected_token,
        observed_token=observed_token,
        expected_rank=expected_rank,
        expected_probability=expected_probability,
        top_k=top_k,
        behavioral_pass=behavioral_pass,
        execution_pass=execution_pass,
        gate_decision=gate_decision,
        block_reason=block_reason,
    )


def assert_promotion_gate(validation: BehavioralValidation) -> None:
    """
    Raises ValueError if the behavioral gate blocks promotion.

    Call this before any mechanistic result enters the causal discovery pipeline.

    Raises
    ------
    ValueError
        When gate_decision is BLOCK or ERROR, with a diagnostic message
        that names the observed vs expected token and top-1 predictions.
    """
    if validation.gate_decision != BehavioralGateDecision.PROMOTE:
        top_tokens = ", ".join(
            f"'{t}' ({p:.4f})"
            for t, p in validation.top_k[:5]
        )
        raise ValueError(
            f"[BehavioralGate BLOCKED] probe='{validation.probe_id}' "
            f"gate={validation.gate_decision.value}\n"
            f"  expected : '{validation.expected_token}'\n"
            f"  observed : '{validation.observed_token}'\n"
            f"  rank     : {validation.expected_rank}\n"
            f"  p(target): {validation.expected_probability:.6f}\n"
            f"  top-5    : [{top_tokens}]\n"
            f"  reason   : {validation.block_reason}"
        )


def check_promotion_gate(validation: BehavioralValidation) -> bool:
    """
    Non-raising variant.  Returns True iff the gate allows promotion.
    Suitable for pipeline filters that log and skip rather than crash.
    """
    return validation.gate_decision == BehavioralGateDecision.PROMOTE
