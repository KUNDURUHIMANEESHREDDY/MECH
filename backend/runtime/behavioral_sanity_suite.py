"""Behavioral Sanity Suite for MECH.

A small set of deterministic anchors that must pass before MECH is
permitted to run mechanistic experiments.

Design principles
-----------------
1. Does NOT require exact top-1 match — different models legitimately
   produce different top-1 tokens for the same prompt.
2. DOES require the expected target token to appear within top-k
   (default k=50) with a meaningful probability above a model-specific
   threshold.
3. Records four empirical quantities per anchor:
       R_target  — rank of expected token (0-indexed)
       P_target  — softmax probability of expected token
       P_top1    — probability of model's actual top-1 prediction
       Δz_target — logit of expected token (relative strength)
4. A probe is "sane" when R_target <= max_acceptable_rank.

Anchors are model-agnostic prompt/target pairs drawn from well-known
factual or arithmetic facts.  They are intentionally conservative —
we only demand the expected token is plausible (in top-k), not dominant.

The full suite result feeds the ModelIntegrityGate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class SanityAnchor:
    """A single deterministic sanity probe anchor."""
    anchor_id: str
    prompt: str
    expected_token: str
    description: str                      # human-readable label
    max_acceptable_rank: int = 50         # expected token must be in top-N


@dataclass
class SanityAnchorResult:
    """Empirical result for one sanity anchor."""
    anchor_id: str
    prompt: str
    expected_token: str
    observed_top1_token: str
    R_target: int           # rank of expected_token (0-indexed; -1 = not in top-k)
    P_target: float         # softmax probability of expected_token
    P_top1: float           # softmax probability of model's actual top-1
    delta_z_target: float   # logit value of expected_token
    top_k_tokens: List[str] # token strings of actual top-k predictions (live from model)
    sanity_pass: bool       # R_target <= max_acceptable_rank
    max_acceptable_rank: int
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BehavioralSanityReport:
    """Aggregate sanity report across all anchors."""
    model_id: str
    anchors_total: int
    anchors_passed: int
    anchors_failed: int
    all_passed: bool
    failed_anchor_ids: List[str]
    anchor_results: List[SanityAnchorResult]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "anchors_total": self.anchors_total,
            "anchors_passed": self.anchors_passed,
            "anchors_failed": self.anchors_failed,
            "all_passed": self.all_passed,
            "failed_anchor_ids": self.failed_anchor_ids,
            "summary": self.summary,
            "anchor_results": [r.to_dict() for r in self.anchor_results],
        }


# ── Built-in standard anchors ────────────────────────────────────────────────
# These are factual / arithmetic prompts where the target token is highly
# expected to appear somewhere in the top-50 of any well-formed language model.

STANDARD_SANITY_ANCHORS: List[SanityAnchor] = [
    SanityAnchor(
        anchor_id="sanity_capital_france",
        prompt="The capital of France is",
        expected_token=" Paris",
        description="Capital city factual recall",
        max_acceptable_rank=50,
    ),
    SanityAnchor(
        anchor_id="sanity_two_plus_two",
        prompt="Two plus two equals",
        expected_token=" four",
        description="Basic arithmetic completion",
        max_acceptable_rank=50,
    ),
    SanityAnchor(
        anchor_id="sanity_first_president",
        prompt="The first President of the United States was",
        expected_token=" George",
        description="Historical factual recall",
        max_acceptable_rank=50,
    ),
    SanityAnchor(
        anchor_id="sanity_sky_color",
        prompt="The sky appears blue because",
        expected_token=" of",
        description="Causal sentence structural completion",
        max_acceptable_rank=20,
    ),
    SanityAnchor(
        anchor_id="sanity_eiffel_location",
        prompt="The Eiffel Tower is located in",
        expected_token=" Paris",
        description="Landmark location factual recall",
        max_acceptable_rank=50,
    ),
]


def run_behavioral_sanity_suite(
    runtime,                         # ModelRuntimeInterface
    model_id: str,
    anchors: Optional[List[SanityAnchor]] = None,
    top_k: int = 50,
) -> BehavioralSanityReport:
    """
    Executes all sanity anchors against a live runtime.

    Parameters
    ----------
    runtime  : any ModelRuntimeInterface implementation
    model_id : string label for reporting
    anchors  : list of SanityAnchor (defaults to STANDARD_SANITY_ANCHORS)
    top_k    : vocabulary positions to inspect per anchor (default 50)

    Returns
    -------
    BehavioralSanityReport
        ``all_passed`` is True only when every anchor's expected token
        appears within its ``max_acceptable_rank``.

    Notes
    -----
    top-k is sourced live from the model via ``runtime.project_to_vocabulary``
    and ``runtime.forward``.  Nothing is hardcoded.
    """
    if anchors is None:
        anchors = STANDARD_SANITY_ANCHORS

    results: List[SanityAnchorResult] = []

    for anchor in anchors:
        fwd = runtime.forward(anchor.prompt, target_token=anchor.expected_token)

        # Live top-k from model vocabulary projection
        try:
            vocab_proj = runtime.project_to_vocabulary(
                hidden_state=(
                    fwd.layer_residuals.get(runtime.num_layers - 1)
                    if fwd.layer_residuals else None
                ),
                top_k=top_k,
                target_token=anchor.expected_token,
            )
            top_k_tokens = [e["token"] for e in vocab_proj.get("top_k", [])]
            top_k_probs  = {e["token"]: e["probability"] for e in vocab_proj.get("top_k", [])}
        except (ValueError, KeyError, RuntimeError) as exc:
            logger.debug("Vocab projection failed, falling back to forward pass result: %s", exc)
            top_k_tokens = [fwd.top_predicted_token]
            top_k_probs  = {fwd.top_predicted_token: fwd.target_probability or 0.0}

        # Rank of expected token (0-indexed)
        r_target = fwd.target_rank if fwd.target_rank is not None else -1

        # P_target from forward pass (most accurate source)
        p_target = fwd.target_probability or 0.0

        # P_top1: probability of model's actual top-1
        p_top1 = top_k_probs.get(fwd.top_predicted_token, 0.0)

        # Δz_target: raw logit of expected token
        delta_z_target = fwd.target_logit or 0.0

        sanity_pass = (r_target >= 0) and (r_target <= anchor.max_acceptable_rank)

        note = ""
        if not sanity_pass:
            if r_target < 0:
                note = (
                    f"'{anchor.expected_token}' not found in top-{top_k} at all — "
                    f"model top-1 was '{fwd.top_predicted_token}'. "
                    f"Verify checkpoint identity and tokenizer."
                )
            else:
                note = (
                    f"'{anchor.expected_token}' found at rank {r_target}, "
                    f"exceeds max_acceptable_rank={anchor.max_acceptable_rank}. "
                    f"Model top-1 was '{fwd.top_predicted_token}'."
                )

        results.append(SanityAnchorResult(
            anchor_id=anchor.anchor_id,
            prompt=anchor.prompt,
            expected_token=anchor.expected_token,
            observed_top1_token=fwd.top_predicted_token,
            R_target=r_target,
            P_target=round(p_target, 6),
            P_top1=round(p_top1, 6),
            delta_z_target=round(delta_z_target, 4),
            top_k_tokens=top_k_tokens[:10],   # store first 10 for reporting
            sanity_pass=sanity_pass,
            max_acceptable_rank=anchor.max_acceptable_rank,
            note=note,
        ))

    passed  = [r for r in results if r.sanity_pass]
    failed  = [r for r in results if not r.sanity_pass]
    all_ok  = len(failed) == 0

    if all_ok:
        summary = (
            f"Behavioral sanity PASSED for '{model_id}': "
            f"all {len(results)} anchors found their expected token within top-k."
        )
    else:
        fail_ids = [r.anchor_id for r in failed]
        detail = "; ".join(
            f"{r.anchor_id} expected='{r.expected_token}' rank={r.R_target} "
            f"top1='{r.observed_top1_token}'"
            for r in failed
        )
        summary = (
            f"Behavioral sanity FAILED for '{model_id}': "
            f"{len(failed)}/{len(results)} anchors failed. "
            f"Details: {detail}. "
            f"Investigate checkpoint identity and tokenizer before mechanistic analysis."
        )

    return BehavioralSanityReport(
        model_id=model_id,
        anchors_total=len(results),
        anchors_passed=len(passed),
        anchors_failed=len(failed),
        all_passed=all_ok,
        failed_anchor_ids=[r.anchor_id for r in failed],
        anchor_results=results,
        summary=summary,
    )
