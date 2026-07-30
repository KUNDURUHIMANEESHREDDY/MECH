"""Multi-Agent Peer Review Panel for Mechanistic Claims.

Implements Methodology, Statistical, Mechanistic, and Reproducibility reviewers.
Provides diagnostic scorecards and 3-tier verdicts (PASS, REVISION_REQUIRED, FAIL).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReviewVerdict(str, Enum):
    PASS = "PASS"
    REVISION_REQUIRED = "REVISION_REQUIRED"
    FAIL = "FAIL"


@dataclass
class ReviewerOutput:
    reviewer_name: str
    verdict: ReviewVerdict
    score: int
    max_score: int
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PeerReviewResult:
    claim_id: str
    final_verdict: ReviewVerdict
    reviewer_outputs: List[ReviewerOutput]
    overall_score_pct: float
    unanimous_pass: bool


class PeerReviewPanel:
    """Orchestrates scientific peer review across specialized agents."""

    def review_claim(self, claim_id: str, evidence: Dict[str, Any], metadata: Dict[str, Any]) -> PeerReviewResult:
        outputs = [
            self._review_methodology(evidence, metadata),
            self._review_statistics(evidence, metadata),
            self._review_mechanistic_logic(evidence, metadata),
            self._review_reproducibility(metadata),
        ]

        # Final Verdict Logic: Unanimous PASS required for global PASS
        all_pass = all(o.verdict == ReviewVerdict.PASS for o in outputs)
        any_fail = any(o.verdict == ReviewVerdict.FAIL for o in outputs)

        if all_pass:
            final_verdict = ReviewVerdict.PASS
        elif any_fail:
            final_verdict = ReviewVerdict.FAIL
        else:
            final_verdict = ReviewVerdict.REVISION_REQUIRED

        total_score = sum(o.score for o in outputs)
        max_total = sum(o.max_score for o in outputs)
        score_pct = (total_score / max_total) * 100 if max_total > 0 else 0.0

        return PeerReviewResult(
            claim_id=claim_id,
            final_verdict=final_verdict,
            reviewer_outputs=outputs,
            overall_score_pct=round(score_pct, 2),
            unanimous_pass=all_pass
        )

    def _review_methodology(self, evidence: Dict[str, Any], metadata: Dict[str, Any]) -> ReviewerOutput:
        reasoning = []
        score = 0
        max_score = 9

        # 9-Point Checklist logic (simplified)
        checklist = {
            "clean_corrupted_split": metadata.get("has_corrupted_baseline", False),
            "mean_ablation_used": metadata.get("ablation_type") == "mean",
            "random_seed_fixed": "seed" in metadata,
            "target_tokens_identified": "correct_token_id" in evidence or "io" in evidence,
            "intervention_layer_valid": evidence.get("patch_layer", 0) > 0,
            "prompt_metadata_present": "raw_traces" in metadata or "prompts" in metadata,
            "control_experiment_run": metadata.get("has_controls", False),
            "patching_algorithm_vetted": "patching" in metadata.get("algorithm", "").lower(),
            "dataset_hash_provided": "dataset_hash" in metadata
        }

        for item, passed in checklist.items():
            if passed:
                score += 1
            else:
                reasoning.append(f"Missing methodology requirement: {item}")

        verdict = ReviewVerdict.PASS if score >= 8 else ReviewVerdict.REVISION_REQUIRED
        if score < 5:
            verdict = ReviewVerdict.FAIL

        return ReviewerOutput("Methodology Reviewer", verdict, score, max_score, reasoning)

    def _review_statistics(self, evidence: Dict[str, Any], metadata: Dict[str, Any]) -> ReviewerOutput:
        reasoning = []
        score = 0
        max_score = 20

        n = evidence.get("n_samples", 0)
        ci_low = evidence.get("ci_low", 0)
        ci_high = evidence.get("ci_high", 0)
        ci_width = ci_high - ci_low
        fidelity = evidence.get("fidelity_pct", 0)

        # Scoring
        if n >= 100: score += 5
        elif n >= 40: score += 2
        else: reasoning.append(f"Small sample size: N={n}")

        if 0 < ci_width <= 5.0: score += 10
        elif ci_width <= 15.0: score += 5
        else: reasoning.append(f"High variance: CI width={ci_width:.2f}")

        if fidelity > 90: score += 5
        elif fidelity > 80: score += 3

        verdict = ReviewVerdict.PASS if score >= 15 else ReviewVerdict.REVISION_REQUIRED
        if score < 10:
            verdict = ReviewVerdict.FAIL

        return ReviewerOutput("Statistical Reviewer", verdict, score, max_score, reasoning)

    def _review_mechanistic_logic(self, evidence: Dict[str, Any], metadata: Dict[str, Any]) -> ReviewerOutput:
        reasoning = []
        score = 0
        max_score = 20

        # Causal Chain Validation
        patch_success = evidence.get("patch_success_rate", 0)
        faithfulness = evidence.get("primary_score", 0) # Usually faithfulness in IOI

        if patch_success >= 90: score += 10
        elif patch_success >= 70: score += 5
        else: reasoning.append(f"Low patching success rate: {patch_success}%")

        if faithfulness >= 80: score += 10
        elif faithfulness >= 60: score += 5
        else: reasoning.append(f"Low circuit faithfulness: {faithfulness}%")

        verdict = ReviewVerdict.PASS if score >= 15 else ReviewVerdict.REVISION_REQUIRED
        if score < 10:
            verdict = ReviewVerdict.FAIL

        return ReviewerOutput("Mechanistic Reviewer", verdict, score, max_score, reasoning)

    def _review_reproducibility(self, metadata: Dict[str, Any]) -> ReviewerOutput:
        reasoning = []
        score = 0
        max_score = 10

        requirements = [
            "git_sha", "dataset_hash", "model_id", "torch_version", "transformers_version", "seed"
        ]

        for req in requirements:
            if metadata.get(req) or metadata.get("env", {}).get(req):
                score += 1.6 # approx
            else:
                reasoning.append(f"Missing reproducibility metadata: {req}")

        # rounding
        score = int(min(score, 10))

        verdict = ReviewVerdict.PASS if score >= 8 else ReviewVerdict.REVISION_REQUIRED
        if score < 5:
            verdict = ReviewVerdict.FAIL

        return ReviewerOutput("Reproducibility Reviewer", verdict, score, 10, reasoning)
