r"""Revision Evidence Engine & Bayesian Hypothesis Evaluator for MECH.

Computes posterior probabilities across competing revision hypotheses:
    P(H_i | E) ∝ P(E | H_i) * P(H_i) * exp(-complexity_penalty)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .theory_revision_engine import RevisionHypothesis


@dataclass
class HypothesisEvidenceScore:
    hypothesis_id: str
    log_likelihood: float
    prior_probability: float
    complexity_penalty: float
    unnormalized_score: float
    posterior_probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "log_likelihood": round(self.log_likelihood, 4),
            "prior_probability": round(self.prior_probability, 4),
            "complexity_penalty": round(self.complexity_penalty, 4),
            "unnormalized_score": round(self.unnormalized_score, 4),
            "posterior_probability": round(self.posterior_probability, 4),
        }


@dataclass
class RevisionPosteriorScorecard:
    scores: List[HypothesisEvidenceScore]
    winning_hypothesis_id: str
    selection_margin: float
    is_decisive: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scores": [s.to_dict() for s in self.scores],
            "winning_hypothesis_id": self.winning_hypothesis_id,
            "selection_margin": round(self.selection_margin, 4),
            "is_decisive": self.is_decisive,
        }


class RevisionEvidenceEngine:
    """Evaluates Bayesian evidence across competing theory revisions."""

    def evaluate_hypotheses(
        self,
        hypotheses: List[RevisionHypothesis],
        empirical_r_observed: float,
        empirical_delta_z_observed: float,
        r_routing_feature: float = 0.65,
        h_poly_feature: float = 0.20,
        delta_dim_feature: float = 0.10,
        runner=None,
        probes=None,
    ) -> RevisionPosteriorScorecard:
        """Computes posteriors by comparing candidate equation predictions to empirical evidence.

        Args:
            hypotheses: List of competing revision hypotheses.
            empirical_r_observed: Observed empirical transfer rate.
            empirical_delta_z_observed: Observed empirical delta-z.
            r_routing_feature: Routing feature value.
            h_poly_feature: Polysemanticity feature value.
            delta_dim_feature: Dimension delta feature value.
            runner: Optional GPT-2 runner for live measurements. When provided alongside
                probes, the likelihood variance is empirically calibrated via live causal
                effect estimates.
            probes: Optional list of probes for live measurements.

        When runner and probes are provided, sigma is derived from live GPT-2
        _baseline_causal_effect calls and the empirical observation is anchored to a
        live pass_rate. Otherwise falls back to fixed sigma=0.05 with a deprecation
        warning.
        """
        if runner is not None and probes:
            # Empirically calibrate sigma using live baseline causal effect
            probe = probes[0]
            bce = runner._baseline_causal_effect(probe)
            # Sigma derived as inverse of normalised causal effect (stronger signal -> tighter sigma)
            sigma = max(0.01, 0.5 / (bce + 1.0))

            # Pass-rate weighting: fraction of probes where target_rank < 100
            passed = 0
            for p in probes:
                fwd = runner.runtime.forward(p.clean_prompt, target_token=p.target_token)
                if fwd.target_rank is not None and fwd.target_rank < 100:
                    passed += 1
            pass_rate = passed / len(probes)
            # Scale empirical observation by live pass_rate (acts as a reliability anchor)
            effective_r_observed = empirical_r_observed * pass_rate
        else:
            warnings.warn(
                "evaluate_hypotheses called without runner/probes. "
                "Falling back to fixed sigma=0.05. Pass runner and probes for live GPT-2 measurements.",
                stacklevel=2,
            )
            sigma = 0.05
            effective_r_observed = empirical_r_observed

        scores: List[HypothesisEvidenceScore] = []
        raw_scores: List[float] = []

        for h in hypotheses:
            pred_r = h.candidate_equation.predict_transfer(
                s_role=0.85,
                a_linear=0.80,
                h_poly=h_poly_feature,
                delta_dim=delta_dim_feature,
                r_routing=r_routing_feature,
            )

            # Gaussian likelihood
            err = abs(pred_r - effective_r_observed)
            log_lik = -0.5 * (err / sigma) ** 2

            score = math.exp(log_lik) * h.prior_probability * math.exp(-h.complexity_penalty)
            raw_scores.append(score)
            scores.append(
                HypothesisEvidenceScore(
                    hypothesis_id=h.hypothesis_id,
                    log_likelihood=log_lik,
                    prior_probability=h.prior_probability,
                    complexity_penalty=h.complexity_penalty,
                    unnormalized_score=score,
                    posterior_probability=0.0,
                )
            )

        total_score = sum(raw_scores) or 1.0
        for i, s in enumerate(scores):
            s.posterior_probability = raw_scores[i] / total_score

        sorted_scores = sorted(scores, key=lambda x: x.posterior_probability, reverse=True)
        winner = sorted_scores[0]
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else None

        margin = (winner.posterior_probability - runner_up.posterior_probability) if runner_up else 1.0
        is_decisive = (winner.posterior_probability >= 0.70) and (margin >= 0.40)

        return RevisionPosteriorScorecard(
            scores=sorted_scores,
            winning_hypothesis_id=winner.hypothesis_id,
            selection_margin=margin,
            is_decisive=is_decisive,
        )
