"""Calibrated EIG Planner for MECH.

Integrates Empirical Outcome Likelihoods into EIG Planning:
Computes Expected Information Gain using live calibrated likelihoods P_theta(O | H_i, Category).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from .belief_entropy_engine import BeliefEntropyEngine, HypothesisEntropyProfile
from .eig_utility_optimizer import ScoredExperimentCandidate
from .empirical_outcome_likelihood_engine import EmpiricalOutcomeLikelihoodEngine
from .experiment_design_generator import CandidateExperimentDesign


@dataclass
class CalibratedOptimalSelection:
    """Optimal experiment selected using calibrated empirical likelihood models."""
    best_experiment: ScoredExperimentCandidate
    ranked_candidates: List[ScoredExperimentCandidate]
    current_entropy_profile: HypothesisEntropyProfile
    selection_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_experiment": self.best_experiment.to_dict(),
            "ranked_candidates": [c.to_dict() for c in self.ranked_candidates],
            "current_entropy_profile": self.current_entropy_profile.to_dict(),
            "selection_rationale": self.selection_rationale,
        }


class CalibratedEIGPlanner:
    """Computes empirically calibrated EIG and selects optimal experiments."""

    def __init__(
        self,
        likelihood_engine: Optional[EmpiricalOutcomeLikelihoodEngine] = None,
        entropy_engine: Optional[BeliefEntropyEngine] = None,
    ) -> None:
        self.likelihood_engine = likelihood_engine or EmpiricalOutcomeLikelihoodEngine()
        self.entropy_engine = entropy_engine or BeliefEntropyEngine()

    def simulate_calibrated_posterior(
        self,
        prior_probs: Dict[str, float],
        candidate: CandidateExperimentDesign,
        observed_outcome: str,  # "HIGH" or "LOW"
    ) -> Dict[str, float]:
        """Simulates Bayesian update using learned empirical likelihood P_theta(O | H_i, Category)."""
        unnorm = {}
        for h_id, prior_p in prior_probs.items():
            # Get learned calibrated probability of HIGH for this (hypothesis, category)
            p_high = self.likelihood_engine.get_calibrated_likelihood(h_id, candidate.category.value)
            likelihood = p_high if observed_outcome == "HIGH" else (1.0 - p_high)
            likelihood = max(0.02, min(0.98, likelihood))
            unnorm[h_id] = prior_p * likelihood

        total = sum(unnorm.values())
        if total < 1e-9:
            return {k: 1.0 / len(prior_probs) for k in prior_probs}
        return {k: v / total for k, v in unnorm.items()}

    def evaluate_and_select_best_experiment(
        self,
        current_hypothesis_probabilities: Dict[str, float],
        candidate_battery: List[CandidateExperimentDesign],
    ) -> CalibratedOptimalSelection:
        """Evaluates calibrated EIG and multi-factor utility U(E) across all candidates."""
        entropy_prof = self.entropy_engine.compute_entropy_profile(current_hypothesis_probabilities)
        h_0 = entropy_prof.shannon_entropy_bits

        scored_list: List[ScoredExperimentCandidate] = []

        for cand in candidate_battery:
            # 1. Calibrated marginal probability P(HIGH | E) = sum_i P(H_i) * P_theta(HIGH | H_i, Cat)
            p_high = sum(
                current_hypothesis_probabilities[h] * self.likelihood_engine.get_calibrated_likelihood(h, cand.category.value)
                for h in current_hypothesis_probabilities
            )
            p_low = max(0.01, 1.0 - p_high)
            p_high = min(0.99, max(0.01, p_high))

            # 2. Simulate posteriors and entropies
            post_high = self.simulate_calibrated_posterior(current_hypothesis_probabilities, cand, "HIGH")
            post_low = self.simulate_calibrated_posterior(current_hypothesis_probabilities, cand, "LOW")

            h_post_high = self.entropy_engine.compute_entropy_profile(post_high).shannon_entropy_bits
            h_post_low = self.entropy_engine.compute_entropy_profile(post_low).shannon_entropy_bits

            e_h_post = (p_high * h_post_high) + (p_low * h_post_low)
            eig_cal = max(0.0, h_0 - e_h_post)

            # 3. Multi-Factor Scientific Utility U(E)
            cost_norm = math.sqrt(max(1, cand.estimated_compute_passes))
            utility_u = (max(0.05, eig_cal) * cand.causal_relevance_weight * cand.validity_weight) / cost_norm

            scored_cand = ScoredExperimentCandidate(
                candidate_design=cand,
                prior_entropy_bits=round(h_0, 4),
                expected_posterior_entropy_bits=round(e_h_post, 4),
                expected_information_gain_eig=round(eig_cal, 4),
                causal_relevance=cand.causal_relevance_weight,
                validity_score=cand.validity_weight,
                compute_passes=cand.estimated_compute_passes,
                total_scientific_utility_u=round(utility_u, 4),
                selection_rank=0,
            )
            scored_list.append(scored_cand)

        # Sort descending by scientific utility U(E)
        scored_list.sort(key=lambda x: x.total_scientific_utility_u, reverse=True)

        for rank, item in enumerate(scored_list, 1):
            object.__setattr__(item, "selection_rank", rank)

        best = scored_list[0]
        rationale = (
            f"Calibrated Planner selected '{best.candidate_design.experiment_id}' ({best.candidate_design.category.value}) "
            f"with Calibrated Utility U={best.total_scientific_utility_u:.4f} (Calibrated EIG={best.expected_information_gain_eig:.4f} bits)."
        )

        return CalibratedOptimalSelection(
            best_experiment=best,
            ranked_candidates=scored_list,
            current_entropy_profile=entropy_prof,
            selection_rationale=rationale,
        )
