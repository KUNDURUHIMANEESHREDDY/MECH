"""EIG & Multi-Factor Scientific Utility Optimizer for MECH.

Implements optimal experimental design:
U(E) = ( EIG(E) * CausalRelevance(E) * Validity(E) ) / sqrt( ComputeCost(E) )

Where EIG(E) = H(P(H | D_current)) - E_O[ H(P(H | D_current, E, O)) ]
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from .belief_entropy_engine import BeliefEntropyEngine, HypothesisEntropyProfile
from .experiment_design_generator import CandidateExperimentDesign


@dataclass
class ScoredExperimentCandidate:
    """An experiment candidate scored by EIG, causal relevance, validity, and compute cost."""
    candidate_design: CandidateExperimentDesign
    prior_entropy_bits: float
    expected_posterior_entropy_bits: float
    expected_information_gain_eig: float
    causal_relevance: float
    validity_score: float
    compute_passes: int
    total_scientific_utility_u: float
    selection_rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_design": self.candidate_design.to_dict(),
            "prior_entropy_bits": self.prior_entropy_bits,
            "expected_posterior_entropy_bits": self.expected_posterior_entropy_bits,
            "expected_information_gain_eig": self.expected_information_gain_eig,
            "causal_relevance": self.causal_relevance,
            "validity_score": self.validity_score,
            "compute_passes": self.compute_passes,
            "total_scientific_utility_u": self.total_scientific_utility_u,
            "selection_rank": self.selection_rank,
        }


@dataclass
class OptimalExperimentSelection:
    """The optimal experiment selected by the EIG utility optimizer."""
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


class EIGUtilityOptimizer:
    """Calculates EIG and scores candidate experiments by scientific utility."""

    def __init__(self, entropy_engine: Optional[BeliefEntropyEngine] = None) -> None:
        self.entropy_engine = entropy_engine or BeliefEntropyEngine()

    def simulate_posterior_distribution(
        self,
        prior_probs: Dict[str, float],
        candidate: CandidateExperimentDesign,
        observed_outcome: str,  # "HIGH" or "LOW"
    ) -> Dict[str, float]:
        """Simulates Bayesian update for a given observed outcome."""
        unnorm = {}
        for h_id, prior_p in prior_probs.items():
            pred = candidate.hypothesis_predictions.get(h_id, "HIGH")
            match = (pred == observed_outcome)
            likelihood = 0.90 if match else 0.08
            unnorm[h_id] = prior_p * likelihood

        total = sum(unnorm.values())
        if total < 1e-9:
            return {k: 1.0 / len(prior_probs) for k in prior_probs}
        return {k: v / total for k, v in unnorm.items()}

    def evaluate_and_select_best_experiment(
        self,
        current_hypothesis_probabilities: Dict[str, float],
        candidate_battery: List[CandidateExperimentDesign],
    ) -> OptimalExperimentSelection:
        """Evaluates EIG and multi-factor utility U(E) across all candidates, ranking them."""
        entropy_prof = self.entropy_engine.compute_entropy_profile(current_hypothesis_probabilities)
        h_0 = entropy_prof.shannon_entropy_bits

        scored_list: List[ScoredExperimentCandidate] = []

        for cand in candidate_battery:
            # 1. Marginal probability of outcome HIGH vs LOW
            p_high = sum(
                current_hypothesis_probabilities[h]
                for h, pred in cand.hypothesis_predictions.items()
                if pred == "HIGH" and h in current_hypothesis_probabilities
            )
            p_low = max(0.01, 1.0 - p_high)
            p_high = min(0.99, max(0.01, p_high))

            # 2. Simulate posteriors and entropies
            post_high = self.simulate_posterior_distribution(current_hypothesis_probabilities, cand, "HIGH")
            post_low = self.simulate_posterior_distribution(current_hypothesis_probabilities, cand, "LOW")

            h_post_high = self.entropy_engine.compute_entropy_profile(post_high).shannon_entropy_bits
            h_post_low = self.entropy_engine.compute_entropy_profile(post_low).shannon_entropy_bits

            e_h_post = (p_high * h_post_high) + (p_low * h_post_low)
            eig = max(0.0, h_0 - e_h_post)

            # 3. Scientific Utility U(E)
            cost_norm = math.sqrt(max(1, cand.estimated_compute_passes))
            utility_u = (max(0.05, eig) * cand.causal_relevance_weight * cand.validity_weight) / cost_norm

            scored_cand = ScoredExperimentCandidate(
                candidate_design=cand,
                prior_entropy_bits=round(h_0, 4),
                expected_posterior_entropy_bits=round(e_h_post, 4),
                expected_information_gain_eig=round(eig, 4),
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
            f"Selected optimal experiment '{best.candidate_design.experiment_id}' ({best.candidate_design.category.value}) "
            f"with maximum Scientific Utility U={best.total_scientific_utility_u:.4f} (EIG={best.expected_information_gain_eig:.4f} bits, "
            f"CausalRelevance={best.causal_relevance:.2f}, Cost={best.compute_passes} passes)."
        )

        return OptimalExperimentSelection(
            best_experiment=best,
            ranked_candidates=scored_list,
            current_entropy_profile=entropy_prof,
            selection_rationale=rationale,
        )
