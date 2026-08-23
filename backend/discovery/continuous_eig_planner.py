"""Continuous Differential EIG Planner for MECH.

Computes Continuous Mutual Information I(y; H | E) over multi-variate Gaussian causal outcome distributions:
EIG_cont(E) = h(y | E) - sum_i P(H_i) * h(y | H_i, E)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from .belief_entropy_engine import BeliefEntropyEngine, HypothesisEntropyProfile
from .continuous_outcome_likelihood_engine import ContinuousMeasurementVector, ContinuousOutcomeLikelihoodEngine, GaussianOutcomeParam
from .experiment_design_generator import CandidateExperimentDesign


@dataclass
class ContinuousScoredExperimentCandidate:
    """An experiment candidate scored using continuous differential EIG and multi-variate Gaussian mixtures."""
    candidate_design: CandidateExperimentDesign
    predicted_mean_delta_z: float
    marginal_variance_delta_z: float
    conditional_entropy_nats: float
    marginal_entropy_nats: float
    continuous_eig_bits: float
    causal_relevance: float
    validity_score: float
    compute_passes: int
    total_scientific_utility_u: float
    selection_rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_design": self.candidate_design.to_dict(),
            "predicted_mean_delta_z": round(self.predicted_mean_delta_z, 4),
            "marginal_variance_delta_z": round(self.marginal_variance_delta_z, 6),
            "conditional_entropy_nats": round(self.conditional_entropy_nats, 4),
            "marginal_entropy_nats": round(self.marginal_entropy_nats, 4),
            "continuous_eig_bits": round(self.continuous_eig_bits, 4),
            "causal_relevance": self.causal_relevance,
            "validity_score": self.validity_score,
            "compute_passes": self.compute_passes,
            "total_scientific_utility_u": round(self.total_scientific_utility_u, 4),
            "selection_rank": self.selection_rank,
        }


@dataclass
class ContinuousOptimalSelection:
    """Optimal experiment selection based on continuous differential EIG."""
    best_experiment: ContinuousScoredExperimentCandidate
    ranked_candidates: List[ContinuousScoredExperimentCandidate]
    current_entropy_profile: HypothesisEntropyProfile
    selection_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_experiment": self.best_experiment.to_dict(),
            "ranked_candidates": [c.to_dict() for c in self.ranked_candidates],
            "current_entropy_profile": self.current_entropy_profile.to_dict(),
            "selection_rationale": self.selection_rationale,
        }


class ContinuousEIGPlanner:
    """Computes continuous differential EIG and optimizes multi-factor scientific utility."""

    def __init__(
        self,
        likelihood_engine: Optional[ContinuousOutcomeLikelihoodEngine] = None,
        entropy_engine: Optional[BeliefEntropyEngine] = None,
    ) -> None:
        self.likelihood_engine = likelihood_engine or ContinuousOutcomeLikelihoodEngine()
        self.entropy_engine = entropy_engine or BeliefEntropyEngine()

    def calculate_continuous_eig(
        self,
        hypothesis_probabilities: Dict[str, float],
        candidate: CandidateExperimentDesign,
    ) -> Tuple[float, float, float, float, float]:
        """Calculates Continuous EIG (in bits), marginal mean, marginal variance, and entropies."""
        category = candidate.category.value
        active_hypotheses = list(hypothesis_probabilities.keys())

        # Collect Gaussian parameters for each hypothesis
        params: List[Tuple[float, GaussianOutcomeParam]] = []
        for h in active_hypotheses:
            p = hypothesis_probabilities[h]
            key = (h, category)
            param = self.likelihood_engine.likelihood_matrix.get(key)
            if param is not None:
                params.append((p, param))

        if not params:
            return 0.10, 0.15, 0.01, 0.5, 0.6

        # Calculate marginal means across metrics (Δz, rescue, specificity)
        mean_dz = sum(p * g.mu_delta_z for p, g in params)
        mean_res = sum(p * g.mu_rescue for p, g in params)
        mean_spec = sum(p * g.mu_specificity for p, g in params)

        # Calculate marginal variances (law of total variance: Var = E[Var] + Var(E))
        var_dz_marginal = sum(p * (g.var_delta_z + (g.mu_delta_z - mean_dz)**2) for p, g in params)
        var_res_marginal = sum(p * (g.var_rescue + (g.mu_rescue - mean_res)**2) for p, g in params)
        var_spec_marginal = sum(p * (g.var_specificity + (g.mu_specificity - mean_spec)**2) for p, g in params)

        # 1. Marginal Differential Entropy h(y | E) = sum_k 0.5 * ln(2*pi*e * Var_k)
        def _1d_diff_entropy(v: float) -> float:
            return 0.5 * math.log(max(1e-8, 2.0 * math.pi * math.e * v))

        h_marginal = _1d_diff_entropy(var_dz_marginal) + _1d_diff_entropy(var_res_marginal) + _1d_diff_entropy(var_spec_marginal)

        # 2. Conditional Differential Entropy sum_i P(H_i) * h(y | H_i, E)
        h_conditional = sum(
            p * (_1d_diff_entropy(g.var_delta_z) + _1d_diff_entropy(g.var_rescue) + _1d_diff_entropy(g.var_specificity))
            for p, g in params
        )

        # 3. Mutual Information in nats -> convert to bits (divide by ln 2)
        mi_nats = max(0.0, h_marginal - h_conditional)
        eig_bits = mi_nats / math.log(2.0)

        return eig_bits, mean_dz, var_dz_marginal, h_conditional, h_marginal

    def evaluate_and_select_best_continuous_experiment(
        self,
        current_hypothesis_probabilities: Dict[str, float],
        candidate_battery: List[CandidateExperimentDesign],
    ) -> ContinuousOptimalSelection:
        """Ranks candidates by continuous scientific utility U(E)."""
        entropy_prof = self.entropy_engine.compute_entropy_profile(current_hypothesis_probabilities)
        scored_list: List[ContinuousScoredExperimentCandidate] = []

        for cand in candidate_battery:
            eig_bits, mean_dz, var_dz, h_cond, h_marg = self.calculate_continuous_eig(
                hypothesis_probabilities=current_hypothesis_probabilities,
                candidate=cand,
            )

            # Scientific utility U(E)
            cost_norm = math.sqrt(max(1, cand.estimated_compute_passes))
            utility_u = (max(0.05, eig_bits) * cand.causal_relevance_weight * cand.validity_weight) / cost_norm

            scored = ContinuousScoredExperimentCandidate(
                candidate_design=cand,
                predicted_mean_delta_z=mean_dz,
                marginal_variance_delta_z=var_dz,
                conditional_entropy_nats=h_cond,
                marginal_entropy_nats=h_marg,
                continuous_eig_bits=eig_bits,
                causal_relevance=cand.causal_relevance_weight,
                validity_score=cand.validity_weight,
                compute_passes=cand.estimated_compute_passes,
                total_scientific_utility_u=utility_u,
                selection_rank=0,
            )
            scored_list.append(scored)

        scored_list.sort(key=lambda x: x.total_scientific_utility_u, reverse=True)
        for rank, item in enumerate(scored_list, 1):
            object.__setattr__(item, "selection_rank", rank)

        best = scored_list[0]
        rationale = (
            f"Continuous EIG Planner selected '{best.candidate_design.experiment_id}' ({best.candidate_design.category.value}) "
            f"with Continuous Utility U={best.total_scientific_utility_u:.4f} (Continuous EIG={best.continuous_eig_bits:.4f} bits, "
            f"Expected Δz={best.predicted_mean_delta_z:.3f}±{math.sqrt(best.marginal_variance_delta_z):.3f})."
        )

        return ContinuousOptimalSelection(
            best_experiment=best,
            ranked_candidates=scored_list,
            current_entropy_profile=entropy_prof,
            selection_rationale=rationale,
        )
