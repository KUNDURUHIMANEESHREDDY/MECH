"""Bayesian Belief Engine & Evidence Fusion Updater.

Ref: Bayesian Inference for Mechanistic Interpretability & Hypothesis Testing.

Replaces opaque confidence scalars with mathematically traceable Bayesian belief states:
    P(Mechanism | Evidence) = P(Evidence | Mechanism) * P(Mechanism) / P(Evidence)

Tracks Beta distribution hyperparameters Alpha (Supporting) and Beta (Contradictory),
95% Bayesian Credible Intervals, and step-by-step Belief Evolution history.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BeliefUpdateRecord:
    """Record of a single Bayesian update step."""
    iteration: int
    algorithm_name: str
    evidence_id: str
    prior: float
    likelihood: float
    marginal_likelihood: float
    posterior: float
    delta_belief: float
    credible_interval_95: Tuple[float, float]
    is_supporting: bool
    rationale: str
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")


@dataclass
class MechanismBelief:
    """Complete mathematically traceable Bayesian belief state for a mechanism claim."""
    claim_id: str
    title: str
    prior: float = 0.50
    posterior: float = 0.50
    uncertainty: float = 0.50
    alpha_param: float = 2.0  # Beta distribution alpha (supporting evidence weight)
    beta_param: float = 2.0   # Beta distribution beta (contradictory evidence weight)
    credible_interval_95: Tuple[float, float] = (0.20, 0.80)
    update_count: int = 0
    evidence_ids: List[str] = field(default_factory=list)
    history: List[BeliefUpdateRecord] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "title": self.title,
            "prior": round(self.prior, 4),
            "posterior": round(self.posterior, 4),
            "uncertainty": round(self.uncertainty, 4),
            "alpha_param": round(self.alpha_param, 2),
            "beta_param": round(self.beta_param, 2),
            "credible_interval_95": [round(self.credible_interval_95[0], 4), round(self.credible_interval_95[1], 4)],
            "update_count": self.update_count,
            "evidence_ids": self.evidence_ids,
            "history": [
                {
                    "iteration": h.iteration,
                    "algorithm": h.algorithm_name,
                    "evidence_id": h.evidence_id,
                    "prior": round(h.prior, 4),
                    "likelihood": round(h.likelihood, 4),
                    "posterior": round(h.posterior, 4),
                    "delta_belief": round(h.delta_belief, 4),
                    "ci_95": [round(h.credible_interval_95[0], 4), round(h.credible_interval_95[1], 4)],
                    "is_supporting": h.is_supporting,
                    "rationale": h.rationale,
                    "timestamp": h.timestamp,
                }
                for h in self.history
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class BayesianBeliefEngine:
    """Bayesian Inference Engine updating mechanism beliefs mathematically."""

    # Algorithm Likelihood Profiles: P(Evidence | Mechanism Exists) vs P(Evidence | Mechanism Absent)
    LIKELIHOOD_PROFILES = {
        "attribution_patching": (0.85, 0.20),  # High true positive, low false positive
        "acdc": (0.90, 0.15),                  # Very high specificity for minimal circuits
        "path_patching": (0.88, 0.18),         # High inter-layer edge specificity
        "causal_scrubbing": (0.95, 0.08),      # Gold standard falsification test
        "transcoders": (0.82, 0.22),           # Feature dictionary decomposition
        "feature_universality": (0.91, 0.12),  # Cross-model generalization test
    }

    def compute_credible_interval(self, alpha: float, beta: float) -> Tuple[float, float]:
        """Calculates 95% Bayesian Credible Interval from Beta(alpha, beta) parameters."""
        total = alpha + beta
        mean = alpha / total
        variance = (alpha * beta) / ((total ** 2) * (total + 1.0))
        std = math.sqrt(variance)

        ci_low = max(0.01, round(mean - (1.96 * std), 4))
        ci_high = min(0.99, round(mean + (1.96 * std), 4))
        return (ci_low, ci_high)

    def update_belief(
        self,
        current_belief: MechanismBelief,
        algorithm_name: str,
        evidence_id: str,
        is_supporting: bool = True,
        evidence_strength: float = 1.0
    ) -> MechanismBelief:
        """Applies Bayes' Rule: P(M|E) = [P(E|M) * P(M)] / P(E)."""
        prior = current_belief.posterior
        p_e_given_m, p_e_given_not_m = self.LIKELIHOOD_PROFILES.get(algorithm_name, (0.85, 0.20))

        if not is_supporting:
            # Invert likelihood for contradictory evidence
            p_e_given_m = 1.0 - p_e_given_m
            p_e_given_not_m = 1.0 - p_e_given_not_m

        # Likelihood P(E|M) adjusted by evidence strength
        likelihood = p_e_given_m * evidence_strength

        # Marginal Likelihood P(E) = P(E|M)*P(M) + P(E|~M)*P(~M)
        marginal_likelihood = (likelihood * prior) + (p_e_given_not_m * (1.0 - prior))

        # Bayes Update
        posterior = min(0.99, max(0.01, (likelihood * prior) / max(1e-6, marginal_likelihood)))
        delta = posterior - prior

        # Update Beta distribution parameters
        if is_supporting:
            current_belief.alpha_param += 1.5 * evidence_strength
        else:
            current_belief.beta_param += 2.0 * evidence_strength

        ci_95 = self.compute_credible_interval(current_belief.alpha_param, current_belief.beta_param)

        current_belief.prior = prior
        current_belief.posterior = round(posterior, 4)
        current_belief.uncertainty = round(1.0 - posterior, 4)
        current_belief.credible_interval_95 = ci_95
        current_belief.update_count += 1
        current_belief.evidence_ids.append(evidence_id)

        verdict_str = "SUPPORTED" if is_supporting else "CONTRADICTED"
        rationale = (
            f"Bayesian update step {current_belief.update_count} [{algorithm_name}]: "
            f"Likelihood P(E|M)={likelihood:.2f}. "
            f"Belief updated {prior:.2f} ➔ {posterior:.2f} (Δ {delta:+.2f}, 95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}])."
        )

        record = BeliefUpdateRecord(
            iteration=current_belief.update_count,
            algorithm_name=algorithm_name,
            evidence_id=evidence_id,
            prior=prior,
            likelihood=likelihood,
            marginal_likelihood=marginal_likelihood,
            posterior=posterior,
            delta_belief=delta,
            credible_interval_95=ci_95,
            is_supporting=is_supporting,
            rationale=rationale
        )

        current_belief.history.append(record)
        current_belief.updated_at = _dt.datetime.utcnow().isoformat() + "Z"
        return current_belief
