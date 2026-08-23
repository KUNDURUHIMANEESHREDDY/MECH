"""Empirical Outcome Likelihood Engine for MECH.

Maintains and adapts empirical likelihood parameters P_theta(O | H_i, E) using online Beta conjugate updating:
1. Replaces static likelihood heuristics with learnable empirical distributions.
2. Updates Beta(alpha, beta) parameters based on real observed causal intervention outcomes.
3. Quantifies likelihood confidence and parameter variance.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .experiment_design_generator import ExperimentCategory


@dataclass
class EmpiricalLikelihoodParam:
    """Beta distribution parameters representing P(HIGH | H_i, Category)."""
    hypothesis_id: str
    experiment_category: str
    alpha_high: float
    beta_low: float
    total_observations: int
    last_updated_utc: str

    @property
    def expected_pass_probability(self) -> float:
        """Posterior mean E[P] = alpha / (alpha + beta)."""
        return self.alpha_high / (self.alpha_high + self.beta_low)

    @property
    def variance(self) -> float:
        """Posterior variance Var[P] = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))."""
        ab = self.alpha_high + self.beta_low
        return (self.alpha_high * self.beta_low) / (ab * ab * (ab + 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "experiment_category": self.experiment_category,
            "alpha_high": round(self.alpha_high, 3),
            "beta_low": round(self.beta_low, 3),
            "total_observations": self.total_observations,
            "expected_pass_probability": round(self.expected_pass_probability, 4),
            "variance": round(self.variance, 6),
            "last_updated_utc": self.last_updated_utc,
        }


class EmpiricalOutcomeLikelihoodEngine:
    """Maintains and updates empirical likelihood models P_theta(O | H_i, Category)."""

    def __init__(self) -> None:
        self.likelihood_matrix: Dict[Tuple[str, str], EmpiricalLikelihoodParam] = {}
        self._initialize_default_priors()

    def _initialize_default_priors(self) -> None:
        """Initializes uninformative or weakly informative Beta priors."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        categories = [
            ExperimentCategory.POSITIONAL_PERTURBATION.value,
            ExperimentCategory.LEXICAL_SYNTAX_CONTROL.value,
            ExperimentCategory.SEMANTIC_TOPIC_PROBE.value,
            ExperimentCategory.MEDIATION_KNOCKOUT.value,
            ExperimentCategory.CAUSAL_ZERO_ABLATION.value,
        ]
        hypotheses = ["H1_Relational", "H2_BroadTopic", "H3_LexicalTrigger", "H4_PositionalArtifact"]

        # Default weak priors (effective sample size = 10)
        for h in hypotheses:
            for c in categories:
                # Default baseline: assume high likelihood if hypothesis aligns with category
                if h == "H1_Relational":
                    a, b = 8.5, 1.5
                elif h == "H4_PositionalArtifact" and c == ExperimentCategory.POSITIONAL_PERTURBATION.value:
                    a, b = 1.5, 8.5
                elif h == "H3_LexicalTrigger" and c == ExperimentCategory.LEXICAL_SYNTAX_CONTROL.value:
                    a, b = 1.5, 8.5
                elif h == "H2_BroadTopic" and c == ExperimentCategory.SEMANTIC_TOPIC_PROBE.value:
                    a, b = 8.5, 1.5
                else:
                    a, b = 5.0, 5.0

                self.likelihood_matrix[(h, c)] = EmpiricalLikelihoodParam(
                    hypothesis_id=h,
                    experiment_category=c,
                    alpha_high=a,
                    beta_low=b,
                    total_observations=0,
                    last_updated_utc=ts,
                )

    def get_calibrated_likelihood(
        self,
        hypothesis_id: str,
        experiment_category: str,
    ) -> float:
        """Returns calibrated probability P_theta(HIGH | H_i, Category)."""
        key = (hypothesis_id, experiment_category)
        if key in self.likelihood_matrix:
            return self.likelihood_matrix[key].expected_pass_probability
        return 0.50

    def update_with_observation(
        self,
        hypothesis_id: str,
        experiment_category: str,
        observed_outcome: str,  # "HIGH" or "LOW"
    ) -> EmpiricalLikelihoodParam:
        """Performs Bayesian Beta conjugate update: alpha += 1 if HIGH, beta += 1 if LOW."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        key = (hypothesis_id, experiment_category)
        if key not in self.likelihood_matrix:
            self.likelihood_matrix[key] = EmpiricalLikelihoodParam(
                hypothesis_id=hypothesis_id,
                experiment_category=experiment_category,
                alpha_high=5.0,
                beta_low=5.0,
                total_observations=0,
                last_updated_utc=ts,
            )

        param = self.likelihood_matrix[key]
        if observed_outcome == "HIGH":
            param.alpha_high += 1.0
        else:
            param.beta_low += 1.0

        param.total_observations += 1
        param.last_updated_utc = ts
        return param
