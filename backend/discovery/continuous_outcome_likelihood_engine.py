"""Continuous Causal Measurement Likelihood Engine for MECH.

Models multi-variate continuous causal outcomes y = [Δz, Δp, R_rescue, Specificity]^T:
1. Replaces binary discretizations with continuous Gaussian likelihood distributions N(mu, Sigma).
2. Implements online Bayesian conjugate / Welford updates over continuous multi-metric outcomes.
3. Computes exact continuous probability density functions P(y | H_i, E) for Bayesian posterior updating.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .experiment_design_generator import ExperimentCategory


@dataclass(frozen=True)
class ContinuousMeasurementVector:
    """Multi-variate continuous measurement vector observed from a live intervention."""
    delta_z: float                      # Logit displacement Δz
    delta_probability: float            # Softmax probability displacement Δp
    mediation_rescue_fraction: float    # Upstream restoration fraction R_rescue
    control_specificity_ratio: float    # Specificity over negative controls

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GaussianOutcomeParam:
    """Continuous Gaussian parameters representing P(y | H_i, Category)."""
    hypothesis_id: str
    experiment_category: str
    mu_delta_z: float
    var_delta_z: float
    mu_rescue: float
    var_rescue: float
    mu_specificity: float
    var_specificity: float
    observations_count: int
    last_updated_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "experiment_category": self.experiment_category,
            "mu_delta_z": round(self.mu_delta_z, 4),
            "var_delta_z": round(self.var_delta_z, 6),
            "std_delta_z": round(math.sqrt(max(1e-6, self.var_delta_z)), 4),
            "mu_rescue": round(self.mu_rescue, 4),
            "var_rescue": round(self.var_rescue, 6),
            "mu_specificity": round(self.mu_specificity, 4),
            "var_specificity": round(self.var_specificity, 6),
            "observations_count": self.observations_count,
            "last_updated_utc": self.last_updated_utc,
        }


class ContinuousOutcomeLikelihoodEngine:
    """Maintains and updates continuous multi-variate Gaussian likelihood models P(y | H_i, Category)."""

    def __init__(self) -> None:
        self.likelihood_matrix: Dict[Tuple[str, str], GaussianOutcomeParam] = {}
        self._initialize_continuous_priors()

    def _initialize_continuous_priors(self) -> None:
        """Initializes continuous Gaussian outcome priors for all hypothesis-category pairs."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        categories = [
            ExperimentCategory.POSITIONAL_PERTURBATION.value,
            ExperimentCategory.LEXICAL_SYNTAX_CONTROL.value,
            ExperimentCategory.SEMANTIC_TOPIC_PROBE.value,
            ExperimentCategory.MEDIATION_KNOCKOUT.value,
            ExperimentCategory.CAUSAL_ZERO_ABLATION.value,
        ]
        hypotheses = ["H1_Relational", "H2_BroadTopic", "H3_LexicalTrigger", "H4_PositionalArtifact"]

        for h in hypotheses:
            for c in categories:
                # 1. H1 (Relational Computation): expects strong Δz, high rescue, high specificity
                if h == "H1_Relational":
                    mu_dz, var_dz = 0.280, 0.050**2
                    mu_res, var_res = 0.760, 0.080**2
                    mu_spec, var_spec = 4.800, 0.600**2
                # 2. H2 (Broad Topic): expects high Δz on topic prompts, but low specificity and low mediation
                elif h == "H2_BroadTopic":
                    mu_dz, var_dz = (0.240, 0.060**2) if c == ExperimentCategory.SEMANTIC_TOPIC_PROBE.value else (0.050, 0.030**2)
                    mu_res, var_res = 0.200, 0.080**2
                    mu_spec, var_spec = 1.300, 0.300**2
                # 3. H3 (Lexical Syntax Trigger): expects near-zero Δz when phrasing changes (lexical control)
                elif h == "H3_LexicalTrigger":
                    mu_dz, var_dz = (0.020, 0.020**2) if c == ExperimentCategory.LEXICAL_SYNTAX_CONTROL.value else (0.180, 0.050**2)
                    mu_res, var_res = 0.120, 0.050**2
                    mu_spec, var_spec = 1.100, 0.200**2
                # 4. H4 (Positional Artifact): expects near-zero Δz when token offsets shift (positional perturb)
                else:
                    mu_dz, var_dz = (0.015, 0.020**2) if c == ExperimentCategory.POSITIONAL_PERTURBATION.value else (0.150, 0.050**2)
                    mu_res, var_res = 0.100, 0.050**2
                    mu_spec, var_spec = 1.050, 0.200**2

                self.likelihood_matrix[(h, c)] = GaussianOutcomeParam(
                    hypothesis_id=h,
                    experiment_category=c,
                    mu_delta_z=mu_dz,
                    var_delta_z=var_dz,
                    mu_rescue=mu_res,
                    var_rescue=var_res,
                    mu_specificity=mu_spec,
                    var_specificity=var_spec,
                    observations_count=0,
                    last_updated_utc=ts,
                )

    def compute_continuous_density(
        self,
        hypothesis_id: str,
        experiment_category: str,
        observed_vector: ContinuousMeasurementVector,
    ) -> float:
        """Computes continuous Gaussian probability density P(y | H_i, Category)."""
        key = (hypothesis_id, experiment_category)
        if key not in self.likelihood_matrix:
            return 1e-4

        param = self.likelihood_matrix[key]

        def _1d_gaussian_pdf(x: float, mu: float, var: float) -> float:
            v = max(1e-6, var)
            diff = x - mu
            exponent = - (diff * diff) / (2.0 * v)
            denom = math.sqrt(2.0 * math.pi * v)
            return max(1e-8, (1.0 / denom) * math.exp(exponent))

        p_dz = _1d_gaussian_pdf(observed_vector.delta_z, param.mu_delta_z, param.var_delta_z)
        p_res = _1d_gaussian_pdf(observed_vector.mediation_rescue_fraction, param.mu_rescue, param.var_rescue)
        p_spec = _1d_gaussian_pdf(observed_vector.control_specificity_ratio, param.mu_specificity, param.var_specificity)

        # Joint density under conditional independence
        joint_density = p_dz * p_res * p_spec
        return max(1e-12, joint_density)

    def update_with_continuous_observation(
        self,
        hypothesis_id: str,
        experiment_category: str,
        observed_vector: ContinuousMeasurementVector,
        learning_rate: float = 0.20,
    ) -> GaussianOutcomeParam:
        """Updates continuous Gaussian parameters via online Bayesian learning."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        key = (hypothesis_id, experiment_category)
        if key not in self.likelihood_matrix:
            self.likelihood_matrix[key] = GaussianOutcomeParam(
                hypothesis_id=hypothesis_id,
                experiment_category=experiment_category,
                mu_delta_z=observed_vector.delta_z,
                var_delta_z=0.010,
                mu_rescue=observed_vector.mediation_rescue_fraction,
                var_rescue=0.010,
                mu_specificity=observed_vector.control_specificity_ratio,
                var_specificity=0.500,
                observations_count=1,
                last_updated_utc=ts,
            )
            return self.likelihood_matrix[key]

        param = self.likelihood_matrix[key]
        n = param.observations_count + 1
        lr = min(learning_rate, 1.0 / (1.0 + 0.1 * n))

        # Online update of means
        diff_dz = observed_vector.delta_z - param.mu_delta_z
        param.mu_delta_z += lr * diff_dz
        param.var_delta_z = max(1e-5, (1.0 - lr) * param.var_delta_z + lr * (diff_dz * diff_dz))

        diff_res = observed_vector.mediation_rescue_fraction - param.mu_rescue
        param.mu_rescue += lr * diff_res
        param.var_rescue = max(1e-5, (1.0 - lr) * param.var_rescue + lr * (diff_res * diff_res))

        diff_spec = observed_vector.control_specificity_ratio - param.mu_specificity
        param.mu_specificity += lr * diff_spec
        param.var_specificity = max(1e-4, (1.0 - lr) * param.var_specificity + lr * (diff_spec * diff_spec))

        param.observations_count = n
        param.last_updated_utc = ts
        return param
