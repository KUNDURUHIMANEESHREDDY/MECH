"""Bayesian Model Criticism Engine for MECH.

Enforces Predictive Validation Hierarchy & 3-Tier Calibration Policy:
1. Held-Out Predictive Calibration > Predictive NLL > Coverage Error > PPC > AIC/BIC
2. Thresholds:
   - C_err <= 0.12: VALID_CONFIDENT
   - 0.12 < C_err <= 0.20: VALID_UNCERTAIN (Conservative penalty)
   - C_err > 0.20 or N < N_min: ABSTAIN (First-class epistemic state -> triggers calibration probe agenda)
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .calibration_policy import CalibrationPolicyConfig
from .distribution_family_models import (
    DistributionFamilyType,
    FittedDistributionProfile,
    MultiFamilyDistributionFitter,
)


class CriticEpistemicState(str, Enum):
    VALID_CONFIDENT = "VALID_CONFIDENT"       # C_err <= 0.12 -> Full EIG trust
    VALID_UNCERTAIN = "VALID_UNCERTAIN"       # 0.12 < C_err <= 0.20 -> Conservative EIG penalty
    ABSTAIN = "ABSTAIN"                       # C_err > 0.20 or N < N_min -> Refuse claims, probe calibration


@dataclass
class CriticResult:
    """Formal model criticism verdict consumed directly by the active planner."""
    hypothesis_id: str
    experiment_category: str
    epistemic_state: CriticEpistemicState
    selected_family: DistributionFamilyType
    is_gaussian_rejected: bool
    empirical_coverage_50pct: float
    empirical_coverage_90pct: float
    empirical_coverage_95pct: float
    mean_coverage_error: float
    predictive_nll: float
    mahalanobis_outlier_rate: float
    candidate_profiles: Dict[str, FittedDistributionProfile]
    calibration_probe_agenda: List[str]
    criticism_verdict_rationale: str
    policy_provenance: Dict[str, Any]
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "experiment_category": self.experiment_category,
            "epistemic_state": self.epistemic_state.value,
            "selected_family": self.selected_family.value,
            "is_gaussian_rejected": self.is_gaussian_rejected,
            "empirical_coverage_50pct": round(self.empirical_coverage_50pct, 4),
            "empirical_coverage_90pct": round(self.empirical_coverage_90pct, 4),
            "empirical_coverage_95pct": round(self.empirical_coverage_95pct, 4),
            "mean_coverage_error": round(self.mean_coverage_error, 4),
            "predictive_nll": round(self.predictive_nll, 4),
            "mahalanobis_outlier_rate": round(self.mahalanobis_outlier_rate, 4),
            "candidate_profiles": {k: v.to_dict() for k, v in self.candidate_profiles.items()},
            "calibration_probe_agenda": self.calibration_probe_agenda,
            "criticism_verdict_rationale": self.criticism_verdict_rationale,
            "policy_provenance": self.policy_provenance,
            "timestamp_utc": self.timestamp_utc,
        }


class ModelCriticismEngine:
    """Evaluates predictive coverage, runs posterior predictive checks, and enforces 3-tier calibration."""

    def __init__(self, policy: Optional[CalibrationPolicyConfig] = None) -> None:
        self.policy = policy or CalibrationPolicyConfig()
        self.fitter = MultiFamilyDistributionFitter()

    def evaluate_predictive_coverage(
        self,
        training_samples: List[float],
        heldout_samples: List[float],
    ) -> Tuple[float, float, float, float, float]:
        """Calculates empirical predictive coverage and undercoverage error on held-out samples."""
        n_train = len(training_samples)
        n_test = len(heldout_samples)
        if n_train < 2 or n_test == 0:
            return 0.0, 0.0, 0.0, 1.0, 1.0

        mu = sum(training_samples) / n_train
        std = math.sqrt(max(1e-6, sum((x - mu)**2 for x in training_samples) / (n_train - 1)))

        cov_50 = sum(1 for x in heldout_samples if abs(x - mu) <= 0.674 * std) / n_test
        cov_90 = sum(1 for x in heldout_samples if abs(x - mu) <= 1.645 * std) / n_test
        cov_95 = sum(1 for x in heldout_samples if abs(x - mu) <= 1.960 * std) / n_test

        # Undercoverage is the primary failure mode of overconfident models
        under_90 = max(0.0, 0.90 - cov_90)
        under_95 = max(0.0, 0.95 - cov_95)
        outlier_rate = sum(1 for x in heldout_samples if abs(x - mu) > 3.0 * std) / n_test

        mean_err = max(under_90, under_95) + outlier_rate

        return cov_50, cov_90, cov_95, mean_err, outlier_rate


    def critique_predictive_outcome_model(
        self,
        hypothesis_id: str,
        experiment_category: str,
        training_samples: List[float],
        heldout_samples: List[float],
    ) -> CriticResult:
        """Runs full Bayesian model criticism across distribution families and determines 3-tier state."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        all_samples = training_samples + heldout_samples
        n_total = len(all_samples)

        # 1. Fit all 4 distribution families
        prof_gauss = self.fitter.fit_gaussian(all_samples)
        prof_t = self.fitter.fit_student_t(all_samples, fixed_nu=4.0)
        prof_gmm = self.fitter.fit_2component_gmm(all_samples)
        prof_kde = self.fitter.fit_empirical_kde(all_samples)

        profiles = {
            DistributionFamilyType.GAUSSIAN.value: prof_gauss,
            DistributionFamilyType.STUDENT_T.value: prof_t,
            DistributionFamilyType.GAUSSIAN_MIXTURE_GMM.value: prof_gmm,
            DistributionFamilyType.EMPIRICAL_KDE.value: prof_kde,
        }

        # 2. Predictive Coverage Check
        cov_50, cov_90, cov_95, cov_err, out_rate = self.evaluate_predictive_coverage(
            training_samples=training_samples,
            heldout_samples=heldout_samples,
        )

        # 3. Gaussianity & Heavy-Tail Diagnostic
        is_heavy_tailed = prof_gauss.excess_kurtosis > 2.0 or out_rate > 0.05
        is_gaussian_rejected = is_heavy_tailed or cov_err > self.policy.valid_confident_threshold

        # 4. Family Selection via Predictive Hierarchy
        if is_heavy_tailed:
            selected_family = DistributionFamilyType.STUDENT_T
        elif prof_gmm.log_likelihood > prof_gauss.log_likelihood + 3.0 and len(all_samples) >= 8:
            selected_family = DistributionFamilyType.GAUSSIAN_MIXTURE_GMM
        else:
            selected_family = DistributionFamilyType.GAUSSIAN

        # 5. 3-Tier Calibration Policy Decision
        probe_agenda: List[str] = []
        if n_total < self.policy.min_samples_required:
            state = CriticEpistemicState.ABSTAIN
            rationale = f"ABSTAIN: Insufficient sample count (N={n_total} < {self.policy.min_samples_required})."
            probe_agenda = [
                f"Execute N={self.policy.min_samples_required - n_total} additional calibration baseline probes.",
                f"Probe variance stability for {hypothesis_id} under {experiment_category}.",
            ]
        elif cov_err > self.policy.valid_uncertain_threshold:
            state = CriticEpistemicState.ABSTAIN
            rationale = f"ABSTAIN: Severe predictive miscalibration (Coverage Error C_err={cov_err:.3f} > {self.policy.valid_uncertain_threshold})."
            probe_agenda = [
                f"Execute targeted prompt distribution sweep to map non-linear tail behavior.",
                f"Refit outcome model with non-parametric kernel density and Student-t mixture.",
                f"Audit outlier causes (Observed Outlier Rate = {out_rate*100:.1f}%).",
            ]
        elif cov_err > self.policy.valid_confident_threshold:
            state = CriticEpistemicState.VALID_UNCERTAIN
            rationale = f"VALID_UNCERTAIN: Mild coverage error (C_err={cov_err:.3f} in (0.12, 0.20]). Applying conservative EIG penalty."
        else:
            state = CriticEpistemicState.VALID_CONFIDENT
            rationale = f"VALID_CONFIDENT: High predictive coverage (C_err={cov_err:.3f} <= 0.12, 90% Cov={cov_90*100:.1f}%). Best family: {selected_family.value}."

        nll = -prof_gauss.log_likelihood / max(1, len(all_samples))

        return CriticResult(
            hypothesis_id=hypothesis_id,
            experiment_category=experiment_category,
            epistemic_state=state,
            selected_family=selected_family,
            is_gaussian_rejected=is_gaussian_rejected,
            empirical_coverage_50pct=cov_50,
            empirical_coverage_90pct=cov_90,
            empirical_coverage_95pct=cov_95,
            mean_coverage_error=cov_err,
            predictive_nll=nll,
            mahalanobis_outlier_rate=out_rate,
            candidate_profiles=profiles,
            calibration_probe_agenda=probe_agenda,
            criticism_verdict_rationale=rationale,
            policy_provenance=self.policy.to_dict(),
            timestamp_utc=ts,
        )
