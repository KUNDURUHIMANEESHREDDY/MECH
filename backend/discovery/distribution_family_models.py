"""Multi-Family Distribution Models for MECH Outcome Modeling.

Fits and evaluates 4 competing distribution families per (H_i, Category):
1. Gaussian: N(mu, Sigma) [Baseline parametric]
2. Student-t: t_nu(mu, Sigma) [Heavy-tailed robustness]
3. 2-component GMM: [Multimodal / polysemantic subpopulations]
4. Empirical KDE: [Nonparametric kernel density benchmark]
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DistributionFamilyType(str, Enum):
    GAUSSIAN = "GAUSSIAN"
    STUDENT_T = "STUDENT_T"
    GAUSSIAN_MIXTURE_GMM = "GAUSSIAN_MIXTURE_GMM"
    EMPIRICAL_KDE = "EMPIRICAL_KDE"


@dataclass
class FittedDistributionProfile:
    """Statistical profile of a fitted distribution family."""
    family_type: DistributionFamilyType
    parameters: Dict[str, Any]
    log_likelihood: float
    aic: float
    bic: float
    sample_count: int
    skewness: float
    excess_kurtosis: float
    last_fitted_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_type": self.family_type.value,
            "parameters": self.parameters,
            "log_likelihood": round(self.log_likelihood, 4),
            "aic": round(self.aic, 4),
            "bic": round(self.bic, 4),
            "sample_count": self.sample_count,
            "skewness": round(self.skewness, 4),
            "excess_kurtosis": round(self.excess_kurtosis, 4),
            "last_fitted_utc": self.last_fitted_utc,
        }


class MultiFamilyDistributionFitter:
    """Fits Gaussian, Student-t, GMM, and KDE models to empirical causal observations."""

    def fit_gaussian(self, samples: List[float]) -> FittedDistributionProfile:
        """Fits Gaussian N(mu, sigma^2)."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        n = len(samples)
        if n == 0:
            return self._empty_profile(DistributionFamilyType.GAUSSIAN, ts)

        mu = sum(samples) / n
        var = max(1e-6, sum((x - mu) ** 2 for x in samples) / max(1, n - 1))
        std = math.sqrt(var)

        # Log-Likelihood
        ll = sum(-0.5 * math.log(2.0 * math.pi * var) - ((x - mu) ** 2) / (2.0 * var) for x in samples)
        k = 2  # mu, var
        aic = 2 * k - 2 * ll
        bic = k * math.log(max(1, n)) - 2 * ll

        # Higher moments (skewness, kurtosis)
        skew = (sum(((x - mu) / std) ** 3 for x in samples) / n) if std > 1e-5 else 0.0
        kurt = (sum(((x - mu) / std) ** 4 for x in samples) / n - 3.0) if std > 1e-5 else 0.0

        return FittedDistributionProfile(
            family_type=DistributionFamilyType.GAUSSIAN,
            parameters={"mu": round(mu, 4), "variance": round(var, 6), "std": round(std, 4)},
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            sample_count=n,
            skewness=skew,
            excess_kurtosis=kurt,
            last_fitted_utc=ts,
        )

    def fit_student_t(self, samples: List[float], fixed_nu: float = 4.0) -> FittedDistributionProfile:
        """Fits Student-t distribution with heavy tail parameter nu."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        n = len(samples)
        if n == 0:
            return self._empty_profile(DistributionFamilyType.STUDENT_T, ts)

        mu = sum(samples) / n
        var = max(1e-6, sum((x - mu) ** 2 for x in samples) / max(1, n - 1))
        scale = math.sqrt(var * (fixed_nu - 2.0) / fixed_nu) if fixed_nu > 2.0 else math.sqrt(var)

        # Student-t Log-Likelihood
        gamma_term = math.lgamma((fixed_nu + 1.0) / 2.0) - math.lgamma(fixed_nu / 2.0)
        norm_const = gamma_term - 0.5 * math.log(fixed_nu * math.pi) - math.log(scale)
        ll = sum(norm_const - ((fixed_nu + 1.0) / 2.0) * math.log(1.0 + ((x - mu) / scale) ** 2 / fixed_nu) for x in samples)

        k = 3  # mu, scale, nu
        aic = 2 * k - 2 * ll
        bic = k * math.log(max(1, n)) - 2 * ll

        skew = 0.0  # symmetric t
        kurt = 6.0 / (fixed_nu - 4.0) if fixed_nu > 4.0 else 999.0

        return FittedDistributionProfile(
            family_type=DistributionFamilyType.STUDENT_T,
            parameters={"mu": round(mu, 4), "scale": round(scale, 4), "degrees_of_freedom_nu": fixed_nu},
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            sample_count=n,
            skewness=skew,
            excess_kurtosis=kurt,
            last_fitted_utc=ts,
        )

    def fit_2component_gmm(self, samples: List[float]) -> FittedDistributionProfile:
        """Fits 2-component Gaussian Mixture Model for multimodal distributions."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        n = len(samples)
        if n < 4:
            return self.fit_gaussian(samples)

        # Split by median
        sorted_s = sorted(samples)
        mid = n // 2
        s1, s2 = sorted_s[:mid], sorted_s[mid:]

        mu1 = sum(s1) / len(s1)
        mu2 = sum(s2) / len(s2)
        var1 = max(1e-6, sum((x - mu1)**2 for x in s1) / len(s1))
        var2 = max(1e-6, sum((x - mu2)**2 for x in s2) / len(s2))
        w1, w2 = 0.5, 0.5

        # Single EM iteration refinement
        resp1 = []
        for x in samples:
            d1 = (1.0 / math.sqrt(2*math.pi*var1)) * math.exp(-((x - mu1)**2)/(2*var1))
            d2 = (1.0 / math.sqrt(2*math.pi*var2)) * math.exp(-((x - mu2)**2)/(2*var2))
            r = (w1 * d1) / max(1e-12, (w1*d1 + w2*d2))
            resp1.append(r)

        w1 = max(0.05, min(0.95, sum(resp1) / n))
        w2 = 1.0 - w1

        # GMM log-likelihood
        ll = sum(
            math.log(max(1e-12,
                w1 * (1.0 / math.sqrt(2*math.pi*var1)) * math.exp(-((x - mu1)**2)/(2*var1)) +
                w2 * (1.0 / math.sqrt(2*math.pi*var2)) * math.exp(-((x - mu2)**2)/(2*var2))
            ))
            for x in samples
        )

        k = 5  # w1, mu1, var1, mu2, var2
        aic = 2 * k - 2 * ll
        bic = k * math.log(n) - 2 * ll

        return FittedDistributionProfile(
            family_type=DistributionFamilyType.GAUSSIAN_MIXTURE_GMM,
            parameters={
                "weight_1": round(w1, 3), "mu_1": round(mu1, 4), "var_1": round(var1, 6),
                "weight_2": round(w2, 3), "mu_2": round(mu2, 4), "var_2": round(var2, 6),
            },
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            sample_count=n,
            skewness=0.0,
            excess_kurtosis=0.0,
            last_fitted_utc=ts,
        )

    def fit_empirical_kde(self, samples: List[float]) -> FittedDistributionProfile:
        """Fits non-parametric Gaussian Kernel Density Estimator."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        n = len(samples)
        if n == 0:
            return self._empty_profile(DistributionFamilyType.EMPIRICAL_KDE, ts)

        std = math.sqrt(max(1e-6, sum((x - sum(samples)/n)**2 for x in samples) / max(1, n-1)))
        h = max(0.01, 1.06 * std * (n ** (-0.2)))

        ll = sum(
            math.log(max(1e-12, sum((1.0 / (math.sqrt(2*math.pi)*h)) * math.exp(-((x - xi)**2)/(2*h*h)) for xi in samples) / n))
            for x in samples
        )

        k = n  # Nonparametric
        aic = 2 * min(10, k) - 2 * ll
        bic = min(10, k) * math.log(n) - 2 * ll

        return FittedDistributionProfile(
            family_type=DistributionFamilyType.EMPIRICAL_KDE,
            parameters={"bandwidth_h": round(h, 4), "kernel_points_count": n},
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            sample_count=n,
            skewness=0.0,
            excess_kurtosis=0.0,
            last_fitted_utc=ts,
        )

    def _empty_profile(self, family: DistributionFamilyType, ts: str) -> FittedDistributionProfile:
        return FittedDistributionProfile(
            family_type=family,
            parameters={},
            log_likelihood=-999.0,
            aic=999.0,
            bic=999.0,
            sample_count=0,
            skewness=0.0,
            excess_kurtosis=0.0,
            last_fitted_utc=ts,
        )
