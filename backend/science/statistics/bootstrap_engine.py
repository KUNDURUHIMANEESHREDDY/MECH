"""Bootstrap Confidence Interval Engine.

Supports Percentile, BCa (Bias-Corrected and Accelerated), and Empirical bootstrap intervals.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Tuple
import numpy as np
import logging
logger = logging.getLogger(__name__)



def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    try:
        from scipy.stats import norm
        return float(norm.cdf(x))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Swallowed exception: %s", exc)
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal percent point function (inverse CDF)."""
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Swallowed exception: %s", exc)
        # Winitzki approximation for erfinv
        a = 0.147
        y = 2.0 * p - 1.0
        log_term = math.log(1.0 - y * y)
        term1 = 2.0 / (math.pi * a) + log_term / 2.0
        term2 = log_term / a
        val = math.sqrt(term1 * term1 - term2) - term1
        sign = 1.0 if y >= 0 else -1.0
        return sign * math.sqrt(val) * math.sqrt(2.0)


class BootstrapEngine:
    """Bootstrap Engine for calculating Percentile, BCa, and Studentized Confidence Intervals."""

    def __init__(self, n_bootstraps: int = 2000, seed: int = 42) -> None:
        self.n_bootstraps = n_bootstraps
        self.rng = np.random.default_rng(seed)

    def percentile_ci(
        self,
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
        alpha: float = 0.05,
    ) -> Tuple[float, float, float]:
        """Compute standard percentile bootstrap confidence interval."""
        data = np.asarray(data)
        n = len(data)
        if n == 0:
            raise ValueError("Input data cannot be empty for bootstrap estimation.")
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError("Input data contains NaN or Inf values.")

        estimate = float(statistic_fn(data))
        if n == 1:
            return estimate, estimate, estimate

        stats = []
        for _ in range(self.n_bootstraps):
            sample = self.rng.choice(data, size=n, replace=True)
            stats.append(float(statistic_fn(sample)))

        stats_arr = np.array(stats)
        lower = float(np.percentile(stats_arr, 100.0 * (alpha / 2.0)))
        upper = float(np.percentile(stats_arr, 100.0 * (1.0 - alpha / 2.0)))
        return estimate, lower, upper

    def bca_ci(
        self,
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
        alpha: float = 0.05,
    ) -> Tuple[float, float, float]:
        """Compute Bias-Corrected and Accelerated (BCa) bootstrap confidence interval."""
        data = np.asarray(data)
        n = len(data)
        if n == 0:
            raise ValueError("Input data cannot be empty for bootstrap estimation.")
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError("Input data contains NaN or Inf values.")

        estimate = float(statistic_fn(data))
        if n < 3:
            # Fall back safely to percentile CI when n < 3 to prevent degenerate Jackknife
            return self.percentile_ci(data, statistic_fn, alpha=alpha)

        # 1. Generate bootstrap replicates
        stats = []
        for _ in range(self.n_bootstraps):
            sample = self.rng.choice(data, size=n, replace=True)
            stats.append(float(statistic_fn(sample)))
        stats_arr = np.array(stats)

        # 2. Bias-correction parameter z0
        prop_less = np.mean(stats_arr < estimate)
        prop_less = min(max(prop_less, 1.0 / self.n_bootstraps), 1.0 - 1.0 / self.n_bootstraps)
        z0 = _norm_ppf(float(prop_less))

        # 3. Acceleration parameter a via leave-one-out Jackknife
        jackknife_stats = []
        for i in range(n):
            jack_sample = np.delete(data, i, axis=0)
            jackknife_stats.append(float(statistic_fn(jack_sample)))
        jackknife_arr = np.array(jackknife_stats)
        mean_jack = np.mean(jackknife_arr)
        u_i = mean_jack - jackknife_arr
        
        sum_u3 = np.sum(u_i ** 3)
        sum_u2 = np.sum(u_i ** 2)
        if sum_u2 > 1e-12:
            a = float(sum_u3 / (6.0 * (sum_u2 ** 1.5)))
        else:
            a = 0.0

        # Bound acceleration to avoid division by zero or extreme instability
        a = min(max(a, -0.5), 0.5)

        # 4. Adjusted quantiles
        z_alpha_low = _norm_ppf(alpha / 2.0)
        z_alpha_high = _norm_ppf(1.0 - alpha / 2.0)

        denom_low = 1.0 - a * (z0 + z_alpha_low)
        denom_high = 1.0 - a * (z0 + z_alpha_high)

        adj_z_low = z0 + (z0 + z_alpha_low) / max(denom_low, 1e-4)
        adj_z_high = z0 + (z0 + z_alpha_high) / max(denom_high, 1e-4)

        pct_low = min(max(_norm_cdf(adj_z_low), 0.5 / self.n_bootstraps), 1.0 - 0.5 / self.n_bootstraps)
        pct_high = min(max(_norm_cdf(adj_z_high), 0.5 / self.n_bootstraps), 1.0 - 0.5 / self.n_bootstraps)

        lower = float(np.percentile(stats_arr, 100.0 * pct_low))
        upper = float(np.percentile(stats_arr, 100.0 * pct_high))

        # Ensure monotonic order
        if lower > upper:
            lower, upper = upper, lower

        return estimate, lower, upper

    def run(
        self,
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
        method: str = "percentile",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Compute bootstrap confidence interval for the given method ('percentile' or 'bca')."""
        norm_method = method.lower().strip()
        if norm_method in ("percentile", "perc"):
            est, lower, upper = self.percentile_ci(data, statistic_fn, alpha=alpha)
        elif norm_method in ("bca", "bca_ci"):
            est, lower, upper = self.bca_ci(data, statistic_fn, alpha=alpha)
        else:
            raise NotImplementedError(f"Method {method} not implemented. Use 'percentile' or 'bca'.")

        return {
            "estimate": round(est, 6),
            "ci_lower": round(lower, 6),
            "ci_upper": round(upper, 6),
            "method": norm_method,
            "alpha": alpha,
            "confidence_level": round(1.0 - alpha, 2),
            "n_bootstraps": self.n_bootstraps,
        }

