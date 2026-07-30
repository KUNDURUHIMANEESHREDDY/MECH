import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple

class BayesianFrequentistComparison:
    """
    Computes Bayesian and Frequentist statistics side-by-side.
    """

    @staticmethod
    def _bayes_factor_t_test(t_stat: float, n1: int, n2: int, r: float = 0.707) -> float:
        """
        Approximates Bayes Factor (BF10) for a two-sample t-test using the JZS prior.
        r = 0.707 is the default scale for the Cauchy prior.
        """
        n = n1 * n2 / (n1 + n2)
        df = n1 + n2 - 2
        
        # This is a highly simplified approximation of the JZS Bayes Factor
        # In a real scenario, you'd use numerical integration
        # Here we use the BIC approximation for simplicity
        bic_null = 0 # Baseline
        bic_alt = - (t_stat**2) + np.log(n)
        
        # Approximate BF10
        bf10 = np.exp((bic_null - bic_alt) / 2)
        return float(bf10)

    @staticmethod
    def _bayesian_estimation(group_a: np.ndarray, group_b: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Simple Bayesian estimation of difference in means with conjugate priors.
        Assuming known (pooled) variance for simplicity of closed-form.
        """
        n1, n2 = len(group_a), len(group_b)
        mean_diff = np.mean(group_a) - np.mean(group_b)
        
        # Pooled variance
        var1, var2 = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))
        
        # Vague prior: Posterior is approximately centered at sample mean diff
        # with standard error se_diff
        
        # Credible Interval (Highest Density Interval)
        z = stats.norm.ppf(1 - alpha/2)
        cred_lower = mean_diff - z * se_diff
        cred_upper = mean_diff + z * se_diff
        
        # Posterior probability that difference > 0 (if mean_diff > 0)
        z_stat = mean_diff / se_diff
        post_prob = stats.norm.cdf(z_stat)
        
        return {
            "posterior_mean_diff": float(mean_diff),
            "credible_interval_lower": float(cred_lower),
            "credible_interval_upper": float(cred_upper),
            "posterior_prob_positive": float(post_prob)
        }

    @classmethod
    def analyze(cls, group_a: np.ndarray, group_b: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Runs both Frequentist and Bayesian analyses.
        """
        n1, n2 = len(group_a), len(group_b)
        
        # Frequentist (Welch's t-test)
        t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
        
        # Bayesian
        bayes_est = cls._bayesian_estimation(group_a, group_b, alpha)
        bf10 = cls._bayes_factor_t_test(t_stat, n1, n2)
        
        return {
            "frequentist": {
                "t_statistic": float(t_stat),
                "p_value": float(p_val)
            },
            "bayesian": {
                "bayes_factor_10": bf10,
                **bayes_est
            }
        }
