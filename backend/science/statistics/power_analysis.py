import numpy as np
from scipy import stats
from typing import Dict, Any

class PowerAnalysis:
    """
    Computes statistical power and required sample sizes.
    """
    
    @staticmethod
    def t_test_power(d: float, n: int, alpha: float = 0.05) -> float:
        """
        Approximates power for a two-sample t-test given Cohen's d and n (per group).
        Uses non-central t-distribution.
        """
        # Non-centrality parameter
        ncp = d * np.sqrt(n / 2)
        df = 2 * n - 2
        # Critical t-value
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        
        # Power is the probability of exceeding t_crit given ncp
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        return float(power)

    @staticmethod
    def required_n_for_power(d: float, target_power: float = 0.8, alpha: float = 0.05) -> float:
        """
        Finds the required sample size per group to achieve the target power.
        """
        if d == 0:
            return float('inf')
        
        n = 2
        while PowerAnalysis.t_test_power(d, n, alpha) < target_power:
            n += 1
            if n > 10000: # safety break
                break
        return n
    
    @classmethod
    def analyze(cls, d: float, current_n: int, target_power: float = 0.8) -> Dict[str, Any]:
        obs_power = cls.t_test_power(d, current_n)
        req_n = cls.required_n_for_power(d, target_power)
        
        return {
            "effect_size": d,
            "current_n": current_n,
            "observed_power": obs_power,
            "target_power": target_power,
            "required_n": req_n
        }
