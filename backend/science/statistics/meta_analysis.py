import numpy as np
from typing import List, Dict, Any, Tuple

class MetaAnalysis:
    """
    Combines results from multiple independent studies (or runs).
    """

    @staticmethod
    def fixed_effects(effect_sizes: np.ndarray, variances: np.ndarray) -> Tuple[float, float, float]:
        """
        Inverse-variance weighting for fixed effects model.
        """
        weights = 1.0 / variances
        sum_weights = np.sum(weights)
        
        pooled_effect = np.sum(weights * effect_sizes) / sum_weights
        pooled_variance = 1.0 / sum_weights
        
        # 95% CI
        z = 1.96
        ci_lower = pooled_effect - z * np.sqrt(pooled_variance)
        ci_upper = pooled_effect + z * np.sqrt(pooled_variance)
        
        return float(pooled_effect), float(ci_lower), float(ci_upper)

    @staticmethod
    def heterogeneity(effect_sizes: np.ndarray, variances: np.ndarray) -> Dict[str, float]:
        """
        Calculates Cochran's Q and I^2 statistics.
        """
        k = len(effect_sizes)
        if k <= 1:
            return {"Q": 0.0, "I2": 0.0, "tau2": 0.0}
            
        weights = 1.0 / variances
        sum_weights = np.sum(weights)
        pooled_effect = np.sum(weights * effect_sizes) / sum_weights
        
        # Cochran's Q
        q = np.sum(weights * (effect_sizes - pooled_effect)**2)
        
        # I^2 (percentage of variation due to heterogeneity)
        if q > k - 1:
            i2 = ((q - (k - 1)) / q) * 100
        else:
            i2 = 0.0
            
        # Tau^2 (between-study variance, DerSimonian-Laird estimator)
        c = sum_weights - np.sum(weights**2) / sum_weights
        if q > k - 1 and c > 0:
            tau2 = (q - (k - 1)) / c
        else:
            tau2 = 0.0
            
        return {"Q": float(q), "I2": float(i2), "tau2": float(tau2)}

    @staticmethod
    def random_effects(effect_sizes: np.ndarray, variances: np.ndarray) -> Tuple[float, float, float]:
        """
        Random effects model using DerSimonian-Laird estimator for tau^2.
        """
        het = MetaAnalysis.heterogeneity(effect_sizes, variances)
        tau2 = het["tau2"]
        
        # Adjusted weights
        weights_star = 1.0 / (variances + tau2)
        sum_weights_star = np.sum(weights_star)
        
        pooled_effect = np.sum(weights_star * effect_sizes) / sum_weights_star
        pooled_variance = 1.0 / sum_weights_star
        
        # 95% CI
        z = 1.96
        ci_lower = pooled_effect - z * np.sqrt(pooled_variance)
        ci_upper = pooled_effect + z * np.sqrt(pooled_variance)
        
        return float(pooled_effect), float(ci_lower), float(ci_upper)

    @classmethod
    def analyze(cls, effect_sizes: List[float], variances: List[float]) -> Dict[str, Any]:
        es_arr = np.array(effect_sizes)
        var_arr = np.array(variances)
        
        fe_eff, fe_low, fe_up = cls.fixed_effects(es_arr, var_arr)
        re_eff, re_low, re_up = cls.random_effects(es_arr, var_arr)
        het = cls.heterogeneity(es_arr, var_arr)
        
        return {
            "fixed_effects": {"effect_size": fe_eff, "ci_lower": fe_low, "ci_upper": fe_up},
            "random_effects": {"effect_size": re_eff, "ci_lower": re_low, "ci_upper": re_up},
            "heterogeneity": het
        }
