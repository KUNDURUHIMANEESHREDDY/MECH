from typing import Dict, Any
from .diagnostics import DistributionDiagnostics
from .power_analysis import PowerAnalysis

class StatisticalRecommender:
    """
    Intelligent statistician that recommends actions based on data.
    """

    @staticmethod
    def recommend(
        group_a: list, 
        group_b: list, 
        effect_size: float, 
        current_p_value: float,
        target_power: float = 0.8,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        recommendations = []
        
        # 1. Distribution & Test
        import numpy as np
        data_a = np.array(group_a)
        data_b = np.array(group_b)
        diag_a = DistributionDiagnostics.test_normality(data_a)
        diag_b = DistributionDiagnostics.test_normality(data_b)
        
        if not (diag_a["is_normal"] and diag_b["is_normal"]):
            recommendations.append("Data is non-normal. Recommend using Permutation test or Mann-Whitney U.")
        else:
             recommendations.append("Data appears normal. Welch's t-test is appropriate.")
             
        # 2. Power Analysis
        current_n = min(len(group_a), len(group_b))
        power_res = PowerAnalysis.analyze(effect_size, current_n, target_power)
        obs_power = power_res["observed_power"]
        
        if obs_power < target_power:
            req_n = power_res["required_n"]
            additional_n = max(0, req_n - current_n)
            recommendations.append(f"Power is low ({obs_power:.2f}). Recommend collecting {additional_n} more samples per group.")
        else:
            recommendations.append(f"Power is sufficient ({obs_power:.2f}).")
            
        # 3. Multiple Testing
        # Assuming context involves multiple features, recommend FDR
        recommendations.append("If testing multiple features, recommend Benjamini-Hochberg (FDR) correction.")
        
        return {
            "diagnostics": {"group_a_normal": diag_a["is_normal"], "group_b_normal": diag_b["is_normal"]},
            "power": obs_power,
            "required_n_total": power_res["required_n"],
            "recommendations": recommendations
        }
