import numpy as np
from typing import Dict, Any, List
from ..statistics.meta_analysis import MetaAnalysis

class FeatureUniversality:
    """
    Measures how 'universal' specific SAE features or neurons are across model families.
    """
    
    @staticmethod
    def compute_universality_score(feature_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Expects a list of statistical results for the same concept from different models.
        Uses Meta-Analysis (I^2) to measure consistency.
        """
        effect_sizes = [res["effect_sizes"]["cohens_d"] for res in feature_stats]
        # Estimate variance
        variances = [(2/100) + (d**2 / 400) for d in effect_sizes]
        
        meta = MetaAnalysis.analyze(effect_sizes, variances)
        
        # High universality = High effect size + Low heterogeneity (I2)
        i2 = meta["heterogeneity"]["I2"]
        avg_d = meta["random_effects"]["effect_size"]
        
        universality = (avg_d / (1 + i2/100)) # Simple score heuristic
        
        return {
            "universality_score": float(universality),
            "heterogeneity_i2": i2,
            "pooled_effect_size": avg_d,
            "is_universal": i2 < 25.0 and avg_d > 1.0
        }
