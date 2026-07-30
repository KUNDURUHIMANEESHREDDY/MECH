import numpy as np
from typing import Dict, Any, List
from ..science.statistics.meta_analysis import MetaAnalysis
from ..science.statistics.statistical_validator import StatisticalValidator

class CrossModelUniversality:
    """
    Discovery Project 4: Cross-model universality.
    Finds shared representations across GPT-2, Gemma, Llama, and Qwen.
    """
    def __init__(self, experiment_id: str):
        self.validator = StatisticalValidator(experiment_id)

    def analyze_universality(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes results from multiple models for the same task and computes a meta-analysis.
        model_results: { "gpt2": stats_dict, "llama": stats_dict, ... }
        """
        effect_sizes = []
        variances = []
        model_names = []
        
        for model_name, res in model_results.items():
            d = res["effect_sizes"]["cohens_d"]
            # Simplified variance estimation for Cohen's d: (n1+n2)/(n1*n2) + d^2 / (2*(n1+n2))
            n = 100 # Assuming n=100 per group in original runs
            var = (2/n) + (d**2 / (4*n))
            
            effect_sizes.append(d)
            variances.append(var)
            model_names.append(model_name)
            
        # Meta-analysis to see if the effect is universal
        meta = MetaAnalysis.analyze(effect_sizes, variances)
        
        # Determine if 'Universality' is achieved
        # I^2 < 25% suggests low heterogeneity (strong universality)
        is_universal = meta["heterogeneity"]["I2"] < 25.0 and meta["random_effects"]["effect_size"] > 1.0
        
        return {
            "is_universal": is_universal,
            "meta_analysis": meta,
            "models_compared": model_names,
            "universality_score": 100.0 - meta["heterogeneity"]["I2"]
        }
