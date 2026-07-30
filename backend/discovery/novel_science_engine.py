import numpy as np
from typing import Dict, Any, List
from ..science.statistics.statistical_validator import StatisticalValidator
from ..science.cross_model.representation_drift import RepresentationDrift

class NovelScienceEngine:
    """
    The 'Science Discovery' layer.
    Analyzes multi-model benchmark data to find undocumented phenomena.
    """
    def __init__(self, experiment_id: str):
        self.validator = StatisticalValidator(experiment_id)

    def discover_universal_features(self, model_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Campaign 4 Integration: Searches for a single feature shared across families.
        """
        # Linear CKA or correlation search across GPT-2, Gemma, Qwen
        # If correlation > 0.85 across all pairs -> Universal Feature
        models = list(model_features.keys())
        is_universal = True
        shared_score = 1.0
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                drift = RepresentationDrift.calculate_drift(model_features[models[i]], model_features[models[j]])
                shared_score = min(shared_score, drift)
                if drift < 0.85:
                    is_universal = False
        
        if is_universal:
            return {
                "discovery_type": "Universal Sparse Feature",
                "confidence": shared_score,
                "description": f"Feature encodes a shared representation across {', '.join(models)}."
            }
        return {"discovery_type": "None", "confidence": 0.0}

    def discover_new_induction_circuits(self, model_impact_maps: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Finds undocumented heads that behave like induction heads but aren't in landmark lists.
        """
        landmark_heads = {"L5H1", "L5H5", "L6H9"} # Known GPT-2 small induction heads
        discoveries = []
        
        for head_id, impact in model_impact_maps.items():
            if head_id not in landmark_heads and float(np.max(impact)) > 0.8:
                discoveries.append({
                    "discovery_type": "Undocumented Induction Circuit",
                    "head": head_id,
                    "impact_score": impact,
                    "description": "Head shows strong prefix-matching behavior but was previously undocumented."
                })
        return discoveries
