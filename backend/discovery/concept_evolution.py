import numpy as np
from typing import Dict, Any, List
from ..science.statistics.statistical_validator import StatisticalValidator

class ConceptEvolution:
    """
    Discovery Project 5 & Campaign 2: Concept evolution.
    Tracks how concepts (Syntax, Names, Entities, Cities, Countries, Reasoning)
    develop across layers and across different models.
    """
    def __init__(self, experiment_id: str):
        self.validator = StatisticalValidator(experiment_id)

    def track_concept(self, layer_activations: Dict[int, np.ndarray], concept_mask: np.ndarray, concept_name: str) -> Dict[str, Any]:
        """
        Calculates separation between concept-active and concept-inactive samples at each layer for a specific concept.
        """
        evolution_trace = []
        
        for layer in sorted(layer_activations.keys()):
            act = layer_activations[layer]
            # Flatten or handle multi-dimensional activations (e.g., [batch, seq, d_model])
            if act.ndim > 1:
                # Taking max or mean activation for the concept
                pos = np.max(act[concept_mask], axis=-1)
                neg = np.max(act[~concept_mask], axis=-1)
            else:
                pos = act[concept_mask]
                neg = act[~concept_mask]
            
            stats = self.validator.compare_groups(pos, neg, f"Layer_{layer}_{concept_name}_Separation")
            
            evolution_trace.append({
                "layer": layer,
                "separation": stats["effect_sizes"]["cohens_d"],
                "p_value": stats["p_value_raw"],
                "rigor": stats["quality_score"]["score"]
            })
            
        return {
            "concept": concept_name,
            "evolution_trace": evolution_trace,
            "peak_layer": max(evolution_trace, key=lambda x: x["separation"])["layer"],
            "emergence_layer": next((x["layer"] for x in evolution_trace if x["p_value"] < 0.001), None)
        }

    def run_campaign(self, model_name: str, activations: Dict[int, np.ndarray], masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Runs the full Concept Evolution campaign for a model.
        Concepts: Syntax, Names, Entities, Cities, Countries, Reasoning.
        """
        results = {}
        for concept_name, mask in masks.items():
            results[concept_name] = self.track_concept(activations, mask, concept_name)
            
        return {
            "model": model_name,
            "concepts": results,
            "timestamp": self.validator.trace.trace[0]["timestamp"] if self.validator.trace.trace else None
        }
