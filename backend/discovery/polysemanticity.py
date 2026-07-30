import numpy as np
from typing import Dict, Any, List
from ..science.statistics.statistical_validator import StatisticalValidator

class PolysemanticityDiscovery:
    """
    Discovery Project 6: Polysemanticity.
    Identifies neurons/features that respond to multiple distinct concepts.
    """
    def __init__(self, experiment_id: str):
        self.validator = StatisticalValidator(experiment_id)

    def analyze_neuron(self, neuron_activations: np.ndarray, concept_masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Tests if a neuron is significantly active across multiple, non-overlapping domains.
        concept_masks: { "Programming": mask, "Math": mask, "French": mask }
        """
        concept_results = {}
        
        for name, mask in concept_masks.items():
            pos = neuron_activations[mask]
            neg = neuron_activations[~mask]
            
            stats = self.validator.compare_groups(pos, neg, f"Neuron_Response_{name}")
            concept_results[name] = stats
            
        # A neuron is polysemantic if it has significant effect sizes (> 1.0) for multiple concepts
        active_concepts = [
            name for name, res in concept_results.items() 
            if res["p_value_raw"] < 0.001 and res["effect_sizes"]["cohens_d"] > 1.0
        ]
        
        return {
            "active_concepts": active_concepts,
            "is_polysemantic": len(active_concepts) > 1,
            "concept_stats": concept_results,
            "polysemanticity_score": len(active_concepts) * 33.3 # Simple heuristic
        }
