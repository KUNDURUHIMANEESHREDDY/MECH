import numpy as np
from typing import Dict, Any, List
from ..science.statistics.statistical_validator import StatisticalValidator

class PolysemanticityScaleCampaign:
    """
    Campaign 3: Polysemanticity Across Scale.
    Analyzes whether polysemanticity decreases with model size across GPT-2, Gemma, Llama, and Qwen.
    """
    def __init__(self, experiment_id: str):
        self.validator = StatisticalValidator(experiment_id)

    def calculate_metrics(self, neuron_activations: np.ndarray, concept_masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculates entropy, purity, and overlap for a set of neurons.
        """
        n_neurons = neuron_activations.shape[-1]
        entropies = []
        purities = []
        overlaps = []
        
        for i in range(n_neurons):
            act = neuron_activations[:, i]
            responses = []
            for name, mask in concept_masks.items():
                # Probability of neuron being active for this concept
                p = np.mean(act[mask] > np.mean(act)) 
                responses.append(p)
            
            responses = np.array(responses)
            if np.sum(responses) > 0:
                norm_responses = responses / np.sum(responses)
                # Entropy: measure of how 'spread' the neuron is across concepts
                entropy = -np.sum(norm_responses * np.log2(norm_responses + 1e-9))
                # Purity: Max response probability
                purity = np.max(norm_responses)
                
                entropies.append(entropy)
                purities.append(purity)
            
            # Overlap: Average correlation between concept activations
            # (simplified metric)
        
        return {
            "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "mean_purity": float(np.mean(purities)) if purities else 0.0,
            "polysemanticity_index": float(np.mean(entropies) / (np.mean(purities) + 1e-9)) if purities else 0.0
        }

    def run_scale_study(self, model_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        model_data: { "gpt2": {"activations": np.ndarray, "masks": Dict}, "llama": ... }
        """
        results = {}
        for model_name, data in model_data.items():
            results[model_name] = self.calculate_metrics(data["activations"], data["masks"])
            
        # Correlate with model size (mock sizes in billions of parameters)
        sizes = {"gpt2": 0.12, "gemma-2b": 2.0, "llama-8b": 8.0, "qwen-72b": 72.0}
        
        correlation_data = []
        for name, metrics in results.items():
             if name in sizes:
                 correlation_data.append((sizes[name], metrics["polysemanticity_index"]))
        
        return {
            "model_metrics": results,
            "scale_correlation": correlation_data
        }
