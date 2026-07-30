import numpy as np
from typing import Dict, List, Any

class SemanticEvolutionComparison:
    """
    Compares the 'Evolution Trace' (layer-wise concept emergence) across models.
    """
    
    @staticmethod
    def compare_traces(model_traces: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        model_traces: { "gpt2": [{"layer": 0, "separation": 0.1}, ...], "llama": [...] }
        Since models have different depths, we normalize layers to [0, 1].
        """
        comparison = {}
        
        for model_name, trace in model_traces.items():
            layers = [t["layer"] for t in trace]
            max_l = max(layers)
            normalized_trace = [
                {"norm_layer": t["layer"] / max_l, "separation": t["separation"]}
                for t in trace
            ]
            
            # Find point of 50% max separation (Emergence Point)
            max_sep = max(t["separation"] for t in normalized_trace)
            emergence = next((t["norm_layer"] for t in normalized_trace if t["separation"] >= 0.5 * max_sep), 1.0)
            
            comparison[model_name] = {
                "emergence_norm": emergence,
                "peak_norm": next(t["norm_layer"] for t in normalized_trace if t["separation"] == max_sep)
            }
            
        return {
            "model_comparisons": comparison,
            "mean_emergence": float(np.mean([c["emergence_norm"] for c in comparison.values()]))
        }
