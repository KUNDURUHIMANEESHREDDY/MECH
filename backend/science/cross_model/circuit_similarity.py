import numpy as np
from typing import Dict, List, Any

class CircuitSimilarity:
    """
    Compares circuit structures (e.g., IOI, Induction) across different models.
    Uses Graph Edit Distance or Layer-wise correlation of head impact.
    """
    
    @staticmethod
    def compute_head_impact_correlation(model_a_impact: np.ndarray, model_b_impact: np.ndarray) -> float:
        """
        Computes correlation between head impact maps (e.g., 144 heads for GPT-2 Small).
        If models have different sizes, uses interpolated or rank-based comparison.
        """
        # Simple Pearson correlation for same-sized maps
        if model_a_impact.shape == model_b_impact.shape:
            return float(np.corrcoef(model_a_impact.flatten(), model_b_impact.flatten())[0, 1])
        
        # For different sizes, use rank-based top-k overlap (Jaccard)
        top_k = min(len(model_a_impact), len(model_b_impact)) // 10
        top_a = set(np.argsort(model_a_impact.flatten())[-top_k:])
        top_b = set(np.argsort(model_b_impact.flatten())[-top_k:])
        
        intersection = len(top_a.intersection(top_b))
        union = len(top_a.union(top_b))
        return intersection / union if union > 0 else 0.0

    @classmethod
    def compare_circuits(cls, circuits: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        circuits: { "gpt2": impact_map, "gemma": impact_map, ... }
        """
        models = list(circuits.keys())
        similarity_matrix = np.zeros((len(models), len(models)))
        
        for i in range(len(models)):
            for j in range(len(models)):
                similarity_matrix[i, j] = cls.compute_head_impact_correlation(
                    circuits[models[i]], circuits[models[j]]
                )
                
        return {
            "models": models,
            "similarity_matrix": similarity_matrix.tolist(),
            "mean_similarity": float(np.mean(similarity_matrix[np.triu_indices(len(models), k=1)]))
        }
