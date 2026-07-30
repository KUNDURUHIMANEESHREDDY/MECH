import numpy as np
from typing import Dict, List, Any

class RepresentationDrift:
    """
    Analyzes how representations drift within the same family (e.g., GPT-2 Small vs Medium)
    vs across different families (GPT-2 vs Gemma).
    """
    
    @staticmethod
    def calculate_drift(base_model_acts: np.ndarray, target_model_acts: np.ndarray) -> float:
        """
        Calculates Canonical Correlation Analysis (CCA) or CKA between activations.
        For simplicity, using Centered Kernel Alignment (CKA) approximation.
        """
        # Linear CKA approximation
        def linear_cka(X, Y):
            def gram_linear(x): return x @ x.T
            K = gram_linear(X)
            L = gram_linear(Y)
            
            # Centering matrix
            n = K.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            
            Kc = H @ K @ H
            Lc = H @ L @ H
            
            hsic = np.sum(Kc * Lc)
            denom = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
            return hsic / denom if denom > 0 else 0.0

        # Note: In reality, X and Y must have same number of samples
        return float(linear_cka(base_model_acts, target_model_acts))

    @classmethod
    def analyze_drift(cls, activations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        models = list(activations.keys())
        drift_matrix = np.zeros((len(models), len(models)))
        
        for i in range(len(models)):
            for j in range(len(models)):
                drift_matrix[i, j] = cls.calculate_drift(activations[models[i]], activations[models[j]])
                
        return {
            "drift_matrix": drift_matrix.tolist(),
            "models": models
        }
