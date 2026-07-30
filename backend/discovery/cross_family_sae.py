import numpy as np
from typing import Dict, Any, List

class CrossFamilySAEAlignment:
    """
    Campaign 4: Cross-family SAE Alignment.
    Determines whether features in different model families (GPT-2, Gemma, Llama)
    encode the same semantic concepts.
    """
    
    @staticmethod
    def align_features(features_a: np.ndarray, features_b: np.ndarray) -> Dict[str, Any]:
        """
        Uses CKA or Top-K activation correlation to find best-matching features.
        features_a/b: [n_samples, d_sae]
        """
        # Compute correlation matrix between features of two models
        # (Assuming samples are aligned/same dataset)
        corr_matrix = np.corrcoef(features_a.T, features_b.T)[:features_a.shape[1], features_a.shape[1]:]
        
        # For each feature in A, find best match in B
        matches = []
        for i in range(features_a.shape[1]):
            best_idx = np.argmax(corr_matrix[i])
            score = corr_matrix[i, best_idx]
            if score > 0.7: # High threshold for 'alignment'
                matches.append({
                    "feature_a_idx": i,
                    "feature_b_idx": int(best_idx),
                    "alignment_score": float(score)
                })
                
        return {
            "num_aligned_features": len(matches),
            "top_matches": sorted(matches, key=lambda x: x["alignment_score"], reverse=True)[:10]
        }

    def run_alignment_study(self, sae_activations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        sae_activations: { "gpt2": acts, "gemma": acts, "llama": acts }
        """
        models = list(sae_activations.keys())
        results = {}
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                pair = f"{models[i]}_vs_{models[j]}"
                results[pair] = self.align_features(sae_activations[models[i]], sae_activations[models[j]])
                
        return results
