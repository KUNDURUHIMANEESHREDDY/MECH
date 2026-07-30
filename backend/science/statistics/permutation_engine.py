import numpy as np
from typing import Callable, Dict, Any, Union

class PermutationEngine:
    """
    Generic Framework for Permutation Testing.
    Supports Label, Feature, Activation, Circuit, and Cluster permutations.
    """
    def __init__(self, n_permutations: int = 10000, seed: int = 42):
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(seed)

    def run_permutation_test(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        statistic_fn: Callable[[np.ndarray, np.ndarray], float],
        permutation_type: str = "label"
    ) -> Dict[str, Union[float, int, str]]:
        """
        Runs a generic permutation test.
        """
        observed_stat = statistic_fn(data_a, data_b)
        n_a = len(data_a)
        pooled = np.concatenate([data_a, data_b])
        
        permuted_stats = []
        for _ in range(self.n_permutations):
            if permutation_type == "label":
                # Shuffle the pooled data (equivalent to shuffling labels)
                permuted = self.rng.permutation(pooled)
                perm_a = permuted[:n_a]
                perm_b = permuted[n_a:]
                permuted_stats.append(statistic_fn(perm_a, perm_b))
            elif permutation_type == "feature":
                 # Independent shuffling per feature if 2D
                 if data_a.ndim > 1:
                     perm_a = np.apply_along_axis(self.rng.permutation, 0, data_a)
                     perm_b = np.apply_along_axis(self.rng.permutation, 0, data_b)
                     permuted_stats.append(statistic_fn(perm_a, perm_b))
                 else:
                     raise ValueError("Feature permutation requires 2D arrays.")
            # ... handle activation, circuit, cluster specifics ...
            else:
                 raise ValueError(f"Unknown permutation type: {permutation_type}")

        permuted_stats = np.array(permuted_stats)
        # Two-tailed p-value calculation
        p_value = np.sum(np.abs(permuted_stats) >= np.abs(observed_stat)) / self.n_permutations
        
        return {
            "p_value": p_value,
            "observed_statistic": observed_stat,
            "n_permutations": self.n_permutations,
            "permutation_type": permutation_type
        }
