import numpy as np
from typing import List, Dict, Union

class HypothesisTesting:
    """
    Multiple Hypothesis Correction Engine.
    Implements Bonferroni, Holm-Bonferroni, Benjamini-Hochberg, and Benjamini-Yekutieli.
    """

    @staticmethod
    def bonferroni(p_values: List[float]) -> List[float]:
        n = len(p_values)
        return [min(1.0, p * n) for p in p_values]

    @staticmethod
    def holm_bonferroni(p_values: List[float]) -> List[float]:
        n = len(p_values)
        indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * n
        for rank, (idx, p) in enumerate(indexed_p, start=1):
            adjusted[idx] = min(1.0, p * (n - rank + 1))
        
        # Enforce monotonicity
        max_adj = 0.0
        for rank, (idx, p) in enumerate(indexed_p, start=1):
            max_adj = max(max_adj, adjusted[idx])
            adjusted[idx] = max_adj
        return adjusted

    @staticmethod
    def benjamini_hochberg(p_values: List[float]) -> List[float]:
        n = len(p_values)
        indexed_p = sorted(enumerate(p_values), key=lambda x: x[1], reverse=True)
        adjusted = [0.0] * n
        min_adj = 1.0
        for rank_rev, (idx, p) in enumerate(indexed_p):
            rank = n - rank_rev
            adj = p * n / rank
            min_adj = min(min_adj, adj)
            adjusted[idx] = min(1.0, min_adj)
        return adjusted

    @staticmethod
    def benjamini_yekutieli(p_values: List[float]) -> List[float]:
        n = len(p_values)
        if n == 0:
            return []
        c = sum(1.0 / i for i in range(1, n + 1))
        indexed_p = sorted(enumerate(p_values), key=lambda x: x[1], reverse=True)
        adjusted = [0.0] * n
        min_adj = 1.0
        for rank_rev, (idx, p) in enumerate(indexed_p):
            rank = n - rank_rev
            adj = p * n * c / rank
            min_adj = min(min_adj, adj)
            adjusted[idx] = min(1.0, min_adj)
        return adjusted

    @classmethod
    def apply_correction(
        cls, 
        p_values: List[float], 
        method: str = 'BH', 
        alpha: float = 0.05
    ) -> Dict[str, Union[List[float], List[bool], int, str, float]]:
        """
        Applies the selected multiple hypothesis testing correction.
        """
        if not p_values:
            return {
                'raw_p_values': [], 'adjusted_p_values': [],
                'rejected': [], 'num_rejected': 0, 'method': method, 'alpha': alpha
            }

        method_upper = method.upper()
        if method_upper in ['BH', 'BENJAMINI-HOCHBERG']:
            q_values = cls.benjamini_hochberg(p_values)
        elif method_upper in ['BY', 'BENJAMINI-YEKUTIELI']:
            q_values = cls.benjamini_yekutieli(p_values)
        elif method_upper == 'BONFERRONI':
            q_values = cls.bonferroni(p_values)
        elif method_upper in ['HOLM', 'HOLM-BONFERRONI']:
            q_values = cls.holm_bonferroni(p_values)
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        rejected = [q < alpha for q in q_values]
        return {
            'raw_p_values': p_values,
            'adjusted_p_values': q_values,
            'rejected': rejected,
            'num_rejected': sum(rejected),
            'method': method,
            'alpha': alpha
        }
