import numpy as np
from typing import Dict, Any

class EffectSizes:
    """
    Computes various effect sizes: Cohen's d, Hedges' g, Glass's delta, Cliff's delta.
    """

    @staticmethod
    def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
        if pooled_std == 0:
            return 0.0
        return float((np.mean(x) - np.mean(y)) / pooled_std)

    @staticmethod
    def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
        d = EffectSizes.cohens_d(x, y)
        n = len(x) + len(y)
        # Hedges' correction factor
        j = 1 - (3 / (4 * n - 9))
        return d * j

    @staticmethod
    def glass_delta(treatment: np.ndarray, control: np.ndarray) -> float:
        std_control = np.std(control, ddof=1)
        if std_control == 0:
            return 0.0
        return float((np.mean(treatment) - np.mean(control)) / std_control)
    
    @staticmethod
    def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
        """
        Non-parametric effect size.
        """
        nx, ny = len(x), len(y)
        dominations = 0
        for xi in x:
            for yi in y:
                if xi > yi:
                    dominations += 1
                elif xi < yi:
                    dominations -= 1
        return dominations / (nx * ny)

    @classmethod
    def compute_all(cls, group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
        return {
            "cohens_d": cls.cohens_d(group1, group2),
            "hedges_g": cls.hedges_g(group1, group2),
            "glass_delta": cls.glass_delta(group1, group2), # Assuming group2 is control
            "cliffs_delta": cls.cliffs_delta(group1, group2)
        }
