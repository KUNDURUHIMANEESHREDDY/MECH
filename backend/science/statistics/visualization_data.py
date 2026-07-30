import numpy as np
from typing import Dict, Any, List

class VisualizationDataGenerator:
    """
    Generates structured JSON data for frontend visualization.
    """

    @staticmethod
    def forest_plot(studies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Expects studies format: [{"name": str, "effect_size": float, "ci_lower": float, "ci_upper": float, "weight": float}]
        """
        return {"plot_type": "forest", "data": studies}

    @staticmethod
    def volcano_plot(features: List[str], effect_sizes: List[float], p_values: List[float]) -> Dict[str, Any]:
        """
        Generates data for a volcano plot (-log10(p) vs effect size).
        """
        # Add small epsilon to prevent log10(0)
        eps = 1e-300
        neg_log10_p = [-np.log10(p + eps) for p in p_values]
        
        data = [
            {"feature": f, "effect_size": es, "neg_log10_p": nlp, "p_value": p}
            for f, es, nlp, p in zip(features, effect_sizes, neg_log10_p, p_values)
        ]
        return {"plot_type": "volcano", "data": data}

    @staticmethod
    def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        bins = np.linspace(0., 1., n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        curve_data = []
        for i in range(n_bins):
            mask = binids == i
            if np.sum(mask) > 0:
                prob_pred = np.mean(y_prob[mask])
                prob_true = np.mean(y_true[mask])
                count = int(np.sum(mask))
                curve_data.append({"bin_mid": (bins[i] + bins[i+1])/2, "prob_pred": float(prob_pred), "prob_true": float(prob_true), "count": count})
                
        return {"plot_type": "calibration", "data": curve_data}
