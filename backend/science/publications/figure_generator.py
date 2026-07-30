import json
import os
from typing import Dict, Any, List

class FigureGenerator:
    """
    Automates the generation of scientific figures for publications.
    Produces metadata and placeholders for plots like Attention Heatmaps, 
    Volcano plots, Forest plots, etc.
    """
    def __init__(self, experiment_id: str, output_dir: str = "backend/science/publications/figures"):
        self.experiment_id = experiment_id
        self.output_dir = os.path.join(output_dir, experiment_id)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_forest_plot(self, results: List[Dict[str, Any]], title: str = "Effect Size Comparison") -> str:
        """
        Generates data for a Forest plot showing effect sizes and CIs.
        """
        fig_id = f"fig_forest_{self.experiment_id}.json"
        data = {
            "type": "forest",
            "title": title,
            "entries": results
        }
        with open(os.path.join(self.output_dir, fig_id), 'w') as f:
            json.dump(data, f, indent=2)
        return fig_id

    def generate_volcano_plot(self, features: List[str], effects: List[float], p_values: List[float]) -> str:
        """
        Generates data for a Volcano plot (Effect Size vs -log10 p-value).
        """
        fig_id = f"fig_volcano_{self.experiment_id}.json"
        data = {
            "type": "volcano",
            "features": features,
            "effects": effects,
            "p_values": p_values
        }
        with open(os.path.join(self.output_dir, fig_id), 'w') as f:
            json.dump(data, f, indent=2)
        return fig_id

    def generate_calibration_curve(self, ece_data: Dict[str, Any]) -> str:
        fig_id = f"fig_calibration_{self.experiment_id}.json"
        with open(os.path.join(self.output_dir, fig_id), 'w') as f:
            json.dump(ece_data, f, indent=2)
        return fig_id

    def generate_attention_heatmap(self, layer: int, head: int, scores: Any) -> str:
        fig_id = f"fig_attn_L{layer}H{head}_{self.experiment_id}.json"
        # In real implementation, this would save a PNG/PDF
        data = {"type": "heatmap", "layer": layer, "head": head, "data": scores.tolist() if hasattr(scores, 'tolist') else scores}
        with open(os.path.join(self.output_dir, fig_id), 'w') as f:
            json.dump(data, f, indent=2)
        return fig_id
