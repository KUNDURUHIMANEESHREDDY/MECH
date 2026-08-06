"""
Attention experiment definitions.

Pre-built experiment workflows for attention head analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .config import ExperimentConfig
from backend.analysis.experiment_runner import ExperimentRunner


class AttentionExperiment:
    """Pre-built attention analysis experiments.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration for the experiment.
    """

    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()

    def run_single_head_analysis(
        self, layer_index: int = 0, head_index: int = 0
    ) -> Dict[str, Any]:
        """Analyze a single attention head.

        Parameters
        ----------
        layer_index : int
            Layer to analyze.
        head_index : int
            Head to analyze.

        Returns
        -------
        dict
            Analysis results including matrix, importance, and statistics.
        """
        from backend.interpretability.data_generator import MockModelData
        from backend.interpretability.attention_inspector import AttentionInspector

        model = MockModelData(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            seq_len=self.config.seq_len,
            vocab_size=self.config.vocab_size,
            seed=self.config.seed,
        )
        inspector = AttentionInspector(model=model)
        attention_data = inspector.inspect(
            layer_index, head_index, self.config.token_index
        )
        return attention_data.model_dump()

    def run_layer_attention_analysis(self, layer_index: int = 0) -> Dict[str, Any]:
        """Analyze all attention heads in a single layer.

        Parameters
        ----------
        layer_index : int
            Layer to analyze.

        Returns
        -------
        dict
            Analysis results including importance scores and summary.
        """
        runner = ExperimentRunner()
        result = runner.run_attention_experiment(
            layer_index=layer_index,
            token_index=self.config.token_index,
        )
        return result.to_dict()

    def run_all_layers_attention_analysis(self) -> List[Dict[str, Any]]:
        """Analyze attention heads across all configured layers.

        Returns
        -------
        list of dict
            Analysis results for each layer.
        """
        results = []
        for layer_idx in self.config.get_layers():
            results.append(self.run_layer_attention_analysis(layer_idx))
        return results

    def run_importance_analysis(self) -> Dict[str, Any]:
        """Analyze attention head importance across all layers.

        Returns
        -------
        dict
            Importance statistics per layer and overall trends.
        """
        from backend.interpretability.data_generator import MockModelData
        from backend.interpretability.attention_inspector import AttentionInspector
        import numpy as np

        model = MockModelData(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            seq_len=self.config.seq_len,
            vocab_size=self.config.vocab_size,
            seed=self.config.seed,
        )
        inspector = AttentionInspector(model=model)

        layer_importance = []
        for layer_idx in self.config.get_layers():
            scores = inspector.get_layer_importance(layer_idx)
            layer_importance.append(
                {
                    "layer": model.get_layer_name(layer_idx),
                    "layer_index": layer_idx,
                    "importance_scores": scores,
                    "max_importance": float(max(scores)),
                    "mean_importance": float(np.mean(scores)),
                    "min_importance": float(min(scores)),
                    "top_head": int(np.argmax(scores)),
                }
            )

        all_scores = [
            s for layer in layer_importance for s in layer["importance_scores"]
        ]
        return {
            "layer_importance": layer_importance,
            "overall_max_importance": float(max(all_scores)),
            "overall_mean_importance": float(np.mean(all_scores)),
            "overall_min_importance": float(min(all_scores)),
        }
