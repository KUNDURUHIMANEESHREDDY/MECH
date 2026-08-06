"""
Neuron experiment definitions.

Pre-built experiment workflows for neuron analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .config import ExperimentConfig
from backend.analysis.experiment_runner import ExperimentRunner


class NeuronExperiment:
    """Pre-built neuron analysis experiments.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration for the experiment.
    """

    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()

    def run_single_neuron_analysis(self, layer_index: int = 0, neuron_index: int = 0) -> Dict[str, Any]:
        """Analyze a single neuron across all tokens.

        Parameters
        ----------
        layer_index : int
            Layer to analyze.
        neuron_index : int
            Neuron to analyze.

        Returns
        -------
        dict
            Analysis results including activation, statistics, and
            top activating tokens.
        """
        from backend.interpretability.data_generator import MockModelData
        from backend.interpretability.neuron_inspector import NeuronInspector

        model = MockModelData(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            seq_len=self.config.seq_len,
            vocab_size=self.config.vocab_size,
            seed=self.config.seed,
        )
        inspector = NeuronInspector(model=model)
        neuron_data = inspector.inspect(
            layer_index, neuron_index, self.config.token_index
        )
        return neuron_data.model_dump()

    def run_layer_neuron_analysis(self, layer_index: int = 0) -> Dict[str, Any]:
        """Analyze top neurons in a single layer.

        Parameters
        ----------
        layer_index : int
            Layer to analyze.

        Returns
        -------
        dict
            Analysis results including top neurons and summary statistics.
        """
        runner = ExperimentRunner()
        result = runner.run_neuron_experiment(
            layer_index=layer_index,
            top_k=self.config.top_k_neurons,
            token_index=self.config.token_index,
        )
        return result.to_dict()

    def run_all_layers_neuron_analysis(self) -> List[Dict[str, Any]]:
        """Analyze neurons across all configured layers.

        Returns
        -------
        list of dict
            Analysis results for each layer.
        """
        results = []
        for layer_idx in self.config.get_layers():
            results.append(self.run_layer_neuron_analysis(layer_idx))
        return results

    def run_sparsity_analysis(self) -> Dict[str, Any]:
        """Analyze neuron activation sparsity across all layers.

        Returns
        -------
        dict
            Sparsity statistics per layer and overall trends.
        """
        from backend.interpretability.data_generator import MockModelData
        from backend.interpretability.statistics import StatisticsComputer
        import numpy as np

        model = MockModelData(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            seq_len=self.config.seq_len,
            vocab_size=self.config.vocab_size,
            seed=self.config.seed,
        )
        stats_computer = StatisticsComputer()

        layer_sparsities = []
        for layer_idx in self.config.get_layers():
            layer_name = model.get_layer_name(layer_idx)
            activations = model.get_activation(layer_name)
            stats = stats_computer.compute(activations)
            layer_sparsities.append(
                {
                    "layer": layer_name,
                    "layer_index": layer_idx,
                    "sparsity": stats.sparsity,
                    "mean": stats.mean,
                    "max": stats.max,
                    "variance": stats.variance,
                }
            )

        sparsities = [s["sparsity"] for s in layer_sparsities]
        return {
            "layer_sparsities": layer_sparsities,
            "overall_mean_sparsity": float(np.mean(sparsities)),
            "overall_max_sparsity": float(max(sparsities)),
            "overall_min_sparsity": float(min(sparsities)),
            "sparsity_trend": "increasing" if np.polyfit(range(len(sparsities)), sparsities, 1)[0] > 0 else "decreasing",
        }
