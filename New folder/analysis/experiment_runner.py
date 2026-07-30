"""
Experiment runner for the Neuron Inspector.

Runs analysis experiments across layers, neurons, attention heads,
and residual streams, collecting results for reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from backend.interpretability.data_generator import MockModelData, get_default_model
from backend.interpretability.mock_runtime import MockRuntime
from backend.interpretability.repository import ActivationRepository
from backend.interpretability.neuron_inspector import NeuronInspector
from backend.interpretability.attention_inspector import AttentionInspector
from backend.interpretability.residual_inspector import ResidualInspector
from backend.interpretability.statistics import StatisticsComputer


@dataclass
class ExperimentResult:
    """Container for a single experiment's results."""

    name: str
    experiment_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "experiment_type": self.experiment_type,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "results": self.results,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ExperimentRunner:
    """Runs and collects results from interpretability experiments.

    Parameters
    ----------
    model : MockModelData, optional
        The model data to run experiments on.
    """

    def __init__(self, model: Optional[MockModelData] = None):
        self.model = model if model is not None else get_default_model()
        _runtime = MockRuntime(
            num_layers=self.model.num_layers,
            num_heads=self.model.num_heads,
            hidden_dim=self.model.hidden_dim,
            seq_len=self.model.seq_len,
            vocab_size=self.model.vocab_size,
            seed=self.model.seed,
        )
        _repository = ActivationRepository(_runtime)
        self.neuron_inspector = NeuronInspector(repository=_repository)
        self.attention_inspector = AttentionInspector(repository=_repository)
        self.residual_inspector = ResidualInspector(repository=_repository)
        self.stats_computer = StatisticsComputer()
        self.results: List[ExperimentResult] = []

    def run_neuron_experiment(
        self,
        layer_index: int,
        top_k: int = 10,
        token_index: int = 0,
    ) -> ExperimentResult:
        """Run a neuron analysis experiment.

        Inspects the top-k most active neurons in a layer and collects
        their statistics, activation histories, and top tokens.
        """
        neurons = self.neuron_inspector.inspect_layer(
            layer_index, top_k, token_index
        )

        results = {
            "layer": self.model.get_layer_name(layer_index),
            "layer_index": layer_index,
            "token_index": token_index,
            "neurons": [n.model_dump() for n in neurons],
            "summary": {
                "max_activation": max(n.activation for n in neurons),
                "mean_activation": float(
                    np.mean([n.activation for n in neurons])
                ),
                "mean_sparsity": float(
                    np.mean([n.statistics.sparsity for n in neurons])
                ),
                "mean_variance": float(
                    np.mean([n.statistics.variance for n in neurons])
                ),
            },
        }

        result = ExperimentResult(
            name=f"neuron_analysis_layer_{layer_index}",
            experiment_type="neuron",
            parameters={
                "layer_index": layer_index,
                "top_k": top_k,
                "token_index": token_index,
            },
            results=results,
        )
        self.results.append(result)
        return result

    def run_attention_experiment(
        self,
        layer_index: int,
        token_index: int = 0,
    ) -> ExperimentResult:
        """Run an attention analysis experiment.

        Inspects all attention heads in a layer and collects their
        importance scores, matrices, and statistics.
        """
        heads = self.attention_inspector.inspect_all_heads(
            layer_index, token_index
        )

        importance_scores = [h.importance for h in heads]
        results = {
            "layer": self.model.get_layer_name(layer_index),
            "layer_index": layer_index,
            "token_index": token_index,
            "num_heads": len(heads),
            "importance_scores": importance_scores,
            "top_heads": [
                {
                    "head": h.head,
                    "importance": h.importance,
                    "shape": h.shape,
                    "statistics": h.statistics.model_dump(),
                }
                for h in heads[:5]
            ],
            "summary": {
                "max_importance": max(importance_scores),
                "mean_importance": float(np.mean(importance_scores)),
                "min_importance": min(importance_scores),
                "std_importance": float(np.std(importance_scores)),
            },
        }

        result = ExperimentResult(
            name=f"attention_analysis_layer_{layer_index}",
            experiment_type="attention",
            parameters={
                "layer_index": layer_index,
                "token_index": token_index,
            },
            results=results,
        )
        self.results.append(result)
        return result

    def run_residual_experiment(
        self,
        token_index: int = 0,
    ) -> ExperimentResult:
        """Run a residual stream analysis experiment.

        Inspects the residual stream at all layers and collects
        norms, statistics, and contributions.
        """
        residuals = self.residual_inspector.inspect_all_layers(token_index)
        norms = [r.norm for r in residuals]

        results = {
            "token_index": token_index,
            "num_layers": len(residuals),
            "residuals": [r.model_dump() for r in residuals],
            "norms": norms,
            "contributions": [r.contribution for r in residuals],
            "summary": {
                "max_norm": max(norms),
                "mean_norm": float(np.mean(norms)),
                "min_norm": min(norms),
                "std_norm": float(np.std(norms)),
                "total_contribution": sum(
                    r.contribution for r in residuals
                ),
            },
        }

        result = ExperimentResult(
            name=f"residual_analysis_token_{token_index}",
            experiment_type="residual",
            parameters={
                "token_index": token_index,
            },
            results=results,
        )
        self.results.append(result)
        return result

    def run_full_experiment(
        self,
        layer_index: int = 0,
        top_k: int = 10,
        token_index: int = 0,
    ) -> Dict[str, ExperimentResult]:
        """Run all three experiment types and return combined results."""
        return {
            "neuron": self.run_neuron_experiment(
                layer_index, top_k, token_index
            ),
            "attention": self.run_attention_experiment(
                layer_index, token_index
            ),
            "residual": self.run_residual_experiment(token_index),
        }

    def run_cross_layer_analysis(self) -> ExperimentResult:
        """Run a cross-layer analysis of neuron statistics.

        Collects statistics for all layers and identifies patterns
        such as increasing/decreasing activation sparsity with depth.
        """
        layer_stats = []
        for layer_idx in range(self.model.num_layers):
            layer_name = self.model.get_layer_name(layer_idx)
            activations = self.model.get_activation(layer_name)
            stats = self.stats_computer.compute(activations)
            layer_stats.append(
                {
                    "layer": layer_name,
                    "layer_index": layer_idx,
                    "max": stats.max,
                    "mean": stats.mean,
                    "variance": stats.variance,
                    "sparsity": stats.sparsity,
                }
            )

        sparsities = [s["sparsity"] for s in layer_stats]
        means = [s["mean"] for s in layer_stats]

        results = {
            "layer_stats": layer_stats,
            "trends": {
                "sparsity_increasing": bool(
                    np.polyfit(range(len(sparsities)), sparsities, 1)[0] > 0
                ),
                "mean_increasing": bool(
                    np.polyfit(range(len(means)), means, 1)[0] > 0
                ),
                "max_sparsity_layer": layer_stats[
                    int(np.argmax(sparsities))
                ]["layer"],
                "min_sparsity_layer": layer_stats[
                    int(np.argmin(sparsities))
                ]["layer"],
            },
        }

        result = ExperimentResult(
            name="cross_layer_analysis",
            experiment_type="cross_layer",
            parameters={},
            results=results,
        )
        self.results.append(result)
        return result

    def save_results(self, path: str) -> None:
        """Save all experiment results to a JSON file."""
        data = {
            "experiment_count": len(self.results),
            "experiments": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def clear_results(self) -> None:
        """Clear all stored experiment results."""
        self.results.clear()
