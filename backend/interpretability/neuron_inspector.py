"""
Neuron Inspector.

Inspects individual neurons in a transformer model's layers.
Returns neuron ID, activation value, layer information, and
statistical summaries.

Architecture:

    Runtime

    ↓

    ActivationRepository

    ↓

    NeuronInspector

    ↓

    NeuronInspection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .repository import ActivationRepository
from .mock_runtime import get_default_runtime
from .models import NeuronInspection, Statistics
from .statistics import StatisticsComputer


class NeuronInspector:
    """Inspects neurons in a model's activation data.

    Parameters
    ----------
    repository : ActivationRepository, optional
        The activation repository to inspect. If not provided, a
        default repository wrapping a MockRuntime is used.
    stats_computer : StatisticsComputer, optional
        The statistics computer to use.
    """

    def __init__(
        self,
        repository: Optional[ActivationRepository] = None,
        stats_computer: Optional[StatisticsComputer] = None,
    ):
        self.repository = repository if repository is not None else ActivationRepository(
            get_default_runtime()
        )
        self.stats_computer = (
            stats_computer if stats_computer is not None else StatisticsComputer()
        )

    def inspect(
        self,
        layer_index: int,
        neuron_index: int,
        token_index: int = 0,
    ) -> NeuronInspection:
        """Inspect a single neuron at a specific layer and position.

        Parameters
        ----------
        layer_index : int
            Index of the layer to inspect.
        neuron_index : int
            Index of the neuron within the layer.
        token_index : int, default 0
            Token position to inspect (for sequence data).

        Returns
        -------
        NeuronInspection
            Complete neuron inspection data.
        """
        layer_name = self.repository.get_layer_name(layer_index)
        activation = self.repository.get_neuron_activation(
            layer_name, neuron_index, token_index
        )
        activation_history = self.repository.get_neuron_activations(
            layer_name, neuron_index
        )
        stats = self.stats_computer.compute(activation_history)

        neuron_id = f"{layer_name}.neuron_{neuron_index}"

        return NeuronInspection(
            neuron_id=neuron_id,
            layer=layer_name,
            layer_index=layer_index,
            neuron_index=neuron_index,
            activation=activation,
            activation_history=activation_history.tolist(),
            statistics=stats,
            top_tokens=self._find_top_tokens(
                layer_name, neuron_index, top_k=5
            ),
            description=self._generate_description(
                neuron_index, layer_index, stats
            ),
        )

    def inspect_batch(
        self,
        layer_index: int,
        neuron_indices: List[int],
        token_index: int = 0,
    ) -> List[NeuronInspection]:
        """Inspect multiple neurons in the same layer."""
        return [
            self.inspect(layer_index, ni, token_index)
            for ni in neuron_indices
        ]

    def inspect_layer(
        self,
        layer_index: int,
        top_k: int = 10,
        token_index: int = 0,
    ) -> List[NeuronInspection]:
        """Inspect the top-k most active neurons in a layer."""
        layer_name = self.repository.get_layer_name(layer_index)
        activations = self.repository.get_activations(layer_name)

        token_activations = activations[token_index, :]
        top_indices = np.argsort(np.abs(token_activations))[-top_k:][::-1]

        return [
            self.inspect(layer_index, int(idx), token_index)
            for idx in top_indices
        ]

    def get_neuron_statistics(
        self,
        layer_index: int,
        neuron_index: int,
    ) -> Statistics:
        """Get just the statistics for a neuron across all tokens."""
        layer_name = self.repository.get_layer_name(layer_index)
        activation_history = self.repository.get_neuron_activations(
            layer_name, neuron_index
        )
        return self.stats_computer.compute(activation_history)

    def search(
        self,
        layer_index: int,
        threshold: float = 5.0,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search for neurons with activation above a threshold.

        Parameters
        ----------
        layer_index : int
            Layer to search.
        threshold : float, default 5.0
            Minimum absolute activation value.
        top_k : int, default 20
            Maximum number of results.

        Returns
        -------
        list of dict
            Matching neurons with layer, neuron_index, and activation.
        """
        layer_name = self.repository.get_layer_name(layer_index)
        activations = self.repository.get_activations(layer_name)

        # Find neurons with max absolute activation above threshold
        max_activations = np.max(np.abs(activations), axis=0)
        matching_indices = np.where(max_activations > threshold)[0]

        # Sort by max activation (descending)
        sorted_indices = matching_indices[
            np.argsort(max_activations[matching_indices])[::-1]
        ][:top_k]

        results = []
        for idx in sorted_indices:
            results.append(
                {
                    "layer": layer_name,
                    "layer_index": layer_index,
                    "neuron_index": int(idx),
                    "max_activation": float(max_activations[idx]),
                    "neuron_id": f"{layer_name}.neuron_{int(idx)}",
                }
            )
        return results

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _find_top_tokens(
        self,
        layer_name: str,
        neuron_index: int,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find the top tokens that maximally activate a neuron."""
        activation_history = self.repository.get_neuron_activations(
            layer_name, neuron_index
        )
        tokens = self.repository.get_tokens()

        sorted_indices = np.argsort(activation_history)[::-1][:top_k]

        results = []
        for idx in sorted_indices:
            results.append(
                {
                    "token_index": int(idx),
                    "token_id": int(tokens[idx]) if idx < len(tokens) else None,
                    "activation": float(activation_history[idx]),
                }
            )
        return results

    def _generate_description(
        self,
        neuron_index: int,
        layer_index: int,
        stats: Statistics,
    ) -> str:
        """Generate a human-readable description of a neuron's function."""
        if stats.sparsity > 0.8:
            activity = "highly sparse (rarely activates)"
        elif stats.sparsity > 0.5:
            activity = "moderately sparse"
        else:
            activity = "frequently active"

        return (
            f"Neuron {neuron_index} in layer {layer_index} "
            f"({activity}). "
            f"Mean activation: {stats.mean:.4f}, "
            f"Max: {stats.max:.4f}, "
            f"Variance: {stats.variance:.4f}."
        )
