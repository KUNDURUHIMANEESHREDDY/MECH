"""
Layer Inspector.

Inspects a full transformer layer, including:
- Attention sub-layer
- MLP sub-layer
- Residual stream
- Aggregate statistics
- Metadata

Architecture:

    Runtime

    ↓

    ActivationRepository

    ↓

    LayerInspector

    ↓

    LayerInspection
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .repository import ActivationRepository
from .mock_runtime import get_default_runtime
from .models import LayerInspection, Statistics
from .statistics import StatisticsComputer
from .neuron_inspector import NeuronInspector
from .attention_inspector import AttentionInspector
from .residual_inspector import ResidualInspector


class LayerInspector:
    """Inspects full transformer layers.

    Parameters
    ----------
    repository : ActivationRepository, optional
        The activation repository to inspect.
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
        self.neuron_inspector = NeuronInspector(
            repository=self.repository, stats_computer=self.stats_computer
        )
        self.attention_inspector = AttentionInspector(
            repository=self.repository, stats_computer=self.stats_computer
        )
        self.residual_inspector = ResidualInspector(
            repository=self.repository, stats_computer=self.stats_computer
        )

    def inspect(
        self,
        layer_index: int,
        include_neurons: bool = False,
        include_attention: bool = False,
        include_residual: bool = True,
        top_k_neurons: int = 10,
        token_index: int = 0,
    ) -> LayerInspection:
        """Inspect a full layer.

        Parameters
        ----------
        layer_index : int
            Layer to inspect.
        include_neurons : bool, default False
            Include neuron data.
        include_attention : bool, default False
            Include attention data.
        include_residual : bool, default True
            Include residual stream data.
        top_k_neurons : int, default 10
            Number of top neurons to include.
        token_index : int, default 0
            Token position for analysis.

        Returns
        -------
        LayerInspection
            Complete layer inspection data.
        """
        layer_name = self.repository.get_layer_name(layer_index)
        layer_type = "attention" if layer_index % 2 == 0 else "mlp"

        attention = None
        if include_attention:
            attention = self.attention_inspector.inspect(
                layer_index, 0, token_index
            )

        mlp = None
        if layer_type == "mlp":
            mlp = self._inspect_mlp(layer_index, token_index)

        residual = None
        if include_residual:
            residual = self.residual_inspector.inspect(layer_index, token_index)

        neurons = None
        if include_neurons:
            neurons = self.neuron_inspector.inspect_layer(
                layer_index, top_k_neurons, token_index
            )

        activations = self.repository.get_activations(layer_name)
        stats = self.stats_computer.compute(activations)

        metadata = {
            "num_neurons": self.repository.hidden_dim,
            "num_heads": self.repository.num_heads,
            "hidden_dim": self.repository.hidden_dim,
            "seq_len": self.repository.seq_len,
        }

        return LayerInspection(
            layer=layer_name,
            layer_index=layer_index,
            layer_type=layer_type,
            attention=attention,
            mlp=mlp,
            residual=residual,
            neurons=neurons,
            statistics=stats,
            metadata=metadata,
        )

    def inspect_all_layers(
        self,
        include_neurons: bool = False,
        include_attention: bool = False,
        include_residual: bool = True,
        top_k_neurons: int = 10,
        token_index: int = 0,
    ) -> list:
        """Inspect all layers in the model."""
        return [
            self.inspect(
                layer_idx,
                include_neurons=include_neurons,
                include_attention=include_attention,
                include_residual=include_residual,
                top_k_neurons=top_k_neurons,
                token_index=token_index,
            )
            for layer_idx in range(self.repository.num_layers)
        ]

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _inspect_mlp(
        self,
        layer_index: int,
        token_index: int = 0,
    ) -> Dict[str, Any]:
        """Inspect the MLP sub-layer of a layer."""
        layer_name = self.repository.get_layer_name(layer_index)
        activations = self.repository.get_activations(layer_name)

        # Compute MLP statistics
        stats = self.stats_computer.compute(activations)

        # Compute top activating neurons
        token_activations = activations[token_index, :]
        top_indices = np.argsort(np.abs(token_activations))[-10:][::-1]

        top_neurons = [
            {
                "neuron_index": int(idx),
                "activation": float(token_activations[idx]),
            }
            for idx in top_indices
        ]

        return {
            "layer": layer_name,
            "layer_index": layer_index,
            "num_parameters": self.repository.hidden_dim * 4,
            "top_neurons": top_neurons,
            "statistics": stats.model_dump(),
        }
