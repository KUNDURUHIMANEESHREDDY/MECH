"""
Visualization Data Generator.

Prepares visualization data in JSON format for frontend rendering
of neurons, attention matrices, and residual streams.

Note: This module generates data for visualization. The actual
rendering (heatmap, graph, table, 3D) is handled by the frontend.

Architecture:

    Runtime

    ↓

    ActivationRepository

    ↓

    VisualizationGenerator

    ↓

    VisualizationDTO
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .repository import ActivationRepository
from .mock_runtime import get_default_runtime
from .models import VisualizationDTO
from .neuron_inspector import NeuronInspector
from .attention_inspector import AttentionInspector
from .residual_inspector import ResidualInspector


class VisualizationGenerator:
    """Generates visualization data for frontend rendering.

    Parameters
    ----------
    repository : ActivationRepository, optional
        The activation repository to use.
    """

    COLOR_SCALES = {
        "neuron": "viridis",
        "attention": "plasma",
        "residual": "coolwarm",
    }

    def __init__(self, repository: Optional[ActivationRepository] = None):
        self.repository = repository if repository is not None else ActivationRepository(
            get_default_runtime()
        )
        self.neuron_inspector = NeuronInspector(repository=self.repository)
        self.attention_inspector = AttentionInspector(repository=self.repository)
        self.residual_inspector = ResidualInspector(repository=self.repository)

    def prepare_neuron_visualization(
        self,
        layer_index: int,
        top_k: int = 20,
        token_indices: Optional[List[int]] = None,
    ) -> VisualizationDTO:
        """Prepare visualization data for neuron activations."""
        layer_name = self.repository.get_layer_name(layer_index)
        activations = self.repository.get_activations(layer_name)

        if token_indices is None:
            token_indices = list(range(activations.shape[0]))

        neuron_max = np.max(np.abs(activations), axis=0)
        top_neuron_indices = np.argsort(neuron_max)[-top_k:][::-1]

        heatmap_matrix = activations[np.ix_(token_indices, top_neuron_indices)]
        heatmap_matrix = heatmap_matrix.T

        return VisualizationDTO(
            visualization_type="neuron",
            title=f"Neuron Activations - {layer_name}",
            data=heatmap_matrix.tolist(),
            x_labels=[f"token_{t}" for t in token_indices],
            y_labels=[f"neuron_{int(n)}" for n in top_neuron_indices],
            color_scale=self.COLOR_SCALES["neuron"],
            metadata={
                "layer": layer_name,
                "layer_index": layer_index,
                "num_neurons": int(len(top_neuron_indices)),
                "num_tokens": int(len(token_indices)),
                "neuron_indices": [int(n) for n in top_neuron_indices],
            },
        )

    def prepare_attention_visualization(
        self,
        layer_index: int,
        head_index: int,
        token_index: Optional[int] = None,
    ) -> VisualizationDTO:
        """Prepare visualization data for an attention matrix."""
        layer_name = self.repository.get_layer_name(layer_index)
        full_matrix = self.repository.get_attention_head(layer_name, head_index)

        if token_index is not None:
            matrix = full_matrix[token_index, :].reshape(1, -1)
            x_labels = [f"key_{k}" for k in range(full_matrix.shape[1])]
            y_labels = [f"query_{token_index}"]
            title = (
                f"Attention - {layer_name} Head {head_index} "
                f"(Query: {token_index})"
            )
        else:
            matrix = full_matrix
            x_labels = [f"key_{k}" for k in range(full_matrix.shape[1])]
            y_labels = [f"query_{q}" for q in range(full_matrix.shape[0])]
            title = f"Attention - {layer_name} Head {head_index}"

        importance = self.attention_inspector.get_head_importance(
            layer_index, head_index
        )

        return VisualizationDTO(
            visualization_type="attention",
            title=title,
            data=matrix.tolist(),
            x_labels=x_labels,
            y_labels=y_labels,
            color_scale=self.COLOR_SCALES["attention"],
            metadata={
                "layer": layer_name,
                "layer_index": layer_index,
                "head": head_index,
                "num_heads": self.repository.num_heads,
                "importance": float(importance),
                "shape": list(matrix.shape),
            },
        )

    def prepare_residual_visualization(
        self,
        token_index: int = 0,
        num_dims: int = 64,
    ) -> VisualizationDTO:
        """Prepare visualization data for the residual stream."""
        all_residuals = []
        for layer_idx in range(self.repository.num_layers):
            vec = self.residual_inspector.get_residual_vector(
                layer_idx, token_index
            )
            if len(vec) >= num_dims:
                vec = vec[:num_dims]
            else:
                vec = np.pad(vec, (0, num_dims - len(vec)))
            all_residuals.append(vec)

        matrix = np.array(all_residuals)

        return VisualizationDTO(
            visualization_type="residual",
            title=f"Residual Stream - Token {token_index}",
            data=matrix.tolist(),
            x_labels=[f"dim_{d}" for d in range(num_dims)],
            y_labels=self.repository.layer_names,
            color_scale=self.COLOR_SCALES["residual"],
            metadata={
                "token_index": token_index,
                "num_layers": self.repository.num_layers,
                "num_dims_shown": num_dims,
                "hidden_dim": self.repository.hidden_dim,
            },
        )

    def prepare_all_visualizations(
        self,
        layer_index: int = 0,
        head_index: int = 0,
        token_index: int = 0,
    ) -> Dict[str, VisualizationDTO]:
        """Prepare all visualization types at once."""
        return {
            "neuron": self.prepare_neuron_visualization(
                layer_index=layer_index, token_indices=[token_index]
            ),
            "attention": self.prepare_attention_visualization(
                layer_index=layer_index,
                head_index=head_index,
                token_index=token_index,
            ),
            "residual": self.prepare_residual_visualization(
                token_index=token_index
            ),
        }

    def to_json(
        self,
        dto: VisualizationDTO,
        include_metadata: bool = True,
    ) -> str:
        """Serialize a VisualizationDTO to a JSON string."""
        import json

        data = dto.model_dump()
        if not include_metadata:
            data.pop("metadata", None)
        return json.dumps(data, indent=2)


# Backward-compatible alias
HeatmapGenerator = VisualizationGenerator
