"""Tests for the VisualizationGenerator."""

import json

import numpy as np
import pytest

from backend.interpretability.models import VisualizationDTO
from backend.interpretability.heatmap import VisualizationGenerator


class TestVisualizationGenerator:
    """Tests for the VisualizationGenerator class."""

    def test_prepare_neuron_visualization(self, heatmap_generator):
        """Test preparing a neuron activation visualization."""
        result = heatmap_generator.prepare_neuron_visualization(
            layer_index=0, top_k=5
        )
        assert isinstance(result, VisualizationDTO)
        assert result.visualization_type == "neuron"
        assert result.title is not None
        assert len(result.data) == 5
        assert len(result.data[0]) == 8
        assert result.x_labels is not None
        assert result.y_labels is not None
        assert result.color_scale == "viridis"
        assert result.metadata is not None

    def test_prepare_neuron_visualization_custom_tokens(self, heatmap_generator):
        """Test preparing a neuron visualization with specific tokens."""
        result = heatmap_generator.prepare_neuron_visualization(
            layer_index=0, top_k=3, token_indices=[0, 2, 4]
        )
        assert len(result.data[0]) == 3
        assert len(result.data) == 3

    def test_prepare_attention_visualization_full(self, heatmap_generator):
        """Test preparing a full attention visualization."""
        result = heatmap_generator.prepare_attention_visualization(
            layer_index=0, head_index=0
        )
        assert isinstance(result, VisualizationDTO)
        assert result.visualization_type == "attention"
        assert len(result.data) == 8
        assert len(result.data[0]) == 8
        assert result.color_scale == "plasma"
        assert result.metadata["head"] == 0
        assert result.metadata["importance"] is not None

    def test_prepare_attention_visualization_token(self, heatmap_generator):
        """Test preparing an attention visualization for a specific token."""
        result = heatmap_generator.prepare_attention_visualization(
            layer_index=0, head_index=0, token_index=3
        )
        assert len(result.data) == 1
        assert len(result.data[0]) == 8

    def test_prepare_residual_visualization(self, heatmap_generator):
        """Test preparing a residual stream visualization."""
        result = heatmap_generator.prepare_residual_visualization(
            token_index=0, num_dims=16
        )
        assert isinstance(result, VisualizationDTO)
        assert result.visualization_type == "residual"
        assert len(result.data) == 4
        assert len(result.data[0]) == 16
        assert result.color_scale == "coolwarm"
        assert result.metadata["num_layers"] == 4

    def test_prepare_all_visualizations(self, heatmap_generator):
        """Test preparing all visualization types at once."""
        results = heatmap_generator.prepare_all_visualizations(
            layer_index=0, head_index=0, token_index=0
        )
        assert "neuron" in results
        assert "attention" in results
        assert "residual" in results
        assert all(isinstance(v, VisualizationDTO) for v in results.values())

    def test_to_json(self, heatmap_generator):
        """Test serializing a visualization to JSON."""
        dto = heatmap_generator.prepare_neuron_visualization(
            layer_index=0, top_k=3
        )
        json_str = heatmap_generator.to_json(dto)
        data = json.loads(json_str)

        assert data["visualization_type"] == "neuron"
        assert "title" in data
        assert "data" in data
        assert "x_labels" in data
        assert "y_labels" in data
        assert "color_scale" in data
        assert "metadata" in data

    def test_to_json_without_metadata(self, heatmap_generator):
        """Test serializing a visualization to JSON without metadata."""
        dto = heatmap_generator.prepare_neuron_visualization(
            layer_index=0, top_k=3
        )
        json_str = heatmap_generator.to_json(dto, include_metadata=False)
        data = json.loads(json_str)

        assert "metadata" not in data

    def test_attention_visualization_importance_in_metadata(self, heatmap_generator):
        """Test that importance is included in attention visualization metadata."""
        result = heatmap_generator.prepare_attention_visualization(
            layer_index=0, head_index=0
        )
        assert "importance" in result.metadata
        assert 0.0 <= result.metadata["importance"] <= 1.0

    def test_residual_visualization_metadata(self, heatmap_generator):
        """Test that residual visualization metadata is correct."""
        result = heatmap_generator.prepare_residual_visualization(
            token_index=2, num_dims=8
        )
        assert result.metadata["token_index"] == 2
        assert result.metadata["num_dims_shown"] == 8
        assert result.metadata["hidden_dim"] == 32
