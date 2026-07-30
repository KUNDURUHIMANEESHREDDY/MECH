"""Tests for the LayerInspector."""

import pytest

from backend.interpretability.models import LayerInspection, Statistics
from backend.interpretability.layer_inspector import LayerInspector


class TestLayerInspector:
    """Tests for the LayerInspector class."""

    def test_inspect_returns_layerinspection(self, layer_inspector):
        """Test that inspect returns a LayerInspection object."""
        result = layer_inspector.inspect(layer_index=0)
        assert isinstance(result, LayerInspection)

    def test_inspect_layer_info(self, layer_inspector):
        """Test that layer information is correct."""
        result = layer_inspector.inspect(layer_index=2)
        assert result.layer == "layer_2"
        assert result.layer_index == 2
        assert result.layer_type is not None

    def test_inspect_with_residual(self, layer_inspector):
        """Test that residual data is included by default."""
        result = layer_inspector.inspect(layer_index=0)
        assert result.residual is not None

    def test_inspect_without_residual(self, layer_inspector):
        """Test that residual data is excluded when requested."""
        result = layer_inspector.inspect(layer_index=0, include_residual=False)
        assert result.residual is None

    def test_inspect_with_attention(self, layer_inspector):
        """Test that attention data is included when requested."""
        result = layer_inspector.inspect(layer_index=0, include_attention=True)
        assert result.attention is not None

    def test_inspect_with_neurons(self, layer_inspector):
        """Test that neuron data is included when requested."""
        result = layer_inspector.inspect(layer_index=0, include_neurons=True, top_k_neurons=5)
        assert result.neurons is not None
        assert len(result.neurons) == 5

    def test_inspect_statistics(self, layer_inspector):
        """Test that aggregate statistics are computed."""
        result = layer_inspector.inspect(layer_index=0)
        assert isinstance(result.statistics, Statistics)

    def test_inspect_metadata(self, layer_inspector):
        """Test that metadata is included."""
        result = layer_inspector.inspect(layer_index=0)
        assert result.metadata is not None
        assert "num_neurons" in result.metadata
        assert "num_heads" in result.metadata

    def test_inspect_all_layers(self, layer_inspector):
        """Test inspecting all layers."""
        results = layer_inspector.inspect_all_layers()
        assert len(results) == 4
        assert all(isinstance(r, LayerInspection) for r in results)

    def test_inspect_mlp_layer(self, layer_inspector):
        """Test inspecting an MLP layer."""
        # Layer 1 is an MLP layer (odd index)
        result = layer_inspector.inspect(layer_index=1)
        assert result.layer_type == "mlp"
        assert result.mlp is not None

    def test_inspect_attention_layer(self, layer_inspector):
        """Test inspecting an attention layer."""
        # Layer 0 is an attention layer (even index)
        result = layer_inspector.inspect(layer_index=0)
        assert result.layer_type == "attention"

    def test_inspect_invalid_layer(self, layer_inspector):
        """Test that an invalid layer index raises IndexError."""
        with pytest.raises(IndexError):
            layer_inspector.inspect(layer_index=999)
