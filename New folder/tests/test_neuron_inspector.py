"""Tests for the NeuronInspector."""

import numpy as np
import pytest

from backend.interpretability.models import NeuronInspection, Statistics
from backend.interpretability.neuron_inspector import NeuronInspector


class TestNeuronInspector:
    """Tests for the NeuronInspector class."""

    def test_inspect_returns_neurondata(self, neuron_inspector):
        """Test that inspect returns a NeuronInspection object."""
        result = neuron_inspector.inspect(layer_index=0, neuron_index=0, token_index=0)
        assert isinstance(result, NeuronInspection)

    def test_inspect_neuron_id(self, neuron_inspector):
        """Test that neuron_id is correctly formatted."""
        result = neuron_inspector.inspect(layer_index=2, neuron_index=5, token_index=0)
        assert result.neuron_id == "layer_2.neuron_5"

    def test_inspect_layer_info(self, neuron_inspector):
        """Test that layer information is correct."""
        result = neuron_inspector.inspect(layer_index=1, neuron_index=3, token_index=0)
        assert result.layer == "layer_1"
        assert result.layer_index == 1
        assert result.neuron_index == 3

    def test_inspect_activation_value(self, neuron_inspector, small_repository):
        """Test that the activation value matches the model data."""
        layer_name = small_repository.get_layer_name(0)
        expected_activation = small_repository.get_neuron_activation(layer_name, 0, 0)

        result = neuron_inspector.inspect(layer_index=0, neuron_index=0, token_index=0)
        assert result.activation == pytest.approx(expected_activation)

    def test_inspect_activation_history(self, neuron_inspector):
        """Test that activation history is returned."""
        result = neuron_inspector.inspect(layer_index=0, neuron_index=0, token_index=0)
        assert result.activation_history is not None
        assert len(result.activation_history) == 8  # seq_len

    def test_inspect_statistics(self, neuron_inspector):
        """Test that statistics are computed."""
        result = neuron_inspector.inspect(layer_index=0, neuron_index=0, token_index=0)
        assert isinstance(result.statistics, Statistics)
        assert result.statistics.num_elements == 8

    def test_inspect_top_tokens(self, neuron_inspector):
        """Test that top tokens are returned."""
        result = neuron_inspector.inspect(layer_index=0, neuron_index=0, token_index=0)
        assert result.top_tokens is not None
        assert len(result.top_tokens) == 5
        for token in result.top_tokens:
            assert "token_index" in token
            assert "activation" in token

    def test_inspect_description(self, neuron_inspector):
        """Test that a description is generated."""
        result = neuron_inspector.inspect(layer_index=0, neuron_index=0, token_index=0)
        assert result.description is not None
        assert "Neuron" in result.description

    def test_inspect_batch(self, neuron_inspector):
        """Test batch inspection of multiple neurons."""
        results = neuron_inspector.inspect_batch(
            layer_index=0, neuron_indices=[0, 1, 2], token_index=0
        )
        assert len(results) == 3
        assert all(isinstance(r, NeuronInspection) for r in results)
        assert results[0].neuron_index == 0
        assert results[1].neuron_index == 1
        assert results[2].neuron_index == 2

    def test_inspect_layer_top_k(self, neuron_inspector):
        """Test inspecting top-k neurons in a layer."""
        results = neuron_inspector.inspect_layer(
            layer_index=0, top_k=5, token_index=0
        )
        assert len(results) == 5
        assert all(isinstance(r, NeuronInspection) for r in results)
        activations = [abs(r.activation) for r in results]
        assert activations == sorted(activations, reverse=True)

    def test_get_neuron_statistics(self, neuron_inspector):
        """Test getting just the statistics for a neuron."""
        stats = neuron_inspector.get_neuron_statistics(
            layer_index=0, neuron_index=0
        )
        assert isinstance(stats, Statistics)
        assert stats.num_elements == 8

    def test_inspect_invalid_layer(self, neuron_inspector):
        """Test that an invalid layer index raises IndexError."""
        with pytest.raises(IndexError):
            neuron_inspector.inspect(layer_index=999, neuron_index=0)

    def test_inspect_different_token(self, neuron_inspector, small_repository):
        """Test inspecting at a different token position."""
        layer_name = small_repository.get_layer_name(0)
        expected = small_repository.get_neuron_activation(layer_name, 0, 3)

        result = neuron_inspector.inspect(
            layer_index=0, neuron_index=0, token_index=3
        )
        assert result.activation == pytest.approx(expected)

    def test_search(self, neuron_inspector):
        """Test activation search."""
        results = neuron_inspector.search(
            layer_index=0, threshold=0.5, top_k=5
        )
        assert isinstance(results, list)
        for r in results:
            assert "neuron_index" in r
            assert "max_activation" in r
            assert r["max_activation"] > 0.5
