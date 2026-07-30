"""Tests for the ResidualInspector."""

import numpy as np
import pytest

from backend.interpretability.models import ResidualInspection, Statistics
from backend.interpretability.residual_inspector import ResidualInspector


class TestResidualInspector:
    """Tests for the ResidualInspector class."""

    def test_inspect_returns_residualdata(self, residual_inspector):
        """Test that inspect returns a ResidualInspection object."""
        result = residual_inspector.inspect(layer_index=0, token_index=0)
        assert isinstance(result, ResidualInspection)

    def test_inspect_layer_info(self, residual_inspector):
        """Test that layer information is correct."""
        result = residual_inspector.inspect(layer_index=2, token_index=0)
        assert result.layer == "layer_2"
        assert result.layer_index == 2

    def test_inspect_residual_vector(self, residual_inspector):
        """Test that the residual vector is returned."""
        result = residual_inspector.inspect(layer_index=0, token_index=0)
        assert result.residual_vector is not None
        assert len(result.residual_vector) == 32

    def test_inspect_shape(self, residual_inspector):
        """Test that the shape is correct."""
        result = residual_inspector.inspect(layer_index=0, token_index=0)
        assert result.shape == [32]

    def test_inspect_statistics(self, residual_inspector):
        """Test that statistics are computed."""
        result = residual_inspector.inspect(layer_index=0, token_index=0)
        assert isinstance(result.statistics, Statistics)
        assert result.statistics.num_elements == 32

    def test_inspect_norm(self, residual_inspector, small_repository):
        """Test that the L2 norm is computed correctly."""
        layer_name = small_repository.get_layer_name(0)
        expected_norm = np.linalg.norm(small_repository.get_residual(layer_name)[0, :])

        result = residual_inspector.inspect(layer_index=0, token_index=0)
        assert result.norm == pytest.approx(expected_norm)

    def test_inspect_contribution(self, residual_inspector):
        """Test that contribution is computed and in valid range."""
        result = residual_inspector.inspect(layer_index=0, token_index=0)
        assert result.contribution is not None
        assert 0.0 <= result.contribution <= 1.0

    def test_inspect_all_layers(self, residual_inspector):
        """Test inspecting residual streams at all layers."""
        results = residual_inspector.inspect_all_layers(token_index=0)
        assert len(results) == 4
        assert all(isinstance(r, ResidualInspection) for r in results)

    def test_get_residual_norm(self, residual_inspector):
        """Test getting the L2 norm for a specific layer."""
        norm = residual_inspector.get_residual_norm(layer_index=0, token_index=0)
        assert norm > 0.0

    def test_get_residual_norms(self, residual_inspector):
        """Test getting L2 norms for all layers."""
        norms = residual_inspector.get_residual_norms(token_index=0)
        assert len(norms) == 4
        assert all(n > 0.0 for n in norms)

    def test_get_residual_vector(self, residual_inspector, small_repository):
        """Test getting the raw residual vector."""
        vec = residual_inspector.get_residual_vector(layer_index=0, token_index=0)
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 32

    def test_inspect_invalid_layer(self, residual_inspector):
        """Test that an invalid layer index raises IndexError."""
        with pytest.raises(IndexError):
            residual_inspector.inspect(layer_index=999, token_index=0)

    def test_inspect_different_token(self, residual_inspector, small_repository):
        """Test inspecting at a different token position."""
        layer_name = small_repository.get_layer_name(0)
        expected = small_repository.get_residual(layer_name)[3, :]

        result = residual_inspector.inspect(layer_index=0, token_index=3)
        assert result.residual_vector == pytest.approx(expected.tolist())

    def test_contribution_sums_to_one(self, residual_inspector):
        """Test that contributions across all layers sum to ~1.0."""
        results = residual_inspector.inspect_all_layers(token_index=0)
        total = sum(r.contribution for r in results)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_residual_vector_matches_model(self, residual_inspector, small_repository):
        """Test that the residual vector matches the model data."""
        layer_name = small_repository.get_layer_name(1)
        expected = small_repository.get_residual(layer_name)[0, :]

        result = residual_inspector.inspect(layer_index=1, token_index=0)
        assert result.residual_vector == pytest.approx(expected.tolist())
