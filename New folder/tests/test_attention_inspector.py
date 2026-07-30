"""Tests for the AttentionInspector."""

import numpy as np
import pytest

from backend.interpretability.models import AttentionInspection, Statistics, HeadRanking
from backend.interpretability.attention_inspector import AttentionInspector


class TestAttentionInspector:
    """Tests for the AttentionInspector class."""

    def test_inspect_returns_attentiondata(self, attention_inspector):
        """Test that inspect returns an AttentionInspection object."""
        result = attention_inspector.inspect(layer_index=0, head_index=0)
        assert isinstance(result, AttentionInspection)

    def test_inspect_head_info(self, attention_inspector):
        """Test that head information is correct."""
        result = attention_inspector.inspect(layer_index=1, head_index=2)
        assert result.head == 2
        assert result.layer == "layer_1"
        assert result.layer_index == 1
        assert result.num_heads == 4

    def test_inspect_matrix(self, attention_inspector):
        """Test that the attention matrix is returned."""
        result = attention_inspector.inspect(layer_index=0, head_index=0)
        assert result.matrix is not None
        assert len(result.matrix) == 8
        assert len(result.matrix[0]) == 8

    def test_inspect_matrix_with_token(self, attention_inspector):
        """Test that the matrix is correct when token_index is specified."""
        result = attention_inspector.inspect(layer_index=0, head_index=0, token_index=3)
        assert len(result.matrix) == 1
        assert len(result.matrix[0]) == 8
        assert result.shape == [1, 8]

    def test_inspect_importance(self, attention_inspector):
        """Test that importance is computed and in valid range."""
        result = attention_inspector.inspect(layer_index=0, head_index=0)
        assert 0.0 <= result.importance <= 1.0

    def test_inspect_shape(self, attention_inspector):
        """Test that the shape is correct."""
        result = attention_inspector.inspect(layer_index=0, head_index=0)
        assert result.shape == [8, 8]

    def test_inspect_statistics(self, attention_inspector):
        """Test that statistics are computed."""
        result = attention_inspector.inspect(layer_index=0, head_index=0)
        assert isinstance(result.statistics, Statistics)

    def test_inspect_top_connections(self, attention_inspector):
        """Test that top connections are returned."""
        result = attention_inspector.inspect(layer_index=0, head_index=0)
        assert result.top_connections is not None
        assert len(result.top_connections) == 5
        for conn in result.top_connections:
            assert "query" in conn
            assert "key" in conn
            assert "weight" in conn

    def test_inspect_all_heads(self, attention_inspector):
        """Test inspecting all heads in a layer."""
        results = attention_inspector.inspect_all_heads(layer_index=0)
        assert len(results) == 4
        assert all(isinstance(r, AttentionInspection) for r in results)
        importances = [r.importance for r in results]
        assert importances == sorted(importances, reverse=True)

    def test_get_head_importance(self, attention_inspector):
        """Test getting importance for a specific head."""
        importance = attention_inspector.get_head_importance(layer_index=0, head_index=0)
        assert 0.0 <= importance <= 1.0

    def test_get_layer_importance(self, attention_inspector):
        """Test getting importance for all heads in a layer."""
        scores = attention_inspector.get_layer_importance(layer_index=0)
        assert len(scores) == 4
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_inspect_invalid_layer(self, attention_inspector):
        """Test that an invalid layer index raises IndexError."""
        with pytest.raises(IndexError):
            attention_inspector.inspect(layer_index=999, head_index=0)

    def test_inspect_invalid_head(self, attention_inspector):
        """Test that an invalid head index raises IndexError."""
        with pytest.raises(IndexError):
            attention_inspector.inspect(layer_index=0, head_index=999)

    def test_importance_uniform_matrix(self):
        """Test importance for a uniform attention matrix."""
        matrix = np.ones((4, 4)) / 4.0
        importance = AttentionInspector._compute_importance(matrix)
        assert 0.0 <= importance <= 1.0

    def test_importance_focused_matrix(self):
        """Test importance for a focused attention matrix."""
        matrix = np.zeros((4, 4))
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        matrix[2, 2] = 1.0
        matrix[3, 3] = 1.0
        importance = AttentionInspector._compute_importance(matrix)
        assert 0.0 <= importance <= 1.0

    def test_find_top_connections_full(self, attention_inspector):
        """Test finding top connections in a full matrix."""
        matrix = np.array([[0.1, 0.5, 0.4], [0.3, 0.2, 0.5], [0.6, 0.3, 0.1]])
        connections = attention_inspector._find_top_connections(matrix, token_index=None, top_k=3)
        assert len(connections) == 3
        assert connections[0]["weight"] == pytest.approx(0.6)
        assert connections[0]["query"] == 2
        assert connections[0]["key"] == 0

    def test_find_top_connections_token(self, attention_inspector):
        """Test finding top connections for a specific token."""
        matrix = np.array([[0.1, 0.5, 0.4], [0.3, 0.2, 0.5], [0.6, 0.3, 0.1]])
        connections = attention_inspector._find_top_connections(matrix, token_index=0, top_k=2)
        assert len(connections) == 2
        assert all(c["query"] == 0 for c in connections)
        assert connections[0]["key"] == 1
        assert connections[0]["weight"] == pytest.approx(0.5)

    def test_rank_heads(self, attention_inspector):
        """Test ranking heads by importance."""
        ranking = attention_inspector.rank_heads(layer_index=0, metric="importance")
        assert isinstance(ranking, HeadRanking)
        assert ranking.metric == "importance"
        assert len(ranking.rankings) == 4
        scores = [r["score"] for r in ranking.rankings]
        assert scores == sorted(scores, reverse=True)

    def test_rank_heads_entropy(self, attention_inspector):
        """Test ranking heads by entropy."""
        ranking = attention_inspector.rank_heads(layer_index=0, metric="entropy")
        assert ranking.metric == "entropy"
        assert len(ranking.rankings) == 4

    def test_rank_heads_sparsity(self, attention_inspector):
        """Test ranking heads by sparsity."""
        ranking = attention_inspector.rank_heads(layer_index=0, metric="sparsity")
        assert ranking.metric == "sparsity"
        assert len(ranking.rankings) == 4

    def test_rank_heads_invalid_metric(self, attention_inspector):
        """Test ranking heads with an invalid metric."""
        with pytest.raises(ValueError):
            attention_inspector.rank_heads(layer_index=0, metric="invalid")
