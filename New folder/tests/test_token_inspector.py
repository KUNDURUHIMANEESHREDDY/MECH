"""Tests for the TokenInspector."""

import pytest

from backend.interpretability.models import TokenInspection, Statistics
from backend.interpretability.token_inspector import TokenInspector


class TestTokenInspector:
    """Tests for the TokenInspector class."""

    def test_inspect_returns_tokeninspection(self, token_inspector):
        """Test that inspect returns a TokenInspection object."""
        result = token_inspector.inspect(token_index=0)
        assert isinstance(result, TokenInspection)

    def test_inspect_token_info(self, token_inspector):
        """Test that token information is correct."""
        result = token_inspector.inspect(token_index=3)
        assert result.token_index == 3
        assert result.token_id is not None

    def test_inspect_embedding(self, token_inspector):
        """Test that embedding is included by default."""
        result = token_inspector.inspect(token_index=0)
        assert result.embedding is not None
        assert len(result.embedding) == 32
        assert result.embedding_shape == [32]

    def test_inspect_without_embedding(self, token_inspector):
        """Test that embedding is excluded when requested."""
        result = token_inspector.inspect(token_index=0, include_embedding=False)
        assert result.embedding is None

    def test_inspect_attention(self, token_inspector):
        """Test that attention is included by default."""
        result = token_inspector.inspect(token_index=0)
        assert result.attention is not None

    def test_inspect_residual(self, token_inspector):
        """Test that residual is included by default."""
        result = token_inspector.inspect(token_index=0)
        assert result.residual is not None

    def test_inspect_logits(self, token_inspector):
        """Test that logits are included by default."""
        result = token_inspector.inspect(token_index=0)
        assert result.logits is not None

    def test_inspect_top_predictions(self, token_inspector):
        """Test that top predictions are returned."""
        result = token_inspector.inspect(token_index=0, top_k_predictions=5)
        assert result.top_predictions is not None
        assert len(result.top_predictions) == 5
        for pred in result.top_predictions:
            assert "token_id" in pred
            assert "logit" in pred
            assert "probability" in pred

    def test_inspect_statistics(self, token_inspector):
        """Test that statistics are computed."""
        result = token_inspector.inspect(token_index=0)
        assert isinstance(result.statistics, Statistics)

    def test_inspect_all_tokens(self, token_inspector):
        """Test inspecting all tokens."""
        results = token_inspector.inspect_all_tokens()
        assert len(results) == 8  # seq_len
        assert all(isinstance(r, TokenInspection) for r in results)

    def test_inspect_all_tokens_single(self, token_inspector):
        """Test inspecting a single token via inspect_all_tokens."""
        results = token_inspector.inspect_all_tokens(token_index=3)
        assert len(results) == 1
        assert results[0].token_index == 3
