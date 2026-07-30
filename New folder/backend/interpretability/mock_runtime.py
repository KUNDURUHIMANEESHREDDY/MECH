"""
Mock Runtime for the Neuron Inspector.

Provides a concrete Runtime implementation with synthetic data for
testing and development. In production, a real Runtime implementation
wraps the actual model.

This module is NOT production code — it exists for testing and
demonstration purposes only.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .runtime import Runtime


class MockRuntime(Runtime):
    """A mock Runtime that generates synthetic activation data.

    Mimics a transformer model with configurable number of layers,
    attention heads, hidden dimension, and sequence length.

    Parameters
    ----------
    num_layers : int
        Number of transformer layers.
    num_heads : int
        Number of attention heads per layer.
    hidden_dim : int
        Dimensionality of the hidden / residual stream.
    seq_len : int
        Sequence length (number of tokens).
    vocab_size : int
        Size of the vocabulary.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        seq_len: int = 16,
        vocab_size: int = 50257,
        seed: int = 42,
    ):
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_dim = hidden_dim
        self._seq_len = seq_len
        self._vocab_size = vocab_size
        self._seed = seed

        self._rng = np.random.default_rng(seed)
        self._layer_names: List[str] = [f"layer_{i}" for i in range(num_layers)]

        # Pre-generate all data
        self._activations = self._generate_activations()
        self._attention = self._generate_attention()
        self._residuals = self._generate_residuals()
        self._tokens = self._generate_tokens()
        self._logits = self._generate_logits()
        self._embeddings = self._generate_embeddings()

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def layer_names(self) -> List[str]:
        return self._layer_names

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # ------------------------------------------------------------------ #
    #  Data generation
    # ------------------------------------------------------------------ #

    def _generate_tokens(self) -> List[int]:
        return self._rng.integers(0, self._vocab_size, size=self._seq_len).tolist()

    def _generate_activations(self) -> dict:
        activations = {}
        for i in range(self._num_layers):
            dense = self._rng.standard_normal((self._seq_len, self._hidden_dim)) * 0.5
            mask = self._rng.random((self._seq_len, self._hidden_dim)) > 0.6
            dense = dense * mask
            dense += 0.1 * i
            activations[self._layer_names[i]] = dense.astype(np.float64)
        return activations

    def _generate_attention(self) -> dict:
        attention = {}
        for i in range(self._num_layers):
            raw = self._rng.standard_normal(
                (self._num_heads, self._seq_len, self._seq_len)
            )
            raw = raw - np.max(raw, axis=-1, keepdims=True)
            exp_raw = np.exp(raw)
            weights = exp_raw / np.sum(exp_raw, axis=-1, keepdims=True)
            attention[self._layer_names[i]] = weights.astype(np.float64)
        return attention

    def _generate_residuals(self) -> dict:
        residuals = {}
        for i in range(self._num_layers):
            vec = self._rng.standard_normal(
                (self._seq_len, self._hidden_dim)
            ) * (0.3 + 0.05 * i)
            residuals[self._layer_names[i]] = vec.astype(np.float64)
        return residuals

    def _generate_logits(self) -> dict:
        logits = {}
        for i in range(self._num_layers):
            logits[i] = self._rng.standard_normal(
                (self._seq_len, min(self._vocab_size, 1000))
            ).astype(np.float64)
        return logits

    def _generate_embeddings(self) -> np.ndarray:
        return self._rng.standard_normal(
            (self._seq_len, self._hidden_dim)
        ).astype(np.float64)

    # ------------------------------------------------------------------ #
    #  Runtime interface
    # ------------------------------------------------------------------ #

    def get_activations(self, layer_name: str) -> np.ndarray:
        return self._activations[layer_name]

    def get_attention(self, layer_name: str) -> np.ndarray:
        return self._attention[layer_name]

    def get_residual(self, layer_name: str) -> np.ndarray:
        return self._residuals[layer_name]

    def get_tokens(self) -> List[int]:
        return self._tokens

    def get_logits(self, layer_index: int) -> np.ndarray:
        return self._logits[layer_index]

    def get_embeddings(self) -> np.ndarray:
        return self._embeddings


# Singleton instance for convenient access
_default_runtime: MockRuntime = None


def get_default_runtime() -> MockRuntime:
    """Return a singleton default MockRuntime instance."""
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = MockRuntime()
    return _default_runtime
