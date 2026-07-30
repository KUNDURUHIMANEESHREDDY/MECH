"""
Mock data generator for the Neuron Inspector.

Generates synthetic activation data, attention matrices, and residual
vectors that mimic the structure of a transformer model's internals.
This allows the inspectors and API to be tested and demonstrated
without requiring a real model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class MockModelData:
    """Generates and stores synthetic model activation data.

    The data mimics a transformer with configurable number of layers,
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
        Size of the vocabulary (for neuron / token mapping).
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
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed

        self._rng = np.random.default_rng(seed)

        # Layer names follow the convention: "layer_{i}"
        self.layer_names: List[str] = [f"layer_{i}" for i in range(num_layers)]

        # Pre-generate all data
        self._activations = self._generate_activations()
        self._attention = self._generate_attention()
        self._residuals = self._generate_residuals()
        self._tokens = self._generate_tokens()

    # ------------------------------------------------------------------ #
    #  Data generation
    # ------------------------------------------------------------------ #

    def _generate_tokens(self) -> List[int]:
        """Generate a random sequence of token IDs."""
        return self._rng.integers(0, self.vocab_size, size=self.seq_len).tolist()

    def _generate_activations(self) -> Dict[str, np.ndarray]:
        """Generate per-layer neuron activations.

        Returns a dict mapping layer name -> (seq_len, hidden_dim) array.
        """
        activations = {}
        for i in range(self.num_layers):
            # Mix of sparse and dense activations
            dense = self._rng.standard_normal((self.seq_len, self.hidden_dim)) * 0.5
            # Inject sparsity: zero out ~60% of elements
            mask = self._rng.random((self.seq_len, self.hidden_dim)) > 0.6
            dense = dense * mask
            # Add layer-dependent bias
            dense += 0.1 * i
            activations[self.layer_names[i]] = dense.astype(np.float64)
        return activations

    def _generate_attention(self) -> Dict[str, np.ndarray]:
        """Generate per-layer attention matrices.

        Returns a dict mapping layer name -> (num_heads, seq_len, seq_len)
        array of attention weights (rows sum to 1).
        """
        attention = {}
        for i in range(self.num_layers):
            raw = self._rng.standard_normal((self.num_heads, self.seq_len, self.seq_len))
            # Apply softmax along the key dimension (last axis)
            raw = raw - np.max(raw, axis=-1, keepdims=True)
            exp_raw = np.exp(raw)
            weights = exp_raw / np.sum(exp_raw, axis=-1, keepdims=True)
            attention[self.layer_names[i]] = weights.astype(np.float64)
        return attention

    def _generate_residuals(self) -> Dict[str, np.ndarray]:
        """Generate per-layer residual stream vectors.

        Returns a dict mapping layer name -> (seq_len, hidden_dim) array.
        """
        residuals = {}
        for i in range(self.num_layers):
            # Residuals grow slightly with depth
            vec = self._rng.standard_normal((self.seq_len, self.hidden_dim)) * (
                0.3 + 0.05 * i
            )
            residuals[self.layer_names[i]] = vec.astype(np.float64)
        return residuals

    # ------------------------------------------------------------------ #
    #  Accessors
    # ------------------------------------------------------------------ #

    def get_layer_name(self, layer_index: int) -> str:
        """Return the layer name for a given index."""
        if 0 <= layer_index < self.num_layers:
            return self.layer_names[layer_index]
        raise IndexError(
            f"layer_index {layer_index} out of range [0, {self.num_layers})"
        )

    def get_activation(self, layer_name: str) -> np.ndarray:
        """Return the activation matrix for a layer."""
        return self._activations[layer_name]

    def get_attention(self, layer_name: str) -> np.ndarray:
        """Return the attention tensor for a layer."""
        return self._attention[layer_name]

    def get_residual(self, layer_name: str) -> np.ndarray:
        """Return the residual vector for a layer."""
        return self._residuals[layer_name]

    def get_tokens(self) -> List[int]:
        """Return the token sequence."""
        return self._tokens

    def get_neuron_activation(
        self, layer_name: str, neuron_index: int, token_index: int = 0
    ) -> float:
        """Return the activation of a specific neuron at a specific token."""
        return float(self._activations[layer_name][token_index, neuron_index])

    def get_neuron_activations(
        self, layer_name: str, neuron_index: int
    ) -> np.ndarray:
        """Return the activation of a neuron across all tokens."""
        return self._activations[layer_name][:, neuron_index]

    def get_attention_head(
        self, layer_name: str, head_index: int
    ) -> np.ndarray:
        """Return the attention matrix for a specific head."""
        return self._attention[layer_name][head_index]

    def get_all_layer_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Return all data for all layers."""
        return {
            name: {
                "activations": self._activations[name],
                "attention": self._attention[name],
                "residual": self._residuals[name],
            }
            for name in self.layer_names
        }


# Singleton instance for convenient access
_default_model = None


def get_default_model() -> MockModelData:
    """Return a singleton default MockModelData instance."""
    global _default_model
    if _default_model is None:
        _default_model = MockModelData()
    return _default_model
