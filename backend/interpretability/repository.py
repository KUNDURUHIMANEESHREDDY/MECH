"""
Activation Repository for the Neuron Inspector.

The Activation Repository sits between the Runtime and the Inspectors.
It caches activation data from the Runtime and provides a uniform
interface for inspectors to query. This abstraction hides the storage
details and allows multiple inspection sessions to share cached data.

Architecture:

    Runtime

    ↓

    ActivationRepository

    ↓

    Inspectors (Neuron, Attention, Residual, Layer, Token, Logits)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .runtime import Runtime


class ActivationRepository:
    """Caches and provides access to model activation data.

    The repository wraps a Runtime and caches activation data to avoid
    redundant computation. It provides a clean interface for inspectors
    to query activations, attention, residuals, logits, and embeddings.

    Parameters
    ----------
    runtime : Runtime
        The runtime providing raw activation data.
    cache : bool, default True
        Whether to cache activation data in memory.
    """

    def __init__(self, runtime: Runtime, cache: bool = True):
        self._runtime = runtime
        self._cache = cache
        self._activation_cache: Dict[str, np.ndarray] = {}
        self._attention_cache: Dict[str, np.ndarray] = {}
        self._residual_cache: Dict[str, np.ndarray] = {}
        self._logits_cache: Dict[int, np.ndarray] = {}
        self._embeddings_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  Properties (delegated to runtime)
    # ------------------------------------------------------------------ #

    @property
    def num_layers(self) -> int:
        return self._runtime.num_layers

    @property
    def num_heads(self) -> int:
        return self._runtime.num_heads

    @property
    def hidden_dim(self) -> int:
        return self._runtime.hidden_dim

    @property
    def seq_len(self) -> int:
        return self._runtime.seq_len

    @property
    def layer_names(self) -> List[str]:
        return self._runtime.layer_names

    @property
    def runtime(self) -> Runtime:
        """Return the underlying runtime."""
        return self._runtime

    # ------------------------------------------------------------------ #
    #  Data access methods
    # ------------------------------------------------------------------ #

    def get_layer_name(self, layer_index: int) -> str:
        """Return the layer name for a given index."""
        if 0 <= layer_index < self.num_layers:
            return self.layer_names[layer_index]
        raise IndexError(
            f"layer_index {layer_index} out of range [0, {self.num_layers})"
        )

    def get_activations(self, layer_name: str) -> np.ndarray:
        """Return the activation matrix for a layer, with caching."""
        if self._cache and layer_name in self._activation_cache:
            return self._activation_cache[layer_name]
        data = self._runtime.get_activations(layer_name)
        if self._cache:
            self._activation_cache[layer_name] = data
        return data

    def get_attention(self, layer_name: str) -> np.ndarray:
        """Return the attention tensor for a layer, with caching."""
        if self._cache and layer_name in self._attention_cache:
            return self._attention_cache[layer_name]
        data = self._runtime.get_attention(layer_name)
        if self._cache:
            self._attention_cache[layer_name] = data
        return data

    def get_residual(self, layer_name: str) -> np.ndarray:
        """Return the residual stream for a layer, with caching."""
        if self._cache and layer_name in self._residual_cache:
            return self._residual_cache[layer_name]
        data = self._runtime.get_residual(layer_name)
        if self._cache:
            self._residual_cache[layer_name] = data
        return data

    def get_tokens(self) -> List[int]:
        """Return the token sequence."""
        return self._runtime.get_tokens()

    def get_logits(self, layer_index: int) -> np.ndarray:
        """Return the logits at a given layer, with caching."""
        if self._cache and layer_index in self._logits_cache:
            return self._logits_cache[layer_index]
        data = self._runtime.get_logits(layer_index)
        if self._cache:
            self._logits_cache[layer_index] = data
        return data

    def get_embeddings(self) -> np.ndarray:
        """Return the token embeddings, with caching."""
        if self._cache and self._embeddings_cache is not None:
            return self._embeddings_cache
        data = self._runtime.get_embeddings()
        if self._cache:
            self._embeddings_cache = data
        return data

    # ------------------------------------------------------------------ #
    #  Convenience accessors
    # ------------------------------------------------------------------ #

    def get_neuron_activation(
        self, layer_name: str, neuron_index: int, token_index: int = 0
    ) -> float:
        """Return the activation of a specific neuron at a specific token."""
        return float(self.get_activations(layer_name)[token_index, neuron_index])

    def get_neuron_activations(
        self, layer_name: str, neuron_index: int
    ) -> np.ndarray:
        """Return the activation of a neuron across all tokens."""
        return self.get_activations(layer_name)[:, neuron_index]

    def get_attention_head(
        self, layer_name: str, head_index: int
    ) -> np.ndarray:
        """Return the attention matrix for a specific head."""
        return self.get_attention(layer_name)[head_index]

    def get_residual_vector(
        self, layer_name: str, token_index: int = 0
    ) -> np.ndarray:
        """Return the residual vector for a specific token."""
        return self.get_residual(layer_name)[token_index, :]

    def get_all_layer_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Return all data for all layers."""
        return {
            name: {
                "activations": self.get_activations(name),
                "attention": self.get_attention(name),
                "residual": self.get_residual(name),
            }
            for name in self.layer_names
        }

    # ------------------------------------------------------------------ #
    #  Cache management
    # ------------------------------------------------------------------ #

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._activation_cache.clear()
        self._attention_cache.clear()
        self._residual_cache.clear()
        self._logits_cache.clear()
        self._embeddings_cache = None

    def cache_info(self) -> Dict[str, int]:
        """Return information about the cache state."""
        return {
            "activations": len(self._activation_cache),
            "attention": len(self._attention_cache),
            "residuals": len(self._residual_cache),
            "logits": len(self._logits_cache),
            "embeddings": 1 if self._embeddings_cache is not None else 0,
        }
