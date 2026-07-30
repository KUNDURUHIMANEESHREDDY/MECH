"""
Runtime abstraction for the Neuron Inspector.

The Runtime provides access to model activations, attention weights,
and residual streams. It abstracts away the underlying model framework
(e.g., PyTorch, TensorFlow, JAX) and provides a uniform interface for
the Activation Repository.

In production, a concrete Runtime implementation wraps a real model.
For testing, a MockRuntime provides synthetic data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class Runtime(ABC):
    """Abstract base class for model runtimes.

    A Runtime wraps a model and provides access to its internal
    activations. Concrete implementations handle framework-specific
    details (PyTorch hooks, TensorFlow eager execution, etc.).
    """

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer layers in the model."""
        ...

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Number of attention heads per layer."""
        ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Dimensionality of the hidden / residual stream."""
        ...

    @property
    @abstractmethod
    def seq_len(self) -> int:
        """Sequence length (number of tokens)."""
        ...

    @property
    @abstractmethod
    def layer_names(self) -> List[str]:
        """Names of all layers in the model."""
        ...

    @abstractmethod
    def get_activations(self, layer_name: str) -> np.ndarray:
        """Return the activation matrix for a layer.

        Returns
        -------
        np.ndarray
            Shape: (seq_len, hidden_dim)
        """
        ...

    @abstractmethod
    def get_attention(self, layer_name: str) -> np.ndarray:
        """Return the attention tensor for a layer.

        Returns
        -------
        np.ndarray
            Shape: (num_heads, seq_len, seq_len)
        """
        ...

    @abstractmethod
    def get_residual(self, layer_name: str) -> np.ndarray:
        """Return the residual stream for a layer.

        Returns
        -------
        np.ndarray
            Shape: (seq_len, hidden_dim)
        """
        ...

    @abstractmethod
    def get_tokens(self) -> List[int]:
        """Return the token sequence."""
        ...

    @abstractmethod
    def get_logits(self, layer_index: int) -> np.ndarray:
        """Return the logits at a given layer.

        Returns
        -------
        np.ndarray
            Shape: (seq_len, vocab_size) or (vocab_size,)
        """
        ...

    @abstractmethod
    def get_embeddings(self) -> np.ndarray:
        """Return the token embeddings.

        Returns
        -------
        np.ndarray
            Shape: (seq_len, hidden_dim)
        """
        ...
