"""
Experiment configuration for the Neuron Inspector.

Defines the ExperimentConfig dataclass and a default configuration
that can be used to run experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for running interpretability experiments.

    Attributes
    ----------
    num_layers : int
        Number of transformer layers in the model.
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
    top_k_neurons : int
        Number of top neurons to inspect per layer.
    token_index : int
        Token position to use for analysis.
    layers_to_analyze : list of int, optional
        Specific layer indices to analyze. If None, all layers are used.
    output_dir : str
        Directory for saving experiment results.
    """

    num_layers: int = 12
    num_heads: int = 12
    hidden_dim: int = 768
    seq_len: int = 16
    vocab_size: int = 50257
    seed: int = 42
    top_k_neurons: int = 10
    token_index: int = 0
    layers_to_analyze: Optional[List[int]] = None
    output_dir: str = "experiments/results"

    def get_layers(self) -> List[int]:
        """Return the list of layer indices to analyze."""
        if self.layers_to_analyze is not None:
            return self.layers_to_analyze
        return list(range(self.num_layers))


# Default configuration
DEFAULT_CONFIG = ExperimentConfig()
