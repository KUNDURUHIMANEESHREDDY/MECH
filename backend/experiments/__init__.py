"""
Experiments package for the Neuron Inspector.

Defines experiment configurations and pre-built experiment workflows
for neuron, attention, and residual stream analysis.
"""

from .config import ExperimentConfig, DEFAULT_CONFIG
from .neuron_experiment import NeuronExperiment
from .attention_experiment import AttentionExperiment

__all__ = [
    "ExperimentConfig",
    "DEFAULT_CONFIG",
    "NeuronExperiment",
    "AttentionExperiment",
]
