"""
Shared test fixtures for the Neuron Inspector test suite.
"""

import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from backend.interpretability.mock_runtime import MockRuntime
from backend.interpretability.repository import ActivationRepository
from backend.interpretability.statistics import StatisticsComputer
from backend.interpretability.neuron_inspector import NeuronInspector
from backend.interpretability.attention_inspector import AttentionInspector
from backend.interpretability.residual_inspector import ResidualInspector
from backend.interpretability.layer_inspector import LayerInspector
from backend.interpretability.token_inspector import TokenInspector
from backend.interpretability.logit_inspector import LogitInspector
from backend.interpretability.heatmap import VisualizationGenerator


@pytest.fixture
def small_runtime():
    """A small mock runtime for fast testing."""
    return MockRuntime(
        num_layers=4,
        num_heads=4,
        hidden_dim=32,
        seq_len=8,
        vocab_size=1000,
        seed=42,
    )


@pytest.fixture
def small_repository(small_runtime):
    """A small activation repository."""
    return ActivationRepository(small_runtime)


@pytest.fixture
def default_repository():
    """The default repository instance."""
    return ActivationRepository(MockRuntime())


@pytest.fixture
def stats_computer():
    """A StatisticsComputer instance."""
    return StatisticsComputer()


@pytest.fixture
def neuron_inspector(small_repository):
    """A NeuronInspector with the small repository."""
    return NeuronInspector(repository=small_repository)


@pytest.fixture
def attention_inspector(small_repository):
    """An AttentionInspector with the small repository."""
    return AttentionInspector(repository=small_repository)


@pytest.fixture
def residual_inspector(small_repository):
    """A ResidualInspector with the small repository."""
    return ResidualInspector(repository=small_repository)


@pytest.fixture
def layer_inspector(small_repository):
    """A LayerInspector with the small repository."""
    return LayerInspector(repository=small_repository)


@pytest.fixture
def token_inspector(small_repository):
    """A TokenInspector with the small repository."""
    return TokenInspector(repository=small_repository)


@pytest.fixture
def logit_inspector(small_repository):
    """A LogitInspector with the small repository."""
    return LogitInspector(repository=small_repository)


@pytest.fixture
def heatmap_generator(small_repository):
    """A VisualizationGenerator with the small repository."""
    return VisualizationGenerator(repository=small_repository)


@pytest.fixture
def sample_activation_vector():
    """A sample 1-D activation vector."""
    return np.array([0.0, 0.5, -0.3, 0.0, 0.0, 1.2, -0.8, 0.0])


@pytest.fixture
def sample_attention_matrix():
    """A sample 2-D attention matrix (rows sum to 1)."""
    raw = np.array([
        [0.1, 0.2, 0.7],
        [0.5, 0.3, 0.2],
        [0.3, 0.6, 0.1],
    ])
    return raw / raw.sum(axis=1, keepdims=True)
