"""
Neuron Inspector - A neural network activation analysis toolkit.

Architecture:

    Runtime

    ↓

    ActivationRepository

    ↓

    Inspectors
        │
        ├── NeuronInspector
        ├── AttentionInspector
        ├── ResidualInspector
        ├── LayerInspector
        ├── TokenInspector
        └── LogitInspector

    ↓

    REST API (v1)

    ↓

    Frontend
"""

from .models import (
    Statistics,
    NeuronInspection,
    AttentionInspection,
    ResidualInspection,
    LayerInspection,
    TokenInspection,
    LogitInspection,
    HeadRanking,
    ActivationSearchResult,
    VisualizationDTO,
    # Backward-compatible aliases
    NeuronData,
    AttentionData,
    ResidualData,
    LayerData,
    HeatmapData,
)
from .statistics import StatisticsComputer
from .runtime import Runtime
from .mock_runtime import MockRuntime, get_default_runtime
from .repository import ActivationRepository
from .neuron_inspector import NeuronInspector
from .attention_inspector import AttentionInspector
from .residual_inspector import ResidualInspector
from .layer_inspector import LayerInspector
from .token_inspector import TokenInspector
from .logit_inspector import LogitInspector
from .heatmap import VisualizationGenerator, HeatmapGenerator
from .api import app

__version__ = "2.0.0"
__all__ = [
    # Models
    "Statistics",
    "NeuronInspection",
    "AttentionInspection",
    "ResidualInspection",
    "LayerInspection",
    "TokenInspection",
    "LogitInspection",
    "HeadRanking",
    "ActivationSearchResult",
    "VisualizationDTO",
    # Backward-compatible aliases
    "NeuronData",
    "AttentionData",
    "ResidualData",
    "LayerData",
    "HeatmapData",
    # Core
    "StatisticsComputer",
    "Runtime",
    "MockRuntime",
    "get_default_runtime",
    "ActivationRepository",
    # Inspectors
    "NeuronInspector",
    "AttentionInspector",
    "ResidualInspector",
    "LayerInspector",
    "TokenInspector",
    "LogitInspector",
    # Visualization
    "VisualizationGenerator",
    "HeatmapGenerator",
    # API
    "app",
]
