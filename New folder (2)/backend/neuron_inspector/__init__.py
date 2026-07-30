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

    IPC (JSON-lines over stdio)
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
    NeuronData,
    AttentionData,
    ResidualData,
    LayerData,
    HeatmapData,
)
from .gpt2_model import GPT2Model

__version__ = "2.0.0"
__all__ = [
    "GPT2Model",
]
