"""Unified backward-compatible re-export of Neuron Inspectors and Models for the Electron sidecar and legacy modules.

Provides access to:
- ``GPT2Model``: TransformerLens / PyTorch GPT-2 inference engine.
- ``NeuronInspector``: Analytical interpretability neuron inspector (activations, statistics, top tokens, batch/layer search).
- ``ExplorerNeuronInspector``: UI Neural Explorer inspector (histograms, connectivity, SAE feature overlap, literature).
"""

from backend.interpretability.gpt2_model import GPT2Model
from backend.interpretability.neuron_inspector import NeuronInspector
from backend.science.explorer.neuron_inspector import NeuronInspector as ExplorerNeuronInspector

__all__ = ["GPT2Model", "NeuronInspector", "ExplorerNeuronInspector"]
