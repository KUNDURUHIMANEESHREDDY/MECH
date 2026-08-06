"""Backward-compatible re-export of GPT2Model for the Electron desktop sidecar.

The sidecar at ``frontend/scripts/desktop_service.py`` imports
``GPT2Model`` from ``backend.neuron_inspector``, but the class actually
lives in ``backend.interpretability.gpt2_model``. This module re-exports
it so that import resolves correctly.
"""
from backend.interpretability.gpt2_model import GPT2Model

__all__ = ["GPT2Model"]
