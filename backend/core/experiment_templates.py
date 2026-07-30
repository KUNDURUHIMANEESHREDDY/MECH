"""Curated Experiment Templates System."""

from __future__ import annotations

from typing import Any, Dict, List


class ExperimentTemplatesSystem:
    """Provides pre-built research experiment templates (Induction Heads, IOI, SAE Probing, Causal Tracing)."""

    def __init__(self) -> None:
        self.templates: Dict[str, Dict[str, Any]] = {
            "tmpl_ioi": {"name": "Indirect Object Identification", "category": "CircuitDiscovery", "default_model": "GPT-2 Small"},
            "tmpl_induction": {"name": "Induction Head Detection", "category": "AttentionPattern", "default_model": "GPT-2 Small"},
            "tmpl_sae_probing": {"name": "SAE Feature Probing", "category": "FeatureAnalysis", "default_model": "Gemma-2B"},
            "tmpl_causal_tracing": {"name": "Activation Patching & Causal Tracing", "category": "CausalIntervention", "default_model": "Llama-3-8B"},
        }

    def list_templates(self) -> List[Dict[str, Any]]:
        return [{"id": k, **v} for k, v in self.templates.items()]
