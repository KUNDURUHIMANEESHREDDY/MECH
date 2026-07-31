"""Built-in Interpretability Pipeline Templates."""

from __future__ import annotations

from typing import Any, Dict, List


class PipelineTemplates:
    """Pre-configured pipeline templates for common research workflows."""

    TEMPLATES = {
        "sae_analysis": {
            "name": "SAE Feature Analysis Pipeline",
            "stages": ["Prompt", "Activation Search", "SAE Feature Lookup", "Dataset Examples", "Report"],
        },
        "circuit_discovery": {
            "name": "Circuit Discovery Pipeline",
            "stages": ["Prompt", "Attention Head Ranking", "Circuit Graph Mapping", "Report"],
        },
        "causal_tracing": {
            "name": "Causal Tracing & Patching Pipeline",
            "stages": ["Prompt", "Breakpoint Pause", "Activation Patch", "Logit Lens Projection", "Compare"],
        },
        "attribution_patching": {
            "name": "Attribution Patching Pipeline",
            "stages": ["Prompt", "Gradient Attribution", "Patch Set Application", "Compare"],
        },
        "model_comparison": {
            "name": "Cross-Architecture Model Comparison Pipeline",
            "stages": ["Prompt", "Model A Execution", "Model B Execution", "Normalized DTO Comparison", "Report"],
        },
    }

    @classmethod
    def list_templates(cls) -> List[Dict[str, Any]]:
        return [{"id": k, **v} for k, v in cls.TEMPLATES.items()]
