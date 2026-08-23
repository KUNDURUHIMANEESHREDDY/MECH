"""Cross-Model Universality Plugin for MECH Platform."""

import logging
from typing import Any, Dict, List, Optional

from backend.core.plugins.base import BaseTool, FunctionTool, ParameterSpec, Plugin, PluginManifest
from backend.core.plugins.registry import get_registry

logger = logging.getLogger("MECH.plugins.builtins.cross_model_universality")


def _handle_cross_model_sweep(
    circuit_nodes: Optional[List[Dict[str, Any]]] = None,
    circuit_edges: Optional[List[Dict[str, Any]]] = None,
    reference_model: str = "gpt2",
    target_models: Optional[List[str]] = None,
    task_name: str = "Factual Recall",
    **_: Any,
) -> Dict[str, Any]:
    try:
        from backend.science.comparative.cross_model_alignment import CrossModelUniversalityEngine

        engine = CrossModelUniversalityEngine()
        report = engine.evaluate_circuit_universality(
            circuit_nodes=circuit_nodes or [],
            circuit_edges=circuit_edges or [],
            reference_model=reference_model,
            target_models=target_models,
            task_name=task_name,
        )
        return {
            "tool": "run_cross_model_universality_sweep",
            "status": "success",
            "universality_report": report.to_dict(),
        }
    except Exception as e:
        return {"tool": "run_cross_model_universality_sweep", "status": "error", "error": str(e)}


def create_cross_model_universality_tool() -> BaseTool:
    return FunctionTool(
        name="run_cross_model_universality_sweep",
        description="Evaluates whether a discovered circuit represents a universal transformer primitive conserved across Gemma-2, LLaMA-3, and Qwen.",
        handler=_handle_cross_model_sweep,
        parameters={
            "circuit_nodes": ParameterSpec(type="array", required=False, default=[]),
            "circuit_edges": ParameterSpec(type="array", required=False, default=[]),
            "reference_model": ParameterSpec(type="string", required=False, default="gpt2"),
            "target_models": ParameterSpec(type="array", required=False, default=None),
            "task_name": ParameterSpec(type="string", required=False, default="Factual Recall"),
        },
        category="comparative",
    )


class CrossModelUniversalityPlugin(Plugin):
    """Cross-Model Universality plugin for circuit conservation analysis."""

    def __init__(self) -> None:
        manifest = PluginManifest(
            name="cross_model_universality",
            version="1.0.0",
            description="Cross-model circuit universality and comparative alignment evaluation",
            author="MECH Platform",
            capabilities=["comparative", "cross_model", "universality"],
            tags=["interpretability", "comparative", "cross_model", "universality", "alignment"],
        )
        super().__init__(manifest)

    def register_tools(self, registry) -> None:
        tool = create_cross_model_universality_tool()
        self._add_tool(tool)
        registry.register_tool(tool)