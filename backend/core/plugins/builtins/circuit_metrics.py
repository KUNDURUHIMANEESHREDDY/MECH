"""Circuit Metrics Plugin for MECH Platform."""

import logging
from typing import Any, Dict, List, Optional

from backend.core.plugins.base import BaseTool, FunctionTool, ParameterSpec, Plugin, PluginManifest
from backend.core.plugins.registry import get_registry

logger = logging.getLogger("MECH.plugins.builtins.circuit_metrics")


def _handle_circuit_metrics(
    nodes: Optional[List[Dict[str, Any]]] = None,
    edges: Optional[List[Dict[str, Any]]] = None,
    clean_prompt: str = "The Eiffel Tower is in the city of",
    corrupted_prompt: str = "The Colosseum is in the city of",
    target_token: str = " Paris",
    task_name: str = "Fact Retrieval",
    **_: Any,
) -> Dict[str, Any]:
    try:
        from backend.science.circuits.acdc_circuit_metrics import ACDCCircuitMetricsEvaluator

        evaluator = ACDCCircuitMetricsEvaluator()
        eval_res = evaluator.evaluate_circuit(
            nodes=nodes or [],
            edges=edges or [],
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            target_token=target_token,
            task_name=task_name,
        )
        return {
            "tool": "evaluate_circuit_metrics",
            "status": "success",
            "evaluation": eval_res.to_dict(),
        }
    except Exception as e:
        return {"tool": "evaluate_circuit_metrics", "status": "error", "error": str(e)}


def create_circuit_metrics_tool() -> BaseTool:
    return FunctionTool(
        name="evaluate_circuit_metrics",
        description="Computes formal ACDC metrics: Faithfulness (F >= 0.90), Completeness (C >= 0.85), and Minimality (M >= 0.80) on discovered subnetwork circuits.",
        handler=_handle_circuit_metrics,
        parameters={
            "nodes": ParameterSpec(type="array", required=False, default=[]),
            "edges": ParameterSpec(type="array", required=False, default=[]),
            "clean_prompt": ParameterSpec(type="string", required=False, default="The Eiffel Tower is in the city of"),
            "corrupted_prompt": ParameterSpec(type="string", required=False, default="The Colosseum is in the city of"),
            "target_token": ParameterSpec(type="string", required=False, default=" Paris"),
            "task_name": ParameterSpec(type="string", required=False, default="Fact Retrieval"),
        },
        category="verification",
    )


class CircuitMetricsPlugin(Plugin):
    """Circuit Metrics plugin for ACDC standard circuit evaluation."""

    def __init__(self) -> None:
        manifest = PluginManifest(
            name="circuit_metrics",
            version="1.0.0",
            description="ACDC standard circuit metrics evaluation (Faithfulness, Completeness, Minimality)",
            author="MECH Platform",
            capabilities=["verification", "acdc_circuits", "metrics"],
            tags=["interpretability", "verification", "circuits", "acdc", "metrics"],
        )
        super().__init__(manifest)

    def register_tools(self, registry) -> None:
        tool = create_circuit_metrics_tool()
        self._add_tool(tool)
        registry.register_tool(tool)