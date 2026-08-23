"""Scientific Circuit Validation Plugin for MECH Platform."""

import logging
from typing import Any, Dict, List, Optional

from backend.core.plugins.base import BaseTool, FunctionTool, ParameterSpec, Plugin, PluginManifest
from backend.core.plugins.registry import get_registry

logger = logging.getLogger("MECH.plugins.builtins.scientific_validation")


def _handle_scientific_validation(
    circuit_nodes: Optional[List[Dict[str, Any]]] = None,
    circuit_edges: Optional[List[Dict[str, Any]]] = None,
    task_name: str = "Factual Recall",
    hypothesis: str = "Discovered circuit causally mediates target prediction.",
    model_name: str = "gpt2",
    **_: Any,
) -> Dict[str, Any]:
    try:
        from backend.science.validation.scientific_validation_suite import ScientificValidationSuite

        suite = ScientificValidationSuite()
        report = suite.validate_circuit(
            circuit_nodes=circuit_nodes or [],
            circuit_edges=circuit_edges or [],
            task_name=task_name,
            hypothesis=hypothesis,
            model_name=model_name,
        )
        return {
            "tool": "run_scientific_circuit_validation",
            "status": "success",
            "validation_report": report.to_dict(),
        }
    except Exception as e:
        return {"tool": "run_scientific_circuit_validation", "status": "error", "error": str(e)}


def create_scientific_validation_tool() -> BaseTool:
    return FunctionTool(
        name="run_scientific_circuit_validation",
        description="Executes a 5-pillar scientific validation battery: held-out generalization, negative control specificity, multi-intervention triangulation, and statistical 95% confidence intervals.",
        handler=_handle_scientific_validation,
        parameters={
            "circuit_nodes": ParameterSpec(type="array", required=False, default=[]),
            "circuit_edges": ParameterSpec(type="array", required=False, default=[]),
            "task_name": ParameterSpec(type="string", required=False, default="Factual Recall"),
            "hypothesis": ParameterSpec(type="string", required=False, default=""),
            "model_name": ParameterSpec(type="string", required=False, default="gpt2"),
        },
        category="verification",
    )


class ScientificValidationPlugin(Plugin):
    """Scientific Circuit Validation plugin for rigorous circuit verification."""

    def __init__(self) -> None:
        manifest = PluginManifest(
            name="scientific_validation",
            version="1.0.0",
            description="5-pillar scientific validation suite for circuit claims",
            author="MECH Platform",
            capabilities=["verification", "scientific_validation"],
            tags=["interpretability", "verification", "scientific", "validation", "rigor"],
        )
        super().__init__(manifest)

    def register_tools(self, registry) -> None:
        tool = create_scientific_validation_tool()
        self._add_tool(tool)
        registry.register_tool(tool)