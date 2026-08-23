"""Semantic Falsification Probe Plugin for MECH Platform."""

import logging
from typing import Any, Dict, List, Optional

from backend.core.plugins.base import BaseTool, FunctionTool, ParameterSpec, Plugin, PluginManifest
from backend.core.plugins.registry import get_registry

logger = logging.getLogger("MECH.plugins.builtins.semantic_falsification")


def _handle_semantic_falsification_probe(
    component: str = "L6_MLP",
    probe_type: str = "auto",
    **_: Any,
) -> Dict[str, Any]:
    try:
        from backend.science.probes.semantic_falsification import SemanticFalsificationSuite

        suite = SemanticFalsificationSuite()
        res = suite.probe_component(component=component)
        return {
            "tool": "run_semantic_falsification_probe",
            "status": "success",
            "component": component,
            "probe_report": res,
        }
    except Exception as e:
        return {"tool": "run_semantic_falsification_probe", "status": "error", "error": str(e)}


def create_semantic_falsification_tool() -> BaseTool:
    return FunctionTool(
        name="run_semantic_falsification_probe",
        description="Tests candidate components with relational tuple steering (MLPs) or synthetic prefix-matching (Attention Heads) to falsify or scientifically verify semantic role claims.",
        handler=_handle_semantic_falsification_probe,
        parameters={
            "component": ParameterSpec(
                type="string",
                required=True,
                description="Component ID to probe, e.g. L6_MLP or L8_H5",
            ),
            "probe_type": ParameterSpec(
                type="string",
                required=False,
                default="auto",
                enum=["auto", "tuple_steering", "prefix_matching"],
            ),
        },
        category="verification",
    )


class SemanticFalsificationPlugin(Plugin):
    """Semantic Falsification Probe plugin for component semantic role verification."""

    def __init__(self) -> None:
        manifest = PluginManifest(
            name="semantic_falsification",
            version="1.0.0",
            description="Semantic falsification probes for mechanistic claim verification",
            author="MECH Platform",
            capabilities=["verification", "falsification"],
            tags=["interpretability", "verification", "falsification", "semantic", "probes"],
        )
        super().__init__(manifest)

    def register_tools(self, registry) -> None:
        tool = create_semantic_falsification_tool()
        self._add_tool(tool)
        registry.register_tool(tool)