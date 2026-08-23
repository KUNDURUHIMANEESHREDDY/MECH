"""Backup & Redundant Circuits Discovery Plugin for MECH Platform."""

import logging
from typing import Any, Dict, List, Optional

from backend.core.plugins.base import BaseTool, FunctionTool, ParameterSpec, Plugin, PluginManifest
from backend.core.plugins.registry import get_registry

logger = logging.getLogger("MECH.plugins.builtins.backup_circuits")


def _handle_backup_circuits(
    circuit_nodes: Optional[List[Dict[str, Any]]] = None,
    circuit_edges: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "gpt2",
    prompt: str = "The Eiffel Tower is located in the city of",
    target_token: str = " Paris",
    **_: Any,
) -> Dict[str, Any]:
    try:
        from backend.science.redundancy.backup_head_engine import RedundantBackupDiscoveryEngine

        engine = RedundantBackupDiscoveryEngine()
        envelope = engine.discover_redundant_backup_circuits(
            circuit_nodes=circuit_nodes or [],
            circuit_edges=circuit_edges or [],
            model_name=model_name,
            prompt=prompt,
            target_token=target_token,
        )
        return {
            "tool": "discover_backup_redundant_circuits",
            "status": "success",
            "backup_envelope": envelope.to_dict(),
        }
    except Exception as e:
        return {"tool": "discover_backup_redundant_circuits", "status": "error", "error": str(e)}


def create_backup_circuits_tool() -> BaseTool:
    return FunctionTool(
        name="discover_backup_redundant_circuits",
        description="Discovers active compensatory backup heads and quantifies transformer self-repair resilience via multi-order combinatorial knockouts (Wang et al., 2022).",
        handler=_handle_backup_circuits,
        parameters={
            "circuit_nodes": ParameterSpec(type="array", required=False, default=[]),
            "circuit_edges": ParameterSpec(type="array", required=False, default=[]),
            "model_name": ParameterSpec(type="string", required=False, default="gpt2"),
            "prompt": ParameterSpec(type="string", required=False, default="The Eiffel Tower is located in the city of"),
            "target_token": ParameterSpec(type="string", required=False, default=" Paris"),
        },
        category="redundancy",
    )


class BackupCircuitsPlugin(Plugin):
    """Backup & Redundant Circuits Discovery plugin for transformer self-repair analysis."""

    def __init__(self) -> None:
        manifest = PluginManifest(
            name="backup_circuits",
            version="1.0.0",
            description="Redundant backup head discovery and transformer self-repair quantification (Wang et al., 2022)",
            author="MECH Platform",
            capabilities=["redundancy", "backup_heads"],
            tags=["interpretability", "redundancy", "backup", "self_repair", "knockout"],
        )
        super().__init__(manifest)

    def register_tools(self, registry) -> None:
        tool = create_backup_circuits_tool()
        self._add_tool(tool)
        registry.register_tool(tool)