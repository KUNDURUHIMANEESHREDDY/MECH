"""Path Patching Plugin for MECH Platform."""

import logging
from typing import Any, Dict, List, Optional

from backend.core.plugins.base import BaseTool, FunctionTool, ParameterSpec, Plugin, PluginManifest
from backend.core.plugins.registry import get_registry

logger = logging.getLogger("MECH.plugins.builtins.path_patching")


def _handle_path_patching(
    sender: str = "L5_H2",
    receiver: str = "L6_MLP",
    clean_prompt: str = "The Eiffel Tower is in the city of",
    corrupted_prompt: str = "The Colosseum is in the city of",
    target_token: str = " Paris",
    receiver_channel: str = "all",
    **_: Any,
) -> Dict[str, Any]:
    try:
        from backend.interpretability.causal.path_patching import EdgePathPatchingEngine

        engine = EdgePathPatchingEngine()
        res = engine.test_edge_mediation(
            sender=sender,
            receiver=receiver,
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            target_token=target_token,
            receiver_channel=receiver_channel,
        )
        return {
            "tool": "run_path_patching",
            "status": "success",
            "edge_mediation": res.to_dict(),
        }
    except Exception as e:
        return {"tool": "run_path_patching", "status": "error", "error": str(e)}


def create_path_patching_tool() -> BaseTool:
    return FunctionTool(
        name="run_path_patching",
        description="Intercepts and patches activations between sender component A and receiver component B to prove causal edge transmission.",
        handler=_handle_path_patching,
        parameters={
            "sender": ParameterSpec(type="string", required=True, description="Sender component ID, e.g. L5_H2"),
            "receiver": ParameterSpec(type="string", required=True, description="Receiver component ID, e.g. L6_MLP"),
            "clean_prompt": ParameterSpec(type="string", required=True),
            "corrupted_prompt": ParameterSpec(type="string", required=True),
            "target_token": ParameterSpec(type="string", required=False, default=" Paris"),
            "receiver_channel": ParameterSpec(
                type="string",
                required=False,
                default="all",
                enum=["all", "query", "key", "value", "mlp_in"],
            ),
        },
        category="causal",
    )


class PathPatchingPlugin(Plugin):
    """Path Patching plugin for edge-level causal mediation analysis."""

    def __init__(self) -> None:
        manifest = PluginManifest(
            name="path_patching",
            version="1.0.0",
            description="Edge-level path patching for causal edge mediation analysis",
            author="MECH Platform",
            capabilities=["patching", "causal_tracing"],
            tags=["interpretability", "causal", "path_patching", "edges"],
        )
        super().__init__(manifest)

    def register_tools(self, registry) -> None:
        tool = create_path_patching_tool()
        self._add_tool(tool)
        registry.register_tool(tool)