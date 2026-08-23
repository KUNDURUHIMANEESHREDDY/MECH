"""Live Tensor Intervention Plugin for MECH Platform."""

import logging
from typing import Any, Dict, List, Optional

from backend.core.plugins.base import BaseTool, FunctionTool, ParameterSpec, Plugin, PluginManifest
from backend.core.plugins.registry import get_registry

logger = logging.getLogger("MECH.plugins.builtins.live_intervention")


def _handle_live_intervention(
    prompt: str = "The Eiffel Tower is in the city of",
    target_token: str = " Paris",
    node_interventions: Optional[Dict[str, float]] = None,
    **_: Any,
) -> Dict[str, Any]:
    try:
        from backend.interpretability.causal.live_intervention_engine import LiveInterventionEngine

        engine = LiveInterventionEngine()
        res = engine.run_live_intervention(
            prompt=prompt,
            target_token=target_token,
            node_interventions=node_interventions or {},
        )
        return {
            "tool": "run_live_tensor_intervention",
            "status": "success",
            "intervention_result": res.to_dict(),
        }
    except Exception as e:
        return {"tool": "run_live_tensor_intervention", "status": "error", "error": str(e)}


def create_live_intervention_tool() -> BaseTool:
    return FunctionTool(
        name="run_live_tensor_intervention",
        description="Executes a live forward pass with PyTorch hooks intercepting and modifying activation tensors to observe downstream model logits.",
        handler=_handle_live_intervention,
        parameters={
            "prompt": ParameterSpec(type="string", required=True),
            "target_token": ParameterSpec(type="string", required=False, default=" Paris"),
            "node_interventions": ParameterSpec(type="object", required=False, default={}),
        },
        category="causal",
    )


class LiveInterventionPlugin(Plugin):
    """Live Tensor Intervention plugin for real-time activation manipulation."""

    def __init__(self) -> None:
        manifest = PluginManifest(
            name="live_intervention",
            version="1.0.0",
            description="Live forward pass intervention with PyTorch hook-based activation manipulation",
            author="MECH Platform",
            capabilities=["causal", "live_intervention"],
            tags=["interpretability", "causal", "live", "intervention", "hooks"],
        )
        super().__init__(manifest)

    def register_tools(self, registry) -> None:
        tool = create_live_intervention_tool()
        self._add_tool(tool)
        registry.register_tool(tool)