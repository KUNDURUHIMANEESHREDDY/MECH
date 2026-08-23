"""Hallucination Competition Experiment Plugin for MECH Platform."""

import logging
from typing import Any, Dict, List, Optional

from backend.core.plugins.base import BaseTool, FunctionTool, ParameterSpec, Plugin, PluginManifest
from backend.core.plugins.registry import get_registry

logger = logging.getLogger("MECH.plugins.builtins.hallucination_experiment")


def _handle_hallucination_experiment(
    clean_factual_prompt: str = "The primary author of the paper 'Attention Is All You Need' is",
    fabricated_prompt: str = "The primary author of the mythical manuscript 'The Lost Chronicles of Atlantis' is",
    factual_target: str = " Ashish",
    fabricated_target: str = " Eldrin",
    candidate_mlp: Optional[str] = None,
    candidate_head: Optional[str] = None,
    grid_resolution: int = 5,
    **_: Any,
) -> Dict[str, Any]:
    try:
        from backend.science.reproducibility.hallucination_pipeline import HallucinationCompetitionPipeline

        pipeline = HallucinationCompetitionPipeline()
        report = pipeline.run_experiment(
            clean_factual_prompt=clean_factual_prompt,
            fabricated_prompt=fabricated_prompt,
            factual_target=factual_target,
            fabricated_target=fabricated_target,
            candidate_mlp=candidate_mlp,
            candidate_head=candidate_head,
            grid_resolution=grid_resolution,
        )
        return {
            "tool": "run_hallucination_experiment",
            "status": "success",
            "report": report.to_dict(),
        }
    except Exception as e:
        return {"tool": "run_hallucination_experiment", "status": "error", "error": str(e)}


def create_hallucination_experiment_tool() -> BaseTool:
    return FunctionTool(
        name="run_hallucination_experiment",
        description="Executes the 4-pillar causal competition experiment between parametric MLP memory and induction attention heads to prove hallucination circuits.",
        handler=_handle_hallucination_experiment,
        parameters={
            "clean_factual_prompt": ParameterSpec(type="string", required=True),
            "fabricated_prompt": ParameterSpec(type="string", required=True),
            "factual_target": ParameterSpec(type="string", required=False, default=" Ashish"),
            "fabricated_target": ParameterSpec(type="string", required=False, default=" Eldrin"),
            "candidate_mlp": ParameterSpec(type="string", required=False),
            "candidate_head": ParameterSpec(type="string", required=False),
            "grid_resolution": ParameterSpec(type="integer", required=False, default=5),
        },
        category="causal",
    )


class HallucinationExperimentPlugin(Plugin):
    """Hallucination Competition Experiment plugin for causal hallucination analysis."""

    def __init__(self) -> None:
        manifest = PluginManifest(
            name="hallucination_experiment",
            version="1.0.0",
            description="4-pillar causal competition experiment for hallucination circuit discovery",
            author="MECH Platform",
            capabilities=["causal", "hallucination", "reproducibility"],
            tags=["interpretability", "causal", "hallucination", "competition", "pipeline"],
        )
        super().__init__(manifest)

    def register_tools(self, registry) -> None:
        tool = create_hallucination_experiment_tool()
        self._add_tool(tool)
        registry.register_tool(tool)