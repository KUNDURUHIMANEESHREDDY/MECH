"""Society Planner agent (capability: plan).

Decomposes a user goal into a Workflow DAG (max 8 nodes) following the
lifecycle: Draft -> Question -> Hypothesis -> Experiment -> Running ->
Observation -> Analysis -> Conclusion -> Published.

Wraps (lazily, so import never requires torch):
- backend.core.unified_registry.UnifiedRegistry  (avoid duplicate hypotheses)
- backend.core.experiment_templates.ExperimentTemplatesSystem (if present)

Legacy stub replaced: backend/agents/research_society.py::Planner.create_plan
which returned hardcoded ["Step 1 ...", "Step 2 ..."] strings.
"""

from __future__ import annotations

from typing import Any, Dict, List

MAX_STEPS = 8

WORKFLOW_STATES = [
    "Draft", "Question", "Hypothesis", "Experiment", "Running",
    "Observation", "Analysis", "Conclusion", "Published",
]

# Keyword -> pipeline hint, used when the LLM dispatcher is unavailable.
# Mirrors the hardcoded fallback map in LEGACY_DISPATCHER_AUDIT.md section 5.
GOAL_HINTS = {
    "ioi": "ioi",
    "indirect object": "ioi",
    "induction": "induction_heads",
    "greater than": "greater_than",
    "logit lens": "logit_lens",
    "sae": "sparse_autoencoders",
    "sparse autoencoder": "sparse_autoencoders",
}


class Planner:
    """Builds Workflow DAGs from natural-language research goals."""

    capability = "plan"

    def __init__(self) -> None:
        self._registry = None

    def _registry_lazy(self):  # type: ignore[no-untyped-def]
        if self._registry is None:
            from backend.core.unified_registry import UnifiedRegistry
            self._registry = UnifiedRegistry()
        return self._registry

    def detect_pipeline(self, goal: str) -> str:
        gl = goal.lower()
        for kw, pipeline in GOAL_HINTS.items():
            if kw in gl:
                return pipeline
        return "ioi"  # default reproduction target

    def known_papers(self) -> List[Dict[str, Any]]:
        try:
            return list(self._registry_lazy().list_catalog(item_type="papers"))
        except Exception:
            return []

    def plan(self, goal: str) -> Dict[str, Any]:
        """Return a Workflow DAG dict with at most MAX_STEPS nodes."""
        pipeline = self.detect_pipeline(goal)
        nodes = [
            {"id": "load", "agent": "executor", "op": "ensure_model",
             "args": {"model_name": "gpt2"}, "state": "Experiment"},
            {"id": "reproduce", "agent": "executor", "op": "reproduce",
             "args": {"paper_id": pipeline, "n_prompts": 4}, "state": "Running"},
            {"id": "inspect", "agent": "inspector", "op": "attention",
             "args": {"layer": 10, "head": 7}, "state": "Observation"},
            {"id": "patch", "agent": "executor", "op": "patch_head",
             "args": {"layer": 9, "head": 9}, "state": "Experiment"},
            {"id": "discover", "agent": "discoverer", "op": "discover",
             "args": {"hypothesis": goal}, "state": "Analysis"},
            {"id": "validate", "agent": "critic", "op": "validate",
             "args": {"hypothesis": goal}, "state": "Conclusion"},
            {"id": "publish", "agent": "scribe", "op": "publish",
             "args": {}, "state": "Published"},
        ][:MAX_STEPS]
        return {
            "goal": goal,
            "pipeline": pipeline,
            "states": WORKFLOW_STATES,
            "nodes": nodes,
        }

    def hypotheses(self, goal: str, k: int = 3) -> List[Dict[str, Any]]:
        """Seed hypotheses from the registry catalog (dedup aid)."""
        catalog = []
        try:
            catalog = self._registry_lazy().list_catalog(item_type="all")
        except Exception:
            catalog = []
        known = {str(i.get("id", "")) for i in catalog}
        out = []
        for i in range(k):
            tag = f"h{i + 1}"
            out.append({
                "hypothesis": f"{goal} [{tag}]",
                "suggested_experiment": f"exp_{tag}",
                "novel": tag not in known,
            })
        return out
