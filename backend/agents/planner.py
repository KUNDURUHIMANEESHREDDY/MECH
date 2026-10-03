"""Society Planner agent (capability: plan).

Decomposes a user goal into a Workflow DAG (max 8 nodes) following the
lifecycle: Draft -> Question -> Hypothesis -> Experiment -> Running ->
Observation -> Analysis -> Conclusion -> Published.

Wraps (lazily, so import never requires torch):
- backend.core.unified_registry.UnifiedRegistry  (avoid duplicate hypotheses)
- backend.core.experiment_templates.ExperimentTemplatesSystem (if present)

Legacy stub replaced: backend/agents/research_society.py::Planner.create_plan
which returned hardcoded ["Step 1 ...", "Step 2 ..."] strings.

Honesty note on `plan()`. This replaces a constant, but it is still a template,
not a planner: it cannot reason about what a given goal requires. What changed
is that the template is now *labelled* as one. `Planner.plan` returns
`plan_is_templated: True` and `template_reason`, and the per-node `rationale`
fields say why each step is present rather than implying the steps were derived
from this goal. Layer and head numbers are the known IOI circuit heads
(10/7 and 9/9) and are reported as such -- carrying them for a non-IOI goal
was the worst part of the old behaviour, so they are now omitted when the
pipeline is not IOI instead of being silently wrong.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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

# Known IOI circuit heads. Attached only to an IOI plan; for any other pipeline
# they would be invented positions presented as targets.
_IOI_INSPECTION_HEAD = {"layer": 10, "head": 7}
_IOI_PATCH_HEAD = {"layer": 9, "head": 9}


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

    def detect_pipeline(self, goal: str) -> Optional[str]:
        """The reproduction pipeline this goal matches, or None.

        Previously returned "ioi" when nothing matched. That meant
        "investigate protein folding" produced a plan to reproduce indirect
        object identification, with IOI attention heads attached -- confidently
        wrong rather than obviously unmatched.
        """
        gl = goal.lower()
        for kw, pipeline in GOAL_HINTS.items():
            if kw in gl:
                return pipeline
        return None

    def known_papers(self) -> List[Dict[str, Any]]:
        try:
            return list(self._registry_lazy().list_catalog(item_type="papers"))
        except Exception:
            return []

    def plan(self, goal: str) -> Dict[str, Any]:
        """Return a Workflow DAG dict with at most MAX_STEPS nodes.

        The DAG shape is a fixed lifecycle skeleton, not a decomposition of
        `goal`. That is stated in the response rather than implied away.
        """
        pipeline = self.detect_pipeline(goal)
        is_ioi = pipeline == "ioi"

        nodes: List[Dict[str, Any]] = [
            {"id": "load", "agent": "executor", "op": "ensure_model",
             "args": {"model_name": "gpt2"}, "state": "Experiment",
             "rationale": "Every downstream step needs live weights."},
            {"id": "reproduce", "agent": "executor", "op": "reproduce",
             "args": {"paper_id": pipeline, "n_prompts": 4}, "state": "Running",
             "rationale": (
                 f"Measure against the published {pipeline} baseline before "
                 "claiming anything new." if pipeline else
                 "No reproduction target matched this goal, so this step is a "
                 "placeholder and will fail closed rather than guess."
             )},
            {"id": "inspect", "agent": "inspector", "op": "attention",
             "args": dict(_IOI_INSPECTION_HEAD) if is_ioi else {},
             "state": "Observation",
             "rationale": (
                 "Inspect the known IOI induction head (L10H7)." if is_ioi else
                 "No target head is known for this goal; supply one. An "
                 "arbitrary layer/head would be a fabricated target."
             )},
            {"id": "patch", "agent": "executor", "op": "patch_head",
             "args": dict(_IOI_PATCH_HEAD) if is_ioi else {},
             "state": "Experiment",
             "rationale": (
                 "Patch the known IOI name mover (L9H9)." if is_ioi else
                 "No target head is known for this goal; supply one."
             )},
            {"id": "discover", "agent": "discoverer", "op": "discover",
             "args": {"hypothesis": goal}, "state": "Analysis",
             "rationale": "Run causal discovery over the stated goal."},
            {"id": "validate", "agent": "critic", "op": "validate",
             "args": {"hypothesis": goal}, "state": "Conclusion",
             "rationale": (
                 "Compare the discovery against live measurement. This step "
                 "gates publication and is expected to fail for most inputs."
             )},
            {"id": "publish", "agent": "scribe", "op": "publish",
             "args": {}, "state": "Published",
             "rationale": (
                 "Emit a report. Only reachable if validate passes its gate; "
                 "the publish step refuses otherwise."
             )},
        ][:MAX_STEPS]

        unmatched = pipeline is None
        return {
            "goal": goal,
            "pipeline": pipeline,
            "pipeline_matched": not unmatched,
            "states": WORKFLOW_STATES,
            "nodes": nodes,
            # Stated plainly: this is a lifecycle skeleton, not a plan derived
            # from the goal. Callers that need real decomposition should not
            # read the node list as one.
            "plan_is_templated": True,
            "template_reason": (
                "This planner returns a fixed lifecycle skeleton. It does not "
                "decompose the goal; only the pipeline hint, the hypothesis "
                "text, and the IOI head coordinates are goal-dependent."
            ),
            "unresolved": (
                ["pipeline", "inspect_target", "patch_target"] if unmatched
                else ([] if is_ioi else ["inspect_target", "patch_target"])
            ),
            "provenance": "reference",
            "validation_eligible": False,
            "publication_eligible": False,
        }

    def hypotheses(self, goal: str, k: int = 3) -> List[Dict[str, Any]]:
        """Seed hypotheses from the registry catalog (dedup aid).

        Previously generated `f"{goal} [h1]"`, `f"{goal} [h2]"`, ... and
        reported `novel: True` because the string "h1" is never in the catalog.
        That is not a hypothesis -- it is the goal with a suffix -- and the
        novelty flag was true by construction rather than by comparison.

        Returns the catalog's own entries where they are relevant to the goal,
        and reports nothing rather than inventing when there are none.
        """
        catalog: List[Dict[str, Any]] = []
        try:
            catalog = list(self._registry_lazy().list_catalog(item_type="all"))
        except Exception:
            catalog = []

        tokens = {t for t in goal.lower().split() if len(t) > 3}
        related = [
            item for item in catalog
            if tokens & {t for t in str(item.get("title", "")).lower().split()
                         if len(t) > 3}
        ][:k]

        return [
            {
                "hypothesis": str(item.get("title") or item.get("id")),
                "source": "registry_catalog",
                "id": item.get("id"),
                # Novelty is a claim about the catalog, so it is only made when
                # the entry demonstrably came from it.
                "novel": None,
                "derived_from_goal": True,
            }
            for item in related
        ]
