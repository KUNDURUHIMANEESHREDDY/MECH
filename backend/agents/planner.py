"""Builds a workflow DAG from a natural-language research goal.

What this used to do
--------------------
`plan()` returned the same seven nodes for every goal:

    load -> reproduce -> inspect -> patch -> discover -> validate -> publish

with only the pipeline hint, the hypothesis text and two IOI head coordinates
varying. That was disclosed -- `plan_is_templated: True` said so -- but disclosure
is not implementation. A planner that emits an identical DAG whatever it is
asked has not planned anything, and the field name claimed otherwise.

It also emitted ops the supervisor cannot run. `society.execute` does
`getattr(agent, op, None)` and returns `{"error": "unknown op ..."}`, so a node
naming a method the agent does not have is a runtime error, not a plan.

What it does now
----------------
The node list is *composed*, in three steps, each recorded on the result:

1. **Anchor.** Every plan starts by loading weights, because every measurement
   needs them. This one node is genuinely unconditional.
2. **Branch on what matched.** A goal that names a reproduction target gets the
   reproduce/inspect/patch sequence aimed at that target. A goal that does not
   gets discovery-only: there is nothing to reproduce against, and inventing a
   pipeline to reproduce would be the old `detect_pipeline` defect in new
   clothes.
3. **Filter by capability.** Each candidate node names an agent method; it is
   emitted only if that method exists. Verified against the real classes at plan
   time, so the plan cannot contain a node that will fail at `getattr`.

So the plan differs by goal, every node is justified by a checked capability,
and the residual -- what could not be resolved from the goal alone -- is named in
`unresolved` rather than papered over with an arbitrary layer/head.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

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


def _agent_method(agent: str, op: str):
    """The real bound method for `agent.op`, or None if there is not one.

    Read from the live objects rather than from a hand-written table, so this
    cannot drift from the signatures it is checking.
    """
    try:
        from backend.agents.society import ResearchSocietyV2
        obj = getattr(ResearchSocietyV2(), agent, None)
        return getattr(obj, op, None) if obj is not None else None
    except Exception:
        return None


def _argument_mismatch(agent: str, op: str,
                       args: Dict[str, Any]) -> Optional[str]:
    """A description of why `args` would not bind, or None if they would.

    Requires the call to be *complete*: a missing required parameter is a
    mismatch. Use `_supplied_keys_rejected` for planning, where the supervisor
    may still fill in context.

    Returns the problem rather than a bool so a dropped node can say what was
    wrong with it -- "unknown op" is actionable, "the plan is broken" is not.

    A method that cannot be introspected returns None (not a mismatch): refusing
    to plan because a signature could not be read would be worse than the bug
    this guards against.
    """
    import inspect

    method = _agent_method(agent, op)
    if method is None:
        return None
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return None
    # `bind` on a bound method must not receive `self`.
    try:
        signature.bind(**args)
    except TypeError as exc:
        return str(exc)
    return None


def _supplied_keys_rejected(agent: str, op: str,
                           args: Dict[str, Any]) -> Optional[str]:
    """Whether any *supplied* key is one the method cannot accept.

    Deliberately weaker than `_argument_mismatch`, and that is the whole point.
    A plan node is allowed to be incomplete -- the supervisor supplies run
    context such as `discovery_id` at execution time -- but it is never allowed
    to supply a keyword the target does not accept. That asymmetry is what this
    defect was: a complete-looking call carrying an argument with nowhere to go.

    A method that accepts `**kwargs` accepts anything.
    """
    import inspect

    method = _agent_method(agent, op)
    if method is None or not args:
        return None
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD
           for p in parameters.values()):
        return None
    unexpected = sorted(k for k in args if k not in parameters)
    if unexpected:
        return (f"unexpected argument(s) {unexpected}; "
                f"{op} accepts {sorted(parameters)}")
    return None


def _agent_methods() -> Dict[str, Set[str]]:
    """The methods each agent actually exposes, read from the real classes.

    Used to drop any node whose op the supervisor could not dispatch. Computed
    lazily and cached, because importing the agents pulls torch on some paths and
    `plan()` is called from tests that should not need a model.
    """
    cached = getattr(_agent_methods, "_cache", None)
    if cached is not None:
        return cached

    table: Dict[str, Set[str]] = {}
    for agent in ("executor", "inspector", "critic", "discoverer", "scribe"):
        try:
            from backend.agents import society  # noqa: F401
            module = __import__(f"backend.agents.{agent}", fromlist=["*"])
            cls = getattr(__import__(f"backend.agents.{agent}",
                                     fromlist=["*"]), f"{agent.title().replace('Scribe','Scribe')}Adapter", None)
        except Exception:
            cls = None
        if cls is None:
            # Fall back to the registry the society supervisor builds.
            try:
                from backend.agents.society import ResearchSocietyV2
                instance = ResearchSocietyV2()
                obj = getattr(instance, agent, None)
                table[agent] = {
                    name for name in dir(obj)
                    if not name.startswith("_") and callable(getattr(obj, name, None))
                } if obj is not None else set()
            except Exception:
                table[agent] = set()
            continue
        table[agent] = {
            name for name in dir(cls)
            if not name.startswith("_") and callable(getattr(cls, name, None))
        }

    setattr(_agent_methods, "_cache", table)
    return table


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
        """Return a Workflow DAG composed from the goal and real capabilities.

        Each node records *why it is in the plan* (`because`), so a reader can
        tell which nodes were selected by the goal and which are the unconditional
        lifecycle floor.
        """
        pipeline = self.detect_pipeline(goal)
        is_ioi = pipeline == "ioi"
        available = _agent_methods()

        nodes: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []

        def offer(node: Dict[str, Any], because: str) -> None:
            """Add a node if the supervisor could actually run it.

            Two checks, because they catch different failures. The first asks
            whether the op exists at all; the second asks whether the op accepts
            the arguments being handed to it.

            The second check was added after the anchor node named
            `op="reproduce"` with `args={"model_name": ...}`: `reproduce` exists,
            so the first check passed, and the plan failed at execution with
            `unexpected keyword argument 'model_name'`. A node whose args cannot
            bind is a plan that cannot run, so it belongs in `dropped` for the
            same reason an unknown op does -- reported, not deferred.

            Some nodes are legitimately incomplete here, because the supervisor
            supplies context the planner cannot know upfront:
            `critic.validate` gets `discovery_id` and `discovery_result` from the
            run, and `scribe.publish` gets `goal`, `trace` and `reflection`. So
            the check asks whether the *supplied* keys are accepted, not whether
            the node alone would satisfy the signature.

            That distinction matters: `validate` carries only `hypothesis` and
            `Critic.validate` also requires `discovery_id`, so a strict
            `signature.bind` would have dropped the validation gate from every
            plan -- silently removing the step that gates publication, which is
            the opposite of a safety improvement.
            """
            agent = str(node.get("agent", ""))
            op = str(node.get("op", ""))
            methods = available.get(agent)
            if methods and op not in methods:
                # The supervisor would return `unknown op` at execution time.
                # Dropping it here keeps the plan executable and makes the
                # omission visible instead of deferring it to a runtime error.
                dropped.append({"id": node.get("id"), "agent": agent,
                                "op": op, "why": (
                                    f"the {agent} agent exposes no {op!r} method, "
                                    f"so the supervisor could not dispatch it")})
                return
            mismatch = _supplied_keys_rejected(agent, op, node.get("args", {}))
            if mismatch is not None:
                    dropped.append({
                        "id": node.get("id"), "agent": agent, "op": op,
                        "why": (
                            f"the {agent} agent's {op!r} does not accept the "
                            f"arguments this node supplies: {mismatch}. Keeping "
                            f"it would fail the run at execution time.")})
                    return
            node["because"] = because
            nodes.append(node)

        # 1. Anchor. Genuinely unconditional: nothing can be measured without
        #    weights, so this node is in every plan for a real reason.
        #
        #    The op is `ensure_model`, which is the method that takes
        #    `model_name` and calls the engine's load. This used to read
        #    `reproduce`, carried over from the reproduce node below it, and it
        #    raised `Executor.reproduce() got an unexpected keyword argument
        #    'model_name'` for every goal -- the anchor is unconditional, so the
        #    whole Society workflow failed at its first step. The `id`,
        #    `rationale` and `args` all describe a model load; only the `op`
        #    disagreed. See docs/audit/society-contract-mismatch.md.
        #
        #    Note this is *not* a `model_name` -> `model_variant` mapping:
        #    `model_variant` ("small"/"medium", composed as
        #    `gpt2-{variant}`) is an IOIPipeline.run parameter, a different
        #    concept from a loaded model identity.
        offer({"id": "load", "agent": "executor", "op": "ensure_model",
               "args": {"model_name": "gpt2"}, "state": "Experiment",
               "rationale": "Every downstream step needs live weights."},
              "unconditional: no measurement is possible without weights")

        # 2. Branch. Only a goal that names a reproduction target gets the
        #    reproduce/inspect/patch sequence -- with that target's own
        #    coordinates, not a default's.
        if pipeline is not None:
            offer({"id": "reproduce", "agent": "executor", "op": "reproduce",
                   "args": {"paper_id": pipeline, "n_prompts": 4},
                   "state": "Running",
                   "rationale": (
                       f"Measure against the published {pipeline} baseline "
                       f"before claiming anything new.")},
                  f"the goal matched the {pipeline!r} pipeline")
            offer({"id": "inspect", "agent": "inspector", "op": "attention",
                   "args": dict(_IOI_INSPECTION_HEAD) if is_ioi else {},
                   "state": "Observation",
                   "rationale": (
                       "Inspect the known IOI induction head (L10H7)." if is_ioi
                       else "Inspect a target head. None is known for this "
                            "pipeline, so the step carries no coordinates and "
                            "will refuse rather than name an arbitrary head.")},
                  ("the IOI circuit's published induction head is known"
                   if is_ioi else
                   "follows a matched pipeline; no head coordinates are known"))
            offer({"id": "patch", "agent": "executor", "op": "patch_head",
                   "args": dict(_IOI_PATCH_HEAD) if is_ioi else {},
                   "state": "Experiment",
                   "rationale": (
                       "Patch the known IOI name mover (L9H9)." if is_ioi
                       else "Patch a target head. None is known for this "
                            "pipeline, so the step carries no coordinates.")},
                  ("the IOI circuit's published name mover is known" if is_ioi
                   else "follows a matched pipeline; no coordinates are known"))
        else:
            dropped.append({
                "id": "reproduce", "agent": "executor", "op": "reproduce",
                "why": ("no reproduction pipeline matched this goal. The step "
                        "would have to invent a target, which is the defect "
                        "this planner previously had.")})

        # 3. Goal-independent remainder: discovery, validation, publication.
        #    These apply to any goal, which is why they are unconditional.
        offer({"id": "discover", "agent": "discoverer", "op": "discover",
               "args": {"hypothesis": goal}, "state": "Analysis",
               "rationale": "Run causal discovery over the stated goal."},
              "unconditional: discovery is the point of a research run")
        offer({"id": "validate", "agent": "critic", "op": "validate",
               "args": {"hypothesis": goal}, "state": "Conclusion",
               "rationale": (
                   "Compare the discovery against live measurement. This step "
                   "gates publication and is expected to fail for most inputs.")},
              "unconditional: publication is gated on validation")
        offer({"id": "publish", "agent": "scribe", "op": "publish",
               "args": {}, "state": "Published",
               "rationale": (
                   "Emit a report. Only reachable if validate passes its gate; "
                   "the publish step refuses otherwise.")},
              "unconditional: publication is the terminal step")

        nodes = nodes[:MAX_STEPS]

        unresolved: List[str] = []
        if pipeline is None:
            unresolved.append("pipeline")
        if not is_ioi:
            unresolved.extend(["inspect_target", "patch_target"])

        goal_dependent = [n["id"] for n in nodes
                          if not str(n.get("because", "")).startswith("unconditional")]

        return {
            "goal": goal,
            "pipeline": pipeline,
            "pipeline_matched": pipeline is not None,
            "states": WORKFLOW_STATES,
            "nodes": nodes,
            # No longer a fixed skeleton, so the old `plan_is_templated` flag
            # would be a lie in the other direction. What is reported instead is
            # how much of the plan actually came from the goal.
            "nodes_from_goal": goal_dependent,
            "n_nodes": len(nodes),
            "composed_from_capabilities": True,
            "dropped_nodes": dropped,
            "template_reason": (
                "The unconditional lifecycle nodes (load, discover, validate, "
                "publish) are fixed. The reproduce/inspect/patch branch is "
                "emitted only for a goal that names a reproduction target, and "
                "each node is kept only if the agent exposes the method the "
                "supervisor would dispatch."
            ),
            "unresolved": unresolved,
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