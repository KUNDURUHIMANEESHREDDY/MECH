"""Research Society v2 — Supervisor (ReAct: plan -> dispatch -> execute -> reflect).

Replaces backend/agents/research_society.py::ResearchSociety.
run_society_collaboration (stub: hardcoded "Step 1/Step 2", fake 0.95 scores)
with a real loop over the six Society agents in this package:

    Planner -> Executor -> Inspector -> Discoverer -> Critic -> Scribe

Guardrails: MAX_STEPS=8 nodes per plan, MAX_REPLANS=2, fail-fast when a
stage returns {"status": "error"} with confidence below threshold.

All cross-module imports are lazy (inside methods) so importing this module
never requires torch/transformers. Agent-to-op dispatch falls back to
CAPABILITY_MAP keyword matching when no LLM dispatcher is configured.

Emits backend.core.event_schema.ResearchEvent (valid types only) into the
returned trace — the per-run evidence trail.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

MAX_STEPS = 8
MAX_REPLANS = 2

# Node-id keyword -> agent attribute on this supervisor (fallback dispatch).
CAPABILITY_MAP = {
    "load": "executor", "prompt": "executor", "patch": "executor",
    "reproduce": "executor", "orchestrat": "executor", "infer": "executor",
    "ioi": "executor",
    "neuron": "inspector", "attention": "inspector", "residual": "inspector",
    "layer": "inspector", "logit": "inspector", "inspect": "inspector",
    "head": "inspector",
    "discover": "discoverer", "circuit": "discoverer",
    "genealogy": "discoverer", "causal": "discoverer",
    "attribution": "discoverer", "sae": "discoverer",
    "valid": "critic", "benchmark": "critic", "confidence": "critic",
    "evidence": "critic", "reflect": "critic", "calibrat": "critic",
    "report": "scribe", "provenance": "scribe", "lineage": "scribe",
    "dashboard": "scribe", "publish": "scribe",
    "plan": "planner", "workflow": "planner", "hypothesis": "planner",
    "dsl": "planner", "catalog": "planner",
}


def _event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from backend.core.event_schema import ResearchEvent
        return ResearchEvent(event_type, payload).to_dict()
    except Exception as exc:  # invalid type should never kill a run
        return {"event_type": event_type, "payload": payload,
                "error": str(exc)[:200]}


class ResearchSocietyV2:
    """Central orchestrator managing the six specialized Society agents."""

    def __init__(self) -> None:
        from .critic import Critic
        from .discoverer import Discoverer
        from .executor import Executor
        from .inspector import Inspector
        from .planner import Planner
        from .scribe import Scribe
        self.planner = Planner()
        self.executor = Executor()
        self.inspector = Inspector()
        self.discoverer = Discoverer()
        self.critic = Critic()
        self.scribe = Scribe()
        self.replans = 0

    # -- dispatch ------------------------------------------------------
    def dispatch(self, node: Dict[str, Any]) -> str:
        explicit = str(node.get("agent", "")).strip().lower()
        if explicit in ("planner", "executor", "inspector",
                        "discoverer", "critic", "scribe"):
            return explicit
        blob = f"{node.get('id', '')} {node.get('op', '')}".lower()
        for kw, agent in CAPABILITY_MAP.items():
            if kw in blob:
                return agent
        return "executor"  # safe default: executor degrades gracefully

    # -- stage execution ------------------------------------------------
    async def _run_node(self, node: Dict[str, Any],
                        ctx: Dict[str, Any]) -> Dict[str, Any]:
        agent_name = self.dispatch(node)
        agent = getattr(self, agent_name)
        op = str(node.get("op", ""))
        args = dict(node.get("args", {}))
        # Provide cross-stage context the planner cannot know upfront.
        if agent_name == "critic" and op == "validate":
            args.setdefault("discovery_id",
                            ctx.get("discovery_id", "disc_unknown"))
        if agent_name == "scribe" and op == "publish":
            return self.scribe.publish(
                goal=ctx.get("goal", ""), trace=ctx.get("trace", []),
                reflection=ctx.get("reflection", {}))
        fn = getattr(agent, op, None)
        if fn is None:
            return {"status": "error",
                    "error": f"unknown op '{op}' for agent '{agent_name}'"}
        if asyncio.iscoroutinefunction(fn):
            return await fn(**args)
        return await asyncio.to_thread(fn, **args)

    @staticmethod
    def _failed(res: Dict[str, Any]) -> bool:
        return res.get("status") in ("error", "unavailable", "failed")

    @staticmethod
    def _confidence_of(vres: Any) -> float:
        try:
            return float((vres.get("confidence") or {})
                         .get("confidence_score", 0.0))
        except Exception:
            return 0.0

    # -- main loop ------------------------------------------------------
    def _emit(self, events: List[Dict[str, Any]], on_event: Any,
              event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Append a ResearchEvent and forward it to the live hook (SSE)."""
        ev = _event(event_type, payload)
        events.append(ev)
        if on_event is not None:
            try:
                on_event(ev)
            except Exception:
                pass
        return ev

    async def run(self, goal: str,
                  model_name: str = "gpt2",
                  on_event: Any = None,
                  run_id: str = "") -> Dict[str, Any]:
        events: List[Dict[str, Any]] = []
        self._emit(events, on_event, "ResearchStarted", {"goal": goal})
        workflow = self.planner.plan(goal)
        nodes = list(workflow.get("nodes", []))[:MAX_STEPS]
        self._emit(events, on_event, "ExperimentQueued",
                   {"plan": [n["id"] for n in nodes]})

        trace: List[Dict[str, Any]] = []
        ctx: Dict[str, Any] = {"goal": goal, "model_name": model_name,
                               "trace": trace}
        for node in nodes:
            res = await self._run_node(node, ctx)
            step = {"node": node["id"], "agent": self.dispatch(node), **res}
            trace.append(step)
            ctx["trace"] = trace
            if node["id"] == "discover" and res.get("status") == "completed":
                disc = res.get("result", {})
                if isinstance(disc, dict) and disc.get("discovery_id"):
                    ctx["discovery_id"] = disc["discovery_id"]
                    self._emit(events, on_event, "DiscoveryCreated",
                               {"discovery_id": disc["discovery_id"]})
            if self._failed(res):
                self._emit(events, on_event, "HypothesisRejected",
                           {"node": node["id"],
                            "reason": str(res.get("error") or res.get("reason", ""))[:200]})
                break  # fail-fast: GPU/executor failure poisons later stages

        # Reflection (Critic) — one replan allowed when confidence is low.
        successful = [t["node"] for t in trace if not self._failed(t)]
        failed = [t["node"] for t in trace if self._failed(t)]
        reflection = self.critic.reflect(
            campaign_id=f"camp_{abs(hash(goal)) % 10000:04d}",
            successful=successful, failed=failed,
            planner_decisions=[n["id"] for n in nodes])
        # Closed validation loop: live reproduction vs published baselines.
        # Runs in a thread (torch forwards) so the event loop stays free.
        paper_id = workflow.get("pipeline", "ioi")
        repro = await asyncio.to_thread(self.critic.reproduce, paper_id)
        ctx["reproducibility"] = repro
        valid_step = next((t for t in trace if t.get("node") == "validate"),
                          {})
        vres = valid_step.get("result", {}) if isinstance(valid_step, dict) \
            else {}
        confident = self.critic.is_confident(
            vres if isinstance(vres, dict) else {})
        gate = dict(repro.get("gate", {})) if isinstance(repro, dict) else {}
        gate.update({
            "confidence": self._confidence_of(vres),
            "validated": bool(vres.get("validated", False))
            if isinstance(vres, dict) else False,
            "passed": bool(gate.get("passed", False)) and confident,
        })
        ctx["gate"] = gate
        self._emit(events, on_event, "CircuitValidated",
                   {"successful": successful, "failed": failed,
                    "gate_passed": gate["passed"],
                    "fidelity_pct": gate.get("value")})

        needs_replan = bool(failed) and self.replans < MAX_REPLANS
        if needs_replan and not any(t["node"] == "validate" and not self._failed(t)
                                    for t in trace):
            self.replans += 1
            # Narrow retry: re-run only the failed scope would go here;
            # v1 records the replan and proceeds to publish partial results.
            self._emit(events, on_event, "ExperimentQueued",
                       {"replan": self.replans, "failed": failed})

        publication = self.scribe.publish(goal=goal, trace=trace,
                                          reflection=reflection,
                                          run_id=run_id,
                                          reproducibility=repro,
                                          gate=gate)
        self._emit(events, on_event, "PublicationGenerated",
                   {"experiment_id": publication.get("experiment_id")})
        self._emit(events, on_event, "ResearchFinished",
                   {"steps_completed": publication.get("steps_completed")})
        return {
            "status": "completed" if successful else "failed",
            "goal": goal,
            "workflow": workflow,
            "trace": trace,
            "reflection": reflection,
            "publication": publication,
            "events": events,
        }

    def run_blocking(self, goal: str,
                     model_name: str = "gpt2",
                     on_event: Any = None,
                     run_id: str = "") -> Dict[str, Any]:
        """Sync entry point for FastAPI handlers (dispatcher POST /society/run)."""
        return asyncio.run(self.run(goal, model_name=model_name,
                                    on_event=on_event, run_id=run_id))


# Backwards-compatible alias: legacy_dispatcher imports ResearchSociety.
# Keep the old stub class importable; new code should use ResearchSocietyV2.
try:
    from .research_society import ResearchSociety  # noqa: F401
except Exception:  # pragma: no cover
    ResearchSociety = ResearchSocietyV2  # type: ignore
