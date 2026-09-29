"""Research Society v2 — Supervisor (ReAct: plan -> dispatch -> execute -> reflect).

Replaces backend/agents/research_society.py::ResearchSociety.
run_society_collaboration (stub: hardcoded "Step 1/Step 2", fake 0.95 scores)
with a real loop over the six Society agents in this package:

    Planner -> Executor -> Inspector -> Discoverer -> Critic -> Scribe

Guardrails: MAX_STEPS=8 nodes per plan, MAX_REPLANS=2, fail-fast when a
stage returns an error, unavailable, or blocked status. Scientific stages
also require explicit live provenance before they can publish.

All cross-module imports are lazy (inside methods) so importing this module
never requires torch/transformers. Agent-to-op dispatch falls back to
CAPABILITY_MAP keyword matching when no LLM dispatcher is configured.

Emits backend.core.event_schema.ResearchEvent (valid types only) into the
returned trace — the per-run evidence trail.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from .evidence_policy import (
    blocked_reason,
    discovery_is_live,
    field_map,
    provenance_of,
    validation_is_live,
)

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


def _response_field_provenance(response: Dict[str, Any]) -> Dict[str, str]:
    existing = response.get("field_provenance")
    if isinstance(existing, dict):
        return {str(key): str(value) for key, value in existing.items()}
    result = response.get("result")
    label = provenance_of(result if isinstance(result, dict) else response)
    return field_map(("status", "result", "reason", "error"), label)


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
            args.setdefault("discovery_result", ctx.get("discovery_result", {}))
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
        return res.get("status") in ("error", "unavailable", "failed", "blocked")

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

    @staticmethod
    def _emit_plugin_hook(hook_name: str, payload: Dict[str, Any]) -> None:
        """Fan a platform event out to enabled plugins. Never fails a run."""
        try:
            from backend.plugins.service import get_service
            get_service().bus().emit(hook_name, payload)
        except Exception:
            pass

    async def run(self, goal: str,
                  model_name: str = "gpt2",
                  on_event: Any = None,
                  run_id: str = "") -> Dict[str, Any]:
        events: List[Dict[str, Any]] = []
        self._emit(events, on_event, "ResearchStarted", {"goal": goal})
        self._emit_plugin_hook("on_campaign_started",
                               {"campaign_id": run_id or "run_unknown",
                                "goal": goal, "model_name": model_name})
        workflow = self.planner.plan(goal)
        workflow["provenance"] = "reference"
        workflow["field_provenance"] = field_map(
            ("goal", "pipeline", "states", "nodes"), "reference"
        )
        nodes = list(workflow.get("nodes", []))[:MAX_STEPS]
        self._emit(events, on_event, "ExperimentQueued",
                   {"plan": [n["id"] for n in nodes]})

        trace: List[Dict[str, Any]] = []
        ctx: Dict[str, Any] = {"goal": goal, "model_name": model_name,
                               "trace": trace}
        for node in nodes:
            res = await self._run_node(node, ctx)
            step = {"node": node["id"], "agent": self.dispatch(node), **res}
            step["field_provenance"] = _response_field_provenance(res)
            trace.append(step)
            ctx["trace"] = trace
            if node["id"] == "discover":
                disc = res.get("result")
                if not isinstance(disc, dict):
                    disc = {}
                if not disc:
                    # Discoverer now spreads live fields at top level; use res directly
                    disc = res
                ctx["discovery_result"] = disc
                if res.get("discovery_id"):
                    ctx["discovery_id"] = res["discovery_id"]
                if (res.get("status") == "completed"
                        and discovery_is_live(disc)
                        and disc.get("discovery_id")):
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

        paper_id = workflow.get("pipeline", "ioi")
        discovery_step = next(
            (t for t in trace if t.get("node") == "discover"), {})
        # Trace steps store discoverer response fields directly (spread), not in "result"
        discovery_result = discovery_step if isinstance(discovery_step, dict) else {}
        discovery_eligible = discovery_is_live(discovery_result)

        # Do not run or publish reproduction evidence when discovery is not
        # explicitly live.  This keeps synthetic DiscoveryEngine fields out
        # of the validation/publication chain rather than merely hiding them.
        if discovery_eligible:
            # Bounded panel: the trace already ran a live reproduction smoke
            # test; this independent run feeds the publication gate.
            repro = await asyncio.to_thread(
                self.critic.reproduce, paper_id, 8)
        else:
            repro = {
                "status": "blocked",
                "provenance": provenance_of(discovery_result),
                "field_provenance": field_map(
                    ("status", "reason", "observed_metrics", "report", "gate"),
                    provenance_of(discovery_result),
                ),
                "publication_eligible": False,
                "reason": blocked_reason(discovery_result, "Discovery"),
            }
        ctx["reproducibility"] = repro

        valid_step = next((t for t in trace if t.get("node") == "validate"),
                          {})
        vres = valid_step.get("result", {}) if isinstance(valid_step, dict) \
            else {}
        if not isinstance(vres, dict):
            vres = {}
        validation_eligible = validation_is_live(vres)
        confident = self.critic.is_confident(vres)
        if discovery_eligible:
            gate = dict(repro.get("gate", {})) if isinstance(repro, dict) else {}
            gate.update({
                "status": "completed" if repro.get("status") == "completed"
                          else "unavailable",
                "provenance": "live" if repro.get("provenance") == "live"
                              else "unavailable",
                "field_provenance": field_map(
                    ("status", "threshold", "value", "passed", "confidence", "validated"),
                    "live" if repro.get("provenance") == "live" else "unavailable",
                ),
                "confidence": self._confidence_of(vres),
                "validated": bool(vres.get("validated", False)),
                "passed": bool(gate.get("passed", False))
                          and confident and validation_eligible,
            })
        else:
            gate = {
                "status": "blocked",
                "provenance": "unavailable",
                "field_provenance": field_map(
                    ("status", "validated", "passed", "reason"), "unavailable"
                ),
                "validated": False,
                "passed": False,
                "reason": blocked_reason(discovery_result, "Discovery"),
            }
        ctx["gate"] = gate
        self._emit(events, on_event, "CircuitValidated",
                   {"successful": successful, "failed": failed,
                    "status": "completed" if validation_eligible else "blocked",
                    "gate_passed": gate["passed"],
                    "fidelity_pct": gate.get("value"),
                    "reason": gate.get("reason", "")})

        needs_replan = bool(failed) and self.replans < MAX_REPLANS
        if needs_replan and not any(t["node"] == "validate" and not self._failed(t)
                                    for t in trace):
            self.replans += 1
            # Narrow retry: re-run only the failed scope would go here;
            # v1 records the replan and does not publish partial evidence.
            self._emit(events, on_event, "ExperimentQueued",
                       {"replan": self.replans, "failed": failed})

        publication = self.scribe.publish(goal=goal, trace=trace,
                                          reflection=reflection,
                                          run_id=run_id,
                                          reproducibility=repro,
                                          gate=gate)
        if publication.get("status") == "completed":
            self._emit(events, on_event, "PublicationGenerated",
                       {"experiment_id": publication.get("experiment_id")})
        else:
            self._emit(events, on_event, "HypothesisRejected",
                       {"node": "publish",
                        "reason": str(publication.get("reason")
                                      or "Publication evidence is unavailable")
                        [:200]})
        self._emit(events, on_event, "ResearchFinished",
                   {"steps_completed": publication.get("steps_completed")
                    or f"{len(successful)}/{len(trace)}"})
        self._emit_plugin_hook("on_campaign_completed",
                               {"campaign_id": run_id or "run_unknown",
                                "goal": goal,
                                "status": publication.get("status", "")})

        if publication.get("status") == "blocked":
            status = "blocked"
        elif failed:
            status = "failed"
        elif successful:
            status = "completed"
        else:
            status = "failed"
        return {
            "status": status,
            "provenance": publication.get("provenance", "unavailable"),
            "field_provenance": {
                "workflow": workflow.get("provenance", "reference"),
                "trace": "live" if status == "completed" else "unavailable",
                "reflection": reflection.get("provenance", "unavailable"),
                "reproducibility": repro.get("provenance", "unavailable"),
                "gate": gate.get("provenance", "unavailable"),
                "publication": publication.get("provenance", "unavailable"),
            },
            "validation_eligible": bool(publication.get("validation_eligible", False)),
            "publication_eligible": bool(publication.get("publication_eligible", False)),
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


# Keep the legacy import name, but point it at the guarded supervisor.  The
# old standalone stub remains available only from its historical module and
# must not be reachable through the active Society package.
ResearchSociety = ResearchSocietyV2
