"""Multi-Agent Research Society.

A thin, honest facade over the real Society orchestrator.

What this used to be
--------------------
A stub that asserted an outcome it had not reached::

    def run_society_collaboration(self, goal):
        return {
            "goal": goal,
            "participating_agents_count": len(self.agents),   # 7 hardcoded dicts
            "consensus_reached": True,
            "society_status": "Completed",
        }

`self.agents` was a literal list of seven `{"role": ..., "status": "Active"}`
dicts. No agent existed, none was called, and `consensus_reached` was `True` and
`society_status` "Completed" for every goal ever passed -- including goals that
raised, and goals that named components which do not exist.

`backend/agents/society.py` says of the old standalone stub: *"The old standalone
stub remains available only from its historical module and must not be reachable
through the active Society package."* This module was exactly such a stub, and
`research_platform.autonomous` is an active package.

The damage was not contained to this return value. `ai_scientist_engine` reads
its output to build the arguments for the scientific debate, so an unconditional
`consensus_reached: True` sat upstream of the hypothesis comparison. That is now
fixed at this level too.

What it does now
----------------
Delegates to `ResearchSocietyV2`, which really runs Planner -> Executor ->
Inspector -> Discoverer -> Critic -> Scribe under `MAX_STEPS` / `MAX_REPLANS`
guardrails and a live-provenance gate. `consensus_reached` is *derived* from that
result -- it requires the run to have completed and both eligibility flags to be
set, which the gate withholds unless the discovery and validation stages carried
`provenance: "live"`.

A failure inside the society is reported as a failure, with the exception text.
It is not converted into `consensus_reached: False` with a happy-looking status,
and it is certainly not converted into a success.

Hypotheses
----------
The Society runs a *workflow*, not a debate: it produces a plan, a trace and a
gate result, not two competing claims. So this facade does not invent a
`hypothesis_a` / `hypothesis_b` pair for `ai_scientist_engine` to argue about.
Those keys are absent unless a caller supplies competing hypotheses explicitly via
`set_hypotheses`, and `ai_scientist_engine` falls back to the goal itself.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional


class MultiAgentResearchSociety:
    """Coordinates the specialized Society agents.

    Delegates the work; derives its own summary from the result.
    """

    def __init__(self, society: Optional[Any] = None) -> None:
        """`society` may be injected for testing. None constructs the real one
        lazily, so importing this module does not pull in torch/transformers."""
        self._society = society
        self._agents: Optional[List[Dict[str, str]]] = None
        self._hypotheses: Optional[Dict[str, Any]] = None

    # ── agents ────────────────────────────────────────────────────────────

    @property
    def agents(self) -> List[Dict[str, str]]:
        """The real agents, read from the orchestrator.

        Was a hardcoded list of seven dicts with `status: "Active"`, which read
        as though seven agents were running.
        """
        if self._agents is None:
            society = self._resolve()
            inner = getattr(society, "planner", None)
            agents = []
            for name in ("planner", "executor", "inspector",
                         "discoverer", "critic", "scribe"):
                component = getattr(society, name, None)
                agents.append({
                    "role": getattr(component, "role", name),
                    "agent": name,
                    "present": component is not None,
                    "class": type(component).__name__ if component else None,
                })
            self._agents = agents
        return self._agents

    def set_hypotheses(
        self,
        hypothesis_a: str,
        hypothesis_b: str,
        evidence_a: Optional[List[Dict[str, Any]]] = None,
        evidence_b: Optional[List[Dict[str, Any]]] = None,
        counter_evidence_a: Optional[List[Dict[str, Any]]] = None,
        counter_evidence_b: Optional[List[Dict[str, Any]]] = None,
        alternative_hypothesis: Optional[str] = None,
        critique_evidence: Optional[List[Dict[str, Any]]] = None,
        debate_rounds: int = 0,
    ) -> None:
        """Supply the competing hypotheses the Society does not itself generate.

        The Society runs a workflow; it does not produce two rival claims to weigh
        against each other. Anything of that kind has to come from the caller.
        """
        self._hypotheses = {
            "hypothesis_a": hypothesis_a,
            "hypothesis_b": hypothesis_b,
            "evidence_a": evidence_a,
            "evidence_b": evidence_b,
            "counter_evidence_a": counter_evidence_a,
            "counter_evidence_b": counter_evidence_b,
            "alternative_hypothesis": alternative_hypothesis,
            "critique_evidence": critique_evidence,
            "debate_rounds": debate_rounds,
        }

    # ── the run ───────────────────────────────────────────────────────────

    def _resolve(self):
        if self._society is None:
            from backend.agents.society import ResearchSocietyV2
            self._society = ResearchSocietyV2()
        return self._society

    def run_society_collaboration(self, goal: str) -> Dict[str, Any]:
        """Run the Society and report what it actually concluded."""
        try:
            society = self._resolve()
            result = society.run_blocking(goal=goal)
        except Exception as exc:
            # A failure is a failure. Reporting `consensus_reached: False` with
            # `society_status: "Completed"` -- or swallowing it -- would put a
            # success-shaped record on a run that did not happen.
            return {
                "goal": goal,
                "participating_agents_count": 0,
                "consensus_reached": False,
                "society_status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:400],
                "reason": (
                    "The Society orchestrator raised before producing a result. "
                    "No consensus can be claimed from a run that did not finish."
                ),
                "evaluated_at": _now(),
            }

        if not isinstance(result, dict):
            return {
                "goal": goal,
                "participating_agents_count": 0,
                "consensus_reached": False,
                "society_status": "unavailable",
                "reason": (
                    f"The Society orchestrator returned "
                    f"{type(result).__name__}, not a result record."
                ),
                "evaluated_at": _now(),
            }

        status = result.get("status")
        validation_ok = bool(result.get("validation_eligible", False))
        publication_ok = bool(result.get("publication_eligible", False))
        live = result.get("provenance") == "live"

        # Derived, not asserted. Every clause can withhold it: an incomplete run,
        # a stage that did not carry live provenance, a failed gate.
        consensus = bool(
            status == "completed" and live and validation_ok and publication_ok
        )

        payload: Dict[str, Any] = {
            "goal": goal,
            "participating_agents_count": len(self.agents),
            "consensus_reached": consensus,
            "society_status": status or "unknown",
            "status": status,
            "provenance": result.get("provenance"),
            "validation_eligible": validation_ok,
            "publication_eligible": publication_ok,
            "consensus_basis": (
                "run completed with live provenance and both eligibility gates set"
                if consensus else
                _why_not(status, live, validation_ok, publication_ok)
            ),
            "result": result,
            "evaluated_at": _now(),
        }

        # Pass through the full orchestrator record so a caller can see the
        # workflow, trace and gate rather than only the summary.
        for key in ("workflow", "trace", "reflection", "reproducibility",
                    "publication", "events"):
            if key in result:
                payload[key] = result[key]

        if self._hypotheses:
            payload.update(self._hypotheses)
            payload["hypotheses_source"] = "supplied by caller"
        else:
            payload["hypotheses_source"] = (
                "none: the Society runs a workflow and does not generate rival "
                "hypotheses. Call set_hypotheses() to supply them."
            )

        return payload


def _why_not(status: Any, live: bool, validation_ok: bool,
             publication_ok: bool) -> str:
    reasons = []
    if status != "completed":
        reasons.append(f"run status was {status!r}, not 'completed'")
    if not live:
        reasons.append("the result did not carry live provenance")
    if not validation_ok:
        reasons.append("validation was not eligible")
    if not publication_ok:
        reasons.append("publication was not eligible")
    return "No consensus: " + "; ".join(reasons) + "."


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()