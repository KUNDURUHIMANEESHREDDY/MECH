"""Society Scribe agent (capability: publish).

Turns an explicitly live run trace into durable artifacts. Wraps:
- backend.services.report_service.ReportService for a neutral transport report
- backend.core.evidence_graph.TraceableEvidenceGraph for per-run graphs
- backend.knowledge_graph.graph_store.GraphStore for persistent write-back

Stages without explicit live provenance are blocked before any report, graph,
or knowledge-base promotion occurs.

Legacy stubs replaced: _mechanistic_reports ("IOI Circuit Report..."),
_dashboard_summary ({goals:3, discoveries:14}), _evidence_rank,
_confidence_score — all hardcoded literals, now built from real outputs.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from .evidence_policy import publication_block_reason


def _first_score(result: Any, keys: Any) -> float:
    """First nonzero numeric score, scanning one nested dict level."""
    candidates = [result]
    if isinstance(result, dict):
        candidates.extend(v for v in result.values()
                          if isinstance(v, dict))
    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        for key in keys:
            try:
                value = float(obj.get(key) or 0.0)
            except Exception:
                continue
            if value:
                return value
    return 0.0


class Scribe:
    """Publishes mechanistic reports from run traces."""

    capability = "publish"

    def evidence(self, run_id: str = "", goal: str = "",
                  trace: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Build an evidence graph only from an eligible live trace."""
        if trace is None:
            return {
                "status": "blocked",
                "provenance": "unavailable",
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": "A run trace is required; reference/demo graphs cannot be published.",
            }
        reason = publication_block_reason(trace)
        if reason:
            return {
                "status": "blocked",
                "provenance": "unavailable",
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": reason,
            }
        try:
            from backend.core.evidence_graph import TraceableEvidenceGraph
            graph = TraceableEvidenceGraph.from_run(
                run_id or "run_unknown", goal, trace)
            return {"status": "completed", "provenance": "live",
                    "result": graph.to_dict()}
        except Exception as exc:
            return {"status": "error", "provenance": "unavailable",
                    "error": str(exc)[:500]}

    def report(self, experiment_id: str, title: str,
               provenance: str = "unavailable") -> Dict[str, Any]:
        try:
            from backend.services.report_service import ReportService
            res = ReportService().generate_report(
                experiment_id=experiment_id, title=title,
                provenance=provenance)
            return {"status": "completed", "provenance": provenance,
                    "result": res}
        except Exception as exc:
            return {"status": "error", "provenance": "unavailable",
                    "error": str(exc)[:500]}

    def publish(self, goal: str, trace: List[Dict[str, Any]],
                reflection: Dict[str, Any],
                run_id: str = "",
                reproducibility: Optional[Dict[str, Any]] = None,
                gate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Publish only a complete, explicitly live evidence chain."""
        reason = publication_block_reason(trace, reproducibility, gate)
        if reason:
            return {
                "status": "blocked",
                "provenance": "unavailable",
                "validation_eligible": False,
                "publication_eligible": False,
                "goal": goal,
                "reason": reason,
                "steps_completed": f"{sum(1 for t in trace if t.get('status') in ('completed', 'ok', 'loaded'))}/{len(trace)}",
            }

        exp_id = f"exp_{abs(hash(goal)) % 10000:04d}"
        rep = self.report(experiment_id=exp_id,
                          title=f"Mechanistic Report: {goal[:60]}",
                          provenance="live")
        ev = self.evidence(run_id=run_id or exp_id, goal=goal, trace=trace)
        if rep.get("status") != "completed" or ev.get("status") != "completed":
            return {
                "status": "blocked",
                "provenance": "unavailable",
                "validation_eligible": False,
                "publication_eligible": False,
                "goal": goal,
                "reason": "Report or evidence generation did not return a live artifact.",
                "steps_completed": f"{sum(1 for t in trace if t.get('status') in ('completed', 'ok', 'loaded'))}/{len(trace)}",
            }
        ok_steps = sum(1 for t in trace
                       if t.get("status") in ("completed", "ok", "loaded"))
        publication = {
            "status": "completed",
            "provenance": "live",
            "validation_eligible": True,
            "publication_eligible": True,
            "goal": goal,
            "experiment_id": exp_id,
            "steps_completed": f"{ok_steps}/{len(trace)}",
            "report": rep.get("result", rep),
            "evidence_graph": ev.get("result", ev),
            "reflection": reflection,
            "reproducibility": reproducibility or {},
            "gate": gate or {},
            "published_at": _dt.datetime.utcnow().isoformat() + "Z",
        }
        publication["knowledge_writeback"] = self.write_back(
            run_id or exp_id, goal, trace, reflection, publication)
        return publication

    def write_back(self, run_id: str, goal: str,
                   trace: List[Dict[str, Any]],
                   reflection: Dict[str, Any],
                   publication: Dict[str, Any],
                   store: Any = None) -> Dict[str, Any]:
        """Persist run discoveries to the knowledge graph (GraphStore JSON).

        Never raises: KG write-back must not fail a run. Returns a summary
        of stored node ids (empty when nothing measurable was found).
        """
        if not isinstance(publication, dict) or publication.get("status") != "completed":
            return {
                "stored": [],
                "status": "blocked",
                "provenance": "unavailable",
                "reason": "Knowledge write-back requires a completed live publication.",
            }
        publication_repro = (publication.get("reproducibility")
                             if isinstance(publication, dict) else None)
        publication_gate = (publication.get("gate")
                            if isinstance(publication, dict) else None)
        if publication_repro == {}:
            publication_repro = None
        if publication_gate == {}:
            publication_gate = None
        reason = publication_block_reason(trace, publication_repro,
                                          publication_gate)
        if reason:
            return {
                "stored": [],
                "status": "blocked",
                "provenance": "unavailable",
                "reason": reason,
            }

        stored: List[str] = []
        try:
            if store is None:
                import os as _os
                from backend.knowledge_graph.graph_store import GraphStore
                override = _os.environ.get("MECH_KG_PATH", "")
                store = GraphStore(override) if override else GraphStore()
            from backend.knowledge_graph.graph_store import KGEdge, KGNode
            from backend.knowledge_graph.ontology import EdgeType, NodeType

            def add_node(node_id: str, node_type: Any, label: str,
                         properties: Optional[Dict[str, Any]] = None) -> str:
                store.add_node(KGNode(node_id=node_id,
                                      node_type=node_type.value
                                      if hasattr(node_type, "value")
                                      else str(node_type),
                                      label=label,
                                      properties=properties or {}))
                stored.append(node_id)
                return node_id

            def add_edge(source: str, target: str, edge_type: Any,
                         weight: float = 1.0) -> None:
                store.add_edge(KGEdge(
                    edge_id=f"{source}__{target}__{str(edge_type)}",
                    source_id=source, target_id=target,
                    edge_type=edge_type.value
                    if hasattr(edge_type, "value") else str(edge_type),
                    weight=weight))

            ok = sum(1 for t in trace
                     if t.get("status") in ("completed", "ok", "loaded"))
            camp = add_node(f"camp_{run_id}", NodeType.CAMPAIGN,
                            f"Campaign: {goal[:60]}",
                            {"goal": goal, "steps_completed":
                             f"{ok}/{len(trace)}",
                             "status": publication.get("status", "")})
            add_node(f"goal_{run_id}", NodeType.RESEARCH_GOAL, goal[:80], {})
            add_edge(camp, f"goal_{run_id}", EdgeType.CONTAINS)
            exp = add_node(publication.get("experiment_id", f"exp_{run_id}"),
                           NodeType.EXPERIMENT,
                           f"Society run {run_id}",
                           {"steps_completed": f"{ok}/{len(trace)}"})
            add_edge(camp, exp, EdgeType.CONTAINS)

            by_node = {t.get("node", ""): t for t in trace}
            disc = by_node.get("discover", {})
            result = disc.get("result", {}) if isinstance(disc, dict) else {}
            disc_id = result.get("discovery_id", "")
            if disc.get("status") == "completed" and disc_id:
                score = _first_score(result, ("circuit_score",
                                              "confidence_score",
                                              "calibrated_confidence"))
                circ = add_node(f"circuit_{disc_id}", NodeType.CIRCUIT,
                                f"Circuit {disc_id}",
                                {"score": score,
                                 "heads": result.get("top_attributed_nodes",
                                                     result.get("nodes", []))})
                add_edge(camp, circ, EdgeType.DISCOVERED_BY, weight=score)
                add_edge(exp, circ, EdgeType.SUPPORTS, weight=score)

            patch = by_node.get("patch", {})
            if patch.get("status") == "ok" and isinstance(
                    patch.get("delta"), (int, float)):
                ev = add_node(f"ev_{run_id}_patch", NodeType.EVIDENCE,
                              f"patch delta {patch['delta']}",
                              {"delta": patch["delta"],
                               "layer": patch.get("layer"),
                               "head": patch.get("head")})
                add_edge(exp, ev, EdgeType.SUPPORTS,
                         weight=min(1.0, abs(float(patch["delta"]))))

            valid = by_node.get("validate", {})
            vres = valid.get("result", {}) if isinstance(valid, dict) else {}
            conf = _first_score(vres, ("confidence_score",))
            if valid.get("status") == "completed":
                ev = add_node(f"ev_{run_id}_validation", NodeType.EVIDENCE,
                              f"validation confidence {conf}",
                              {"confidence": conf,
                               "validated": vres.get("validated", False)})
                add_edge(exp, ev, EdgeType.VALIDATED_BY,
                         weight=min(1.0, float(conf or 0.0)))
                if vres.get("validated"):
                    claim = add_node(f"claim_{run_id}", NodeType.MECHANISM_CLAIM,
                                     f"Validated: {goal[:60]}",
                                     {"confidence": conf})
                    add_edge(ev, claim, EdgeType.SUPPORTS,
                             weight=min(1.0, float(conf or 0.0)))

            pub = add_node(f"pub_{run_id}", NodeType.PUBLICATION,
                           f"Mechanistic Report {run_id}",
                           {"experiment_id": publication.get("experiment_id",
                                                             "")})
            add_edge(exp, pub, EdgeType.SUPPORTS)
            gate = publication.get("gate", {})
            if isinstance(gate, dict) and gate:
                ev = add_node(
                    f"ev_{run_id}_gate", NodeType.EVIDENCE,
                    f"gate {'PASSED' if gate.get('passed') else 'FAILED'} "
                    f"@ {gate.get('value', 0)}%",
                    {"passed": bool(gate.get("passed", False)),
                     "threshold_pct": gate.get("threshold", 0.85) * 100
                     if isinstance(gate.get("threshold"), (int, float))
                     else gate.get("threshold"),
                     "fidelity_pct": gate.get("value")})
                add_edge(exp, ev, EdgeType.VALIDATED_BY)
        except Exception as exc:
            return {"stored": stored, "error": str(exc)[:300]}
        return {"stored": stored}
