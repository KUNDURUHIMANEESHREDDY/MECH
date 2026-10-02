"""Traceable Evidence Graph.

Connects Neuron ➔ Feature ➔ Circuit ➔ Hypothesis ➔ Experiment ➔ Evidence ➔ Publication.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any, Dict, List, Optional

# One vocabulary for provenance, defined once at the boundary.
from .evidence_boundary import PROVENANCE_VALUES

EVIDENCE_DIR = os.environ.get("MECH_EVIDENCE_DIR", "backend/storage/evidence")

# Numeric step-result keys promoted to Evidence nodes by from_run().
EVIDENCE_KEYS = ("delta", "clean_ld", "patched_ld", "circuit_score",
                 "confidence_score", "attribution_score", "causal_effect",
                 "patch_success_rate", "functional_recovery")


def _iter_evidence(step: Dict[str, Any]):
    """Yield (dotted_key, number) pairs from a trace step, scanning the
    step top level plus one nested dict level (engine results nest)."""
    seen = set()

    def emit(prefix: str, obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            if key in EVIDENCE_KEYS and isinstance(value, (int, float)) \
                    and not isinstance(value, bool):
                dotted = f"{prefix}{key}" if prefix else str(key)
                if dotted not in seen:
                    seen.add(dotted)
                    yield dotted, value

    yield from emit("", step)
    result = step.get("result")
    yield from emit("", result)
    if isinstance(result, dict):
        for key, value in result.items():
            yield from emit(f"{key}.", value)

STEP_TYPES = {"planner": "Plan", "executor": "Execution",
              "inspector": "Observation", "discoverer": "Discovery",
              "critic": "Validation", "scribe": "Publication"}


def _scientific_policy():
    try:
        try:
            from backend.agents.evidence_policy import (
                discovery_is_live,
                provenance_of,
                validation_is_live,
            )
        except ImportError:
            from agents.evidence_policy import (
                discovery_is_live,
                provenance_of,
                validation_is_live,
            )
        return discovery_is_live, provenance_of, validation_is_live
    except Exception:
        return None


def _step_allows_evidence(step: Dict[str, Any]) -> bool:
    """Prevent synthetic discovery/validation numbers becoming evidence."""
    node = str(step.get("node", ""))
    if node not in {"discover", "validate"}:
        return True
    policy = _scientific_policy()
    if policy is None:
        return False
    discovery_is_live, _, validation_is_live = policy
    result = step.get("result")
    return (discovery_is_live(result) if node == "discover"
            else validation_is_live(result))


def _step_provenance(step: Dict[str, Any]) -> str:
    """Provenance declared by the step that produced an evidence value.

    Reads the result's own provenance, then the step's. Anything absent or
    unrecognised becomes "unavailable" -- never "live". Defaulting to live here
    is how a seeded measurement became a live-labelled evidence node.
    """
    for holder in (step.get("result"), step):
        if not isinstance(holder, dict):
            continue
        raw = holder.get("provenance")
        if raw is None:
            nested = holder.get("discovery_provenance")
            raw = nested
        if raw is None:
            continue
        label = str(raw).strip().lower()
        if label in PROVENANCE_VALUES:
            return label
        return "unavailable"
    return "unavailable"



class TraceableEvidenceGraph:
    """DAG graph tracking full evidence provenance from neuron activations to final paper."""

    def __init__(self) -> None:
        self.nodes: List[Dict[str, Any]] = [
            {"id": "n_402", "type": "Neuron", "label": "L8_N402"},
            {"id": "f_1402", "type": "Feature", "label": "SAE #1402"},
            {"id": "c_ioi", "type": "Circuit", "label": "IOI Circuit"},
            {"id": "h_ioi", "type": "Hypothesis", "label": "L8_N402 mediates IOI"},
            {"id": "exp_ioi", "type": "Experiment", "label": "Activation Patching L8_N402"},
            {"id": "ev_ioi", "type": "Evidence", "label": "Logit delta -4.2"},
            {"id": "pub_ioi", "type": "Publication", "label": "Mechanistic Paper #1"},
        ]
        self.edges: List[Dict[str, Any]] = [
            {"source": "n_402", "target": "f_1402", "relation": "encodes"},
            {"source": "f_1402", "target": "c_ioi", "relation": "forms"},
            {"source": "c_ioi", "target": "h_ioi", "relation": "suggests"},
            {"source": "h_ioi", "target": "exp_ioi", "relation": "tested_by"},
            {"source": "exp_ioi", "target": "ev_ioi", "relation": "yields"},
            {"source": "ev_ioi", "target": "pub_ioi", "relation": "substantiates"},
        ]

    def get_provenance_trace(self, target_id: str = "pub_ioi") -> List[Dict[str, Any]]:
        return list(self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes_count": len(self.nodes),
            "edges_count": len(self.edges),
            "nodes": self.nodes,
            "edges": self.edges,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        }

    # -- per-run construction (no demo nodes) --------------------------
    @classmethod
    def empty(cls) -> "TraceableEvidenceGraph":
        graph = cls.__new__(cls)
        graph.nodes = []
        graph.edges = []
        return graph

    def add_node(self, id: str, type: str, label: str,
                 payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        for existing in self.nodes:
            if existing.get("id") == id:
                return existing
        node: Dict[str, Any] = {"id": id, "type": type, "label": label}
        if payload:
            node["payload"] = payload
        self.nodes.append(node)
        return node

    def add_edge(self, source: str, target: str,
                 relation: str) -> Dict[str, Any]:
        edge = {"source": source, "target": target, "relation": relation}
        if edge not in self.edges:
            self.edges.append(edge)
        return edge

    @classmethod
    def from_run(cls, run_id: str, goal: str,
                 trace: List[Dict[str, Any]]) -> "TraceableEvidenceGraph":
        """Build a graph from a real Society run trace (demo-free)."""
        graph = cls.empty()
        goal_id = f"goal_{run_id}"
        graph.add_node(goal_id, "Goal", goal[:80])
        prev = goal_id
        for step in trace:
            node_id = f"{run_id}_{step.get('node', 'step')}"
            agent = str(step.get("agent", ""))
            status = str(step.get("status", ""))
            node_payload = {
                "agent": agent,
                "status": status,
                "op": step.get("op", ""),
            }
            if str(step.get("node", "")) in {"discover", "validate"}:
                policy = _scientific_policy()
                if policy is not None:
                    _, provenance_of, _ = policy
                    node_payload["provenance"] = provenance_of(step.get("result"))
                    node_payload["scientific_eligible"] = _step_allows_evidence(step)
            graph.add_node(
                node_id, STEP_TYPES.get(agent, "Execution"),
                f"{step.get('node', 'step')} ({status})", node_payload)
            graph.add_edge(prev, node_id, "followed_by")
            if _step_allows_evidence(step):
                for key, value in _iter_evidence(step):
                    ev_id = f"{node_id}_ev_{key.replace('.', '_')}"
                    # Provenance is derived, never defaulted to live. This used
                    # to initialise `evidence_provenance = "live"` and only
                    # overwrite it when a result happened to carry a
                    # provenance key -- so a step whose result omitted the key
                    # silently produced a live-labelled evidence node. Steps
                    # other than discover/validate skip the scientific
                    # eligibility gate entirely, which is how a seeded
                    # measurement could reach the graph labelled live.
                    evidence_provenance = _step_provenance(step)
                    node_payload = {
                        "value": value,
                        "provenance": evidence_provenance,
                        "field_provenance": {"value": evidence_provenance},
                    }
                    # An unattested "live" is not evidence. Only a step that
                    # both declares live and passes its eligibility gate may
                    # produce a live-labelled node.
                    if evidence_provenance == "live":
                        node_payload["attested"] = False
                        node_payload["reason"] = (
                            "Declared live by the producing step but not "
                            "attested with a RunAttestation; treated as "
                            "unverified."
                        )
                    graph.add_node(
                        ev_id,
                        "Evidence",
                        f"{key} = {value}",
                        node_payload,
                    )
                    graph.add_edge(node_id, ev_id, "yields")
            prev = node_id
        return graph

    # -- JSON file persistence ------------------------------------------
    def save(self, path: str) -> str:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "TraceableEvidenceGraph":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = cls.empty()
        graph.nodes = data.get("nodes", [])
        graph.edges = data.get("edges", [])
        return graph


def _evidence_dir(directory: Optional[str] = None) -> str:
    return directory or EVIDENCE_DIR


def save_run_record(run_id: str, record: Dict[str, Any],
                    directory: Optional[str] = None) -> str:
    """Persist a full Society run record (trace + graph + publication)."""
    target = _evidence_dir(directory)
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, f"{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def load_run_record(run_id: str,
                    directory: Optional[str] = None) -> Dict[str, Any]:
    path = os.path.join(_evidence_dir(directory), f"{run_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_run_records(directory: Optional[str] = None) -> List[Dict[str, Any]]:
    """Summaries of persisted runs, newest first. Never raises."""
    target = _evidence_dir(directory)
    try:
        files = sorted(
            (f for f in os.listdir(target) if f.endswith(".json")),
            reverse=True)
    except Exception:
        return []
    summaries = []
    for name in files[:200]:
        try:
            with open(os.path.join(target, name), "r",
                      encoding="utf-8") as f:
                record = json.load(f)
            pub = record.get("publication", {})
            summaries.append({
                "run_id": record.get("run_id", name[:-5]),
                "goal": record.get("goal", ""),
                "status": record.get("status", ""),
                "steps_completed": pub.get("steps_completed", ""),
                "created": pub.get("published_at", ""),
            })
        except Exception:
            continue
    return summaries
