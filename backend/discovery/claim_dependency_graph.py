"""Dynamic Epistemic Claim Dependency Graph & Belief Revision Engine for MECH.

Maintains the living DAG of scientific claims and their experimental dependencies:
1. Distinguishes Immutable History (SHA-256 raw logs) from Mutable Scientific Belief (Dynamic Claim DAG).
2. Tracks Claim Lifecycle: ACTIVE_SUPPORTED -> EVIDENCE_WEAKENED -> SUPERSEDED -> FALSIFIED_REVERTED.
3. Automated Cascade Invalidation: An invalidation of an underlying experiment (e.g., failed replication) automatically propagates down dependent claims.
4. Generates Targeted Follow-up Experiment Agendas to resolve weakened beliefs.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class ClaimEpistemicBelief(str, Enum):
    ACTIVE_SUPPORTED = "ACTIVE_SUPPORTED"           # Fully supported by all active dependencies
    EVIDENCE_WEAKENED = "EVIDENCE_WEAKENED"         # An underlying dependency failed or weakened
    SUPERSEDED = "SUPERSEDED"                       # Replaced by a more refined circuit claim
    FALSIFIED_REVERTED = "FALSIFIED_REVERTED"       # Refuted by new counterexample tests


class DependencyType(str, Enum):
    NODE_ABLATION = "NODE_ABLATION"
    EDGE_PATH_PATCHING = "EDGE_PATH_PATCHING"
    CONTROL_BATTERY = "CONTROL_BATTERY"
    MEDIATION_RESCUE = "MEDIATION_RESCUE"
    REPLICATION_BATTERY = "REPLICATION_BATTERY"
    DISCRIMINATING_FALSIFICATION = "DISCRIMINATING_FALSIFICATION"
    SUBCIRCUIT_CLAIM = "SUBCIRCUIT_CLAIM"
    PRIMITIVE_CLAIM = "PRIMITIVE_CLAIM"
    PRIMITIVE_COMPOSITION = "PRIMITIVE_COMPOSITION"



@dataclass
class DependencyEdge:
    """A directed dependency from a claim to an underlying experiment or subcircuit."""
    source_claim_id: str
    target_dependency_id: str
    dependency_type: DependencyType
    is_valid: bool = True
    invalidation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dependency_type"] = self.dependency_type.value
        return d


@dataclass
class RevisionHistoryRecord:
    """Immutable audit record of a scientific belief revision event."""
    revision_id: str
    previous_belief: ClaimEpistemicBelief
    new_belief: ClaimEpistemicBelief
    trigger_event: str
    rationale: str
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["previous_belief"] = self.previous_belief.value
        d["new_belief"] = self.new_belief.value
        return d


@dataclass
class DynamicClaimNode:
    """A living scientific claim in the Epistemic Dependency Graph."""
    claim_id: str
    certificate_id: str
    circuit_or_component_id: str
    behavior_name: str
    claim_statement: str
    belief_status: ClaimEpistemicBelief
    dependencies: List[DependencyEdge] = field(default_factory=list)
    dependent_claim_ids: List[str] = field(default_factory=list)
    revision_history: List[RevisionHistoryRecord] = field(default_factory=list)
    requested_experiments: List[str] = field(default_factory=list)
    last_updated_utc: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "certificate_id": self.certificate_id,
            "circuit_or_component_id": self.circuit_or_component_id,
            "behavior_name": self.behavior_name,
            "claim_statement": self.claim_statement,
            "belief_status": self.belief_status.value,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "dependent_claim_ids": self.dependent_claim_ids,
            "revision_history": [r.to_dict() for r in self.revision_history],
            "requested_experiments": self.requested_experiments,
            "last_updated_utc": self.last_updated_utc,
        }


class ClaimDependencyGraphEngine:
    """Coordinates the living DAG of scientific claims, dependencies, and cascade belief updates."""

    def __init__(self) -> None:
        self.claims: Dict[str, DynamicClaimNode] = {}
        self.experiment_to_claims_index: Dict[str, Set[str]] = {}
        self.revision_events: List[Dict[str, Any]] = []

    def register_claim(
        self,
        claim_id: str,
        certificate_id: str,
        circuit_or_component_id: str,
        behavior_name: str,
        claim_statement: str,
        dependency_experiment_ids: List[Tuple[str, DependencyType]],
        dependent_parent_claim_ids: Optional[List[str]] = None,
    ) -> DynamicClaimNode:
        """Registers a new claim node and creates dependency edges."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        deps: List[DependencyEdge] = []

        for dep_id, dep_type in dependency_experiment_ids:
            edge = DependencyEdge(
                source_claim_id=claim_id,
                target_dependency_id=dep_id,
                dependency_type=dep_type,
                is_valid=True,
            )
            deps.append(edge)
            if dep_id not in self.experiment_to_claims_index:
                self.experiment_to_claims_index[dep_id] = set()
            self.experiment_to_claims_index[dep_id].add(claim_id)

        node = DynamicClaimNode(
            claim_id=claim_id,
            certificate_id=certificate_id,
            circuit_or_component_id=circuit_or_component_id,
            behavior_name=behavior_name,
            claim_statement=claim_statement,
            belief_status=ClaimEpistemicBelief.ACTIVE_SUPPORTED,
            dependencies=deps,
            dependent_claim_ids=dependent_parent_claim_ids or [],
            revision_history=[
                RevisionHistoryRecord(
                    revision_id=f"rev_init_{claim_id[:8]}",
                    previous_belief=ClaimEpistemicBelief.ACTIVE_SUPPORTED,
                    new_belief=ClaimEpistemicBelief.ACTIVE_SUPPORTED,
                    trigger_event="INITIAL_CERTIFICATION",
                    rationale="Initial certification published to ledger with verified 7-criterion evidence.",
                    timestamp_utc=ts,
                )
            ],
            requested_experiments=[],
            last_updated_utc=ts,
        )

        self.claims[claim_id] = node
        return node

    def link_claim_dependency(self, parent_claim_id: str, child_claim_id: str) -> None:
        """Links a higher-level composite circuit claim to a child subcircuit/component claim."""
        if child_claim_id in self.claims and parent_claim_id in self.claims:
            self.claims[child_claim_id].dependent_claim_ids.append(parent_claim_id)
            self.claims[parent_claim_id].dependencies.append(
                DependencyEdge(
                    source_claim_id=parent_claim_id,
                    target_dependency_id=child_claim_id,
                    dependency_type=DependencyType.SUBCIRCUIT_CLAIM,
                    is_valid=True,
                )
            )

    def invalidate_experiment(
        self,
        experiment_id: str,
        refutation_rationale: str,
    ) -> List[str]:
        """Invalidates an underlying experiment and recursively cascades belief revisions down dependent claims."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        affected_claims = list(self.experiment_to_claims_index.get(experiment_id, set()))
        cascade_stack = list(affected_claims)
        revised_claim_ids: Set[str] = set()

        while cascade_stack:
            cid = cascade_stack.pop(0)
            if cid not in self.claims or cid in revised_claim_ids:
                continue

            node = self.claims[cid]
            # Mark the specific dependency edge as invalid
            for edge in node.dependencies:
                if edge.target_dependency_id == experiment_id:
                    edge.is_valid = False
                    edge.invalidation_reason = refutation_rationale

            # Determine new belief status
            prev_status = node.belief_status
            new_status = ClaimEpistemicBelief.EVIDENCE_WEAKENED

            # Formulate targeted follow-up experiments to resolve ambiguity
            req_exps = [
                f"Re-run {experiment_id} with wider sample battery.",
                f"Execute counterfactual boundary test for {node.circuit_or_component_id}.",
                f"Verify control battery specificity under updated prompt distribution.",
            ]

            node.belief_status = new_status
            node.requested_experiments = req_exps
            node.last_updated_utc = ts
            node.revision_history.append(
                RevisionHistoryRecord(
                    revision_id=f"rev_inval_{experiment_id[:6]}_{len(node.revision_history)}",
                    previous_belief=prev_status,
                    new_belief=new_status,
                    trigger_event=f"EXPERIMENT_INVALIDATION_{experiment_id}",
                    rationale=refutation_rationale,
                    timestamp_utc=ts,
                )
            )

            revised_claim_ids.add(cid)

            # Cascade to all higher-level dependent composite claims
            for parent_cid in node.dependent_claim_ids:
                if parent_cid not in revised_claim_ids:
                    cascade_stack.append(parent_cid)

        event_dict = {
            "event_id": f"event_cascade_{hashlib.sha256(f'{experiment_id}_{ts}'.encode()).hexdigest()[:10]}",
            "trigger_experiment_id": experiment_id,
            "refutation_rationale": refutation_rationale,
            "affected_claims_count": len(revised_claim_ids),
            "affected_claim_ids": list(revised_claim_ids),
            "timestamp_utc": ts,
        }
        self.revision_events.append(event_dict)

        return list(revised_claim_ids)

    def supersede_claim(
        self,
        old_claim_id: str,
        new_claim_id: str,
        superseding_rationale: str,
    ) -> None:
        """Marks an older claim as SUPERSEDED by a newer, more comprehensive claim."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        if old_claim_id in self.claims:
            node = self.claims[old_claim_id]
            prev = node.belief_status
            node.belief_status = ClaimEpistemicBelief.SUPERSEDED
            node.last_updated_utc = ts
            node.revision_history.append(
                RevisionHistoryRecord(
                    revision_id=f"rev_sup_{new_claim_id[:6]}_{len(node.revision_history)}",
                    previous_belief=prev,
                    new_belief=ClaimEpistemicBelief.SUPERSEDED,
                    trigger_event=f"SUPERSEDED_BY_{new_claim_id}",
                    rationale=superseding_rationale,
                    timestamp_utc=ts,
                )
            )

    def render_claim_dependency_tree(self, claim_id: str) -> str:
        """Renders formatted dependency and belief revision tree for a claim."""
        if claim_id not in self.claims:
            return f"Claim '{claim_id}' not found in dependency graph."

        node = self.claims[claim_id]
        lines = [
            f"┌─────────────────────────────────────────────────────────────┐",
            f"│ CLAIM DEPENDENCY & BELIEF STATUS: {node.claim_id}",
            f"│ Status: [{node.belief_status.value}] | Component: {node.circuit_or_component_id}",
            f"├─────────────────────────────────────────────────────────────┤",
            f"│ CLAIM STATEMENT: \"{node.claim_statement}\"",
            f"├─────────────────────────────────────────────────────────────┤",
            f"│ UNDERLYING EXPERIMENTAL DEPENDENCIES:",
        ]

        for d in node.dependencies:
            mark = "✓ VALID" if d.is_valid else f"✗ INVALID ({d.invalidation_reason})"
            lines.append(f"  ├── [{d.dependency_type.value}] {d.target_dependency_id}: {mark}")

        if node.dependent_claim_ids:
            lines.append(f"├─────────────────────────────────────────────────────────────┤")
            lines.append(f"│ PARENT COMPOSITE CLAIMS DEPENDENT ON THIS CLAIM:")
            for p in node.dependent_claim_ids:
                lines.append(f"  └── {p}")

        if node.requested_experiments:
            lines.append(f"├─────────────────────────────────────────────────────────────┤")
            lines.append(f"│ REQUESTED EXPERIMENTAL AGENDA (To resolve weakened belief):")
            for req in node.requested_experiments:
                lines.append(f"  • {req}")

        lines.append(f"├─────────────────────────────────────────────────────────────┤")
        lines.append(f"│ REVISION HISTORY COUNT: {len(node.revision_history)} audit events")
        lines.append(f"└─────────────────────────────────────────────────────────────┘")

        return "\n".join(lines)
