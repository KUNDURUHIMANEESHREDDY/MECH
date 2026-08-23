"""Epistemic Ledger Dynamic Revision Orchestrator for MECH.

Integrates:
1. Autonomous Science Orchestrator (Experimentation & Certification).
2. Claim Dependency Graph Engine (Living DAG, Cascade Invalidation, Belief Lifecycle).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import ModelRuntimeInterface
from .autonomous_science_orchestrator import AutonomousScienceOrchestrator, AutonomousScienceRunResult
from .claim_dependency_graph import (
    ClaimDependencyGraphEngine,
    ClaimEpistemicBelief,
    DependencyType,
    DynamicClaimNode,
)


@dataclass
class EpistemicRevisionSummary:
    """Summary of dynamic belief revision across registered scientific claims."""
    total_registered_claims: int
    active_supported_claims_count: int
    evidence_weakened_claims_count: int
    superseded_claims_count: int
    falsified_reverted_claims_count: int
    total_revision_events: int
    active_claims_list: List[str]
    weakened_claims_list: List[str]
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EpistemicRevisionOrchestrator:
    """Orchestrates dynamic claim registration, dependency tracking, and cascade belief revision."""

    def __init__(
        self,
        runtime: Optional[ModelRuntimeInterface] = None,
        model_id: str = "gpt2",
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.runtime = runtime or InMemoryRuntime(model_id=model_id, device=device)
        self.science_orchestrator = AutonomousScienceOrchestrator(runtime=self.runtime, model_id=model_id, device=device)
        self.graph_engine = ClaimDependencyGraphEngine()

    def register_scientific_run_to_graph(
        self,
        run_result: AutonomousScienceRunResult,
        parent_composite_claim_id: Optional[str] = None,
    ) -> DynamicClaimNode:
        """Converts an AutonomousScienceRunResult into a living claim node in the dependency graph."""
        cert = run_result.mechanistic_claim_certificate
        cid = cert.circuit_or_component_id

        dep_list = [
            (f"EXP_NODE_ABLATION_{cid}", DependencyType.NODE_ABLATION),
            (f"EXP_4CTRL_BATTERY_{cid}", DependencyType.CONTROL_BATTERY),
            (f"EXP_MEDIATION_{cid}", DependencyType.MEDIATION_RESCUE),
            (f"EXP_REPLICATION_{cid}", DependencyType.REPLICATION_BATTERY),
            (f"EXP_DISCRIMINATING_{cid}", DependencyType.DISCRIMINATING_FALSIFICATION),
        ]

        claim_id = f"CLAIM_{cid}_{run_result.behavior_name.upper()}"

        node = self.graph_engine.register_claim(
            claim_id=claim_id,
            certificate_id=cert.certificate_id,
            circuit_or_component_id=cid,
            behavior_name=run_result.behavior_name,
            claim_statement=cert.claim_statement,
            dependency_experiment_ids=dep_list,
            dependent_parent_claim_ids=[parent_composite_claim_id] if parent_composite_claim_id else [],
        )

        return node

    def trigger_experiment_invalidation(
        self,
        experiment_id: str,
        refutation_rationale: str,
    ) -> List[str]:
        """Invalidates an experiment and cascades belief revisions down all dependent claims."""
        return self.graph_engine.invalidate_experiment(experiment_id, refutation_rationale)

    def get_epistemic_ledger_summary(self) -> EpistemicRevisionSummary:
        """Computes summary statistics across all living claims in the dependency graph."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        all_claims = list(self.graph_engine.claims.values())

        active = [c.claim_id for c in all_claims if c.belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED]
        weakened = [c.claim_id for c in all_claims if c.belief_status == ClaimEpistemicBelief.EVIDENCE_WEAKENED]
        superseded = [c.claim_id for c in all_claims if c.belief_status == ClaimEpistemicBelief.SUPERSEDED]
        falsified = [c.claim_id for c in all_claims if c.belief_status == ClaimEpistemicBelief.FALSIFIED_REVERTED]

        return EpistemicRevisionSummary(
            total_registered_claims=len(all_claims),
            active_supported_claims_count=len(active),
            evidence_weakened_claims_count=len(weakened),
            superseded_claims_count=len(superseded),
            falsified_reverted_claims_count=len(falsified),
            total_revision_events=len(self.graph_engine.revision_events),
            active_claims_list=active,
            weakened_claims_list=weakened,
            timestamp_utc=ts,
        )
