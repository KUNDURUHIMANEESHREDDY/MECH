"""Distributed Evidence Aggregator & Single-Threaded Belief Updater.

Collects worker execution results from concurrent channels, performs evidence fusion, 
and executes single-threaded Bayesian belief updates.

Guarantees thread-safe belief updates and mechanism claim registry synchronization.
"""

from __future__ import annotations

import datetime as _dt
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .worker import WorkerExecutionResult
from ..interpretability.reasoning.bayesian_belief_engine import BayesianBeliefEngine, MechanismBelief
from ..interpretability.discovery.mechanism_claim_registry import MechanismClaimRegistry, RegisteredMechanismClaim


@dataclass
class AggregatedEvidenceResult:
    """Consolidated evidence result after distributed fusion."""
    campaign_id: str
    total_worker_results: int
    successful_results: int
    failed_results: int
    fused_confidence: float
    updated_belief: Dict[str, Any]
    registered_claim_id: str
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")


class EvidenceAggregator:
    """Thread-safe evidence fusion aggregator & single-threaded Bayesian belief updater."""

    def __init__(
        self,
        belief_engine: Optional[BayesianBeliefEngine] = None,
        claim_registry: Optional[MechanismClaimRegistry] = None
    ) -> None:
        self.belief_engine = belief_engine or BayesianBeliefEngine()
        self.claim_registry = claim_registry or MechanismClaimRegistry()
        self._lock = threading.Lock()

    def aggregate_worker_results(
        self,
        campaign_id: str,
        results: List[WorkerExecutionResult],
        claim_title: str = "IOI Name Mover Circuit"
    ) -> AggregatedEvidenceResult:
        """Aggregates worker outputs and performs thread-safe single-threaded Bayesian belief update."""
        with self._lock:
            successful = [r for r in results if r.status == "Success"]
            failed = [r for r in results if r.status == "Error"]

            # Initialize baseline belief
            belief = MechanismBelief(
                claim_id=f"claim_{campaign_id}",
                title=claim_title,
                prior=0.50,
                posterior=0.50
            )

            # Apply sequential single-threaded Bayesian updates for each worker report
            for idx, res in enumerate(successful):
                rep = res.discovery_report or {}
                alg_name = rep.get("algorithm", "attribution_patching")
                conf = rep.get("confidence", 0.90)

                is_supp = conf >= 0.70
                ev_id = f"ev_{res.worker_id}_{idx + 1}"

                belief = self.belief_engine.update_belief(
                    current_belief=belief,
                    algorithm_name=alg_name,
                    evidence_id=ev_id,
                    is_supporting=is_supp,
                    evidence_strength=min(1.0, conf)
                )

            # Register or update claim in persistent MechanismClaimRegistry
            status_str = "Validated" if belief.posterior >= 0.85 else "Under_Revision"
            claim_obj = RegisteredMechanismClaim(
                claim_id=belief.claim_id,
                title=belief.title,
                description=f"Aggregated claim verified across {len(successful)} worker execution runs.",
                status=status_str,
                confidence=belief.posterior,
                replications=len(successful),
                models=["GPT2-S", "Gemma2"],
                supporting_experiments=len(successful),
                contradicting_experiments=len(failed),
                algorithms_used=list(set(r.discovery_report.get("algorithm", "acdc") for r in successful if r.discovery_report))
            )

            self.claim_registry.register(claim_obj)

            return AggregatedEvidenceResult(
                campaign_id=campaign_id,
                total_worker_results=len(results),
                successful_results=len(successful),
                failed_results=len(failed),
                fused_confidence=belief.posterior,
                updated_belief=belief.to_dict(),
                registered_claim_id=claim_obj.claim_id
            )
