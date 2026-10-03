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
    """Consolidated evidence result after distributed fusion.

    `fused_confidence` was the posterior unconditionally, including when no
    worker report carried a confidence -- in which case the posterior is just
    the 0.50 prior being echoed back. It is now Optional, with
    `evidence_measured` distinguishing the two cases.
    """
    campaign_id: str
    total_worker_results: int
    successful_results: int
    failed_results: int
    fused_confidence: Optional[float]
    updated_belief: Dict[str, Any]
    registered_claim_id: str
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")
    reports_with_measured_confidence: int = 0
    reports_without_measured_confidence: List[str] = field(default_factory=list)
    evidence_measured: bool = False
    provenance: str = "unavailable"
    reason: Optional[str] = None


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
        """Aggregates worker outputs and performs thread-safe single-threaded Bayesian belief update.

        Four fabrications were removed here.

        ``rep.get("confidence", 0.90)`` defaulted a *missing* confidence to 0.90,
        which is above the 0.70 supporting threshold -- so every worker report
        that omitted a confidence was silently counted as supporting evidence
        for the claim. A missing input now contributes nothing.

        ``rep.get("algorithm", "attribution_patching")`` / ``"acdc"`` invented
        an algorithm name for any report without one, and those invented names
        were then recorded in ``algorithms_used``.

        ``models=["GPT2-S", "Gemma2"]`` was a constant, so every claim claimed
        replication on two models regardless of which models ran.

        The description read "verified across N worker execution runs" purely
        from the count of successes. N successes is not verification; the word
        is gone.
        """
        with self._lock:
            successful = [r for r in results if r.status == "Success"]
            failed = [r for r in results if r.status == "Error"]

            # Initialize baseline belief. 0.50 is a genuine neutral prior, not
            # an invented observation.
            belief = MechanismBelief(
                claim_id=f"claim_{campaign_id}",
                title=claim_title,
                prior=0.50,
                posterior=0.50
            )

            # Only reports that carry their own measured confidence may move
            # the belief. Skipped reports are counted so the caller can see
            # how much of the input was unusable.
            usable = 0
            skipped: List[str] = []
            for idx, res in enumerate(successful):
                rep = res.discovery_report or {}
                conf = rep.get("confidence")
                alg_name = rep.get("algorithm")

                if conf is None or not isinstance(conf, (int, float)):
                    skipped.append(res.worker_id)
                    continue
                if alg_name is None:
                    skipped.append(res.worker_id)
                    continue

                is_supp = float(conf) >= 0.70
                usable += 1
                ev_id = f"ev_{res.worker_id}_{idx + 1}"

                belief = self.belief_engine.update_belief(
                    current_belief=belief,
                    algorithm_name=str(alg_name),
                    evidence_id=ev_id,
                    is_supporting=is_supp,
                    evidence_strength=min(1.0, float(conf)),
                )

            # With no usable evidence the posterior is still the 0.50 prior. It
            # must not be reported as a finding.
            evidence_measured = usable > 0
            if not evidence_measured:
                status_str = "Unsupported"
            elif belief.posterior >= 0.85:
                status_str = "Validated"
            else:
                status_str = "Under_Revision"

            # Models come from the reports, not from a constant.
            models_used = sorted({
                str((r.discovery_report or {}).get("model"))
                for r in successful
                if (r.discovery_report or {}).get("model")
            })
            algorithms_used = sorted({
                str((r.discovery_report or {}).get("algorithm"))
                for r in successful
                if (r.discovery_report or {}).get("algorithm")
            })

            claim_obj = RegisteredMechanismClaim(
                claim_id=belief.claim_id,
                title=belief.title,
                description=(
                    f"Aggregated from {usable} worker report(s) carrying a "
                    f"measured confidence."
                    if evidence_measured else
                    "No worker report carried a measured confidence, so this "
                    "claim carries no evidence and its posterior is the "
                    "uninformative prior."
                ),
                status=status_str,
                confidence=belief.posterior if evidence_measured else None,
                # "Replications" counts reports, which is not replication.
                replications=None,
                models=models_used,
                supporting_experiments=usable,
                contradicting_experiments=len(failed),
                algorithms_used=algorithms_used,
                provenance="live" if evidence_measured else "unavailable",
                validation_eligible=evidence_measured,
                publication_eligible=False,
                workers_without_measured_confidence=skipped,
                reason=(None if evidence_measured else
                        "Worker reports supplied no confidence or algorithm, so "
                        "no evidence could be aggregated."),
            )

            self.claim_registry.register(claim_obj)

            return AggregatedEvidenceResult(
                campaign_id=campaign_id,
                total_worker_results=len(results),
                successful_results=len(successful),
                failed_results=len(failed),
                # The 0.50 prior echoed back is not a fused confidence.
                fused_confidence=belief.posterior if evidence_measured else None,
                updated_belief=belief.to_dict(),
                registered_claim_id=claim_obj.claim_id,
                reports_with_measured_confidence=usable,
                reports_without_measured_confidence=skipped,
                evidence_measured=evidence_measured,
                provenance="live" if evidence_measured else "unavailable",
                reason=claim_obj.reason,
            )
