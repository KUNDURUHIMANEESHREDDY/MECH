r"""Adversarial Claim Policy & Hierarchical DAG Engine for MECH.

Manages adversarial epistemic statuses:
1. ADVERSARIAL_SUPPORTED: Causal transportability survived deliberate adversarial challenge.
2. ADVERSARIAL_BOUNDARY_CONFIRMED: Successfully identified architecture boundary and abstained.
3. ADVERSARIAL_FAIL: Divergence under adversarial conditions -> Generates revision agenda.
4. ADVERSARIAL_UNRESOLVED: Awaiting independent external adversarial trial data.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief, DependencyType


class AdversarialEpistemicStatus(str, Enum):
    ADVERSARIAL_SUPPORTED = "ADVERSARIAL_SUPPORTED"
    ADVERSARIAL_BOUNDARY_CONFIRMED = "ADVERSARIAL_BOUNDARY_CONFIRMED"
    ADVERSARIAL_FAIL = "ADVERSARIAL_FAIL"
    ADVERSARIAL_UNRESOLVED = "ADVERSARIAL_UNRESOLVED"


@dataclass
class AdversarialClaimRecord:
    claim_id: str
    parent_generalization_claim_id: str
    status: AdversarialEpistemicStatus
    fcr_adv: float
    bdr_score: float
    ar_score: float
    timestamp_utc: str
    audit_verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "parent_generalization_claim_id": self.parent_generalization_claim_id,
            "status": self.status.value,
            "fcr_adv": round(self.fcr_adv, 4),
            "bdr_score": round(self.bdr_score, 4),
            "ar_score": round(self.ar_score, 4),
            "timestamp_utc": self.timestamp_utc,
            "audit_verdict": self.audit_verdict,
        }


class AdversarialClaimPolicy:
    """Manages the lifecycle of adversarial replication claims in the living Claim DAG."""

    def __init__(self, claim_graph: Optional[ClaimDependencyGraphEngine] = None) -> None:
        self.claim_graph = claim_graph or ClaimDependencyGraphEngine()

    def register_adversarial_claim(
        self,
        claim_id: str,
        parent_generalization_claim_id: str,
        fcr_adv: float,
        bdr_score: float,
        ar_score: float,
        is_boundary_confirmed: bool = False,
    ) -> AdversarialClaimRecord:
        """Registers the adversarial claim and assigns hierarchical epistemic belief status."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()

        if fcr_adv <= 0.02 and bdr_score >= 0.90 and ar_score >= 0.85:
            if is_boundary_confirmed:
                status = AdversarialEpistemicStatus.ADVERSARIAL_BOUNDARY_CONFIRMED
                verdict = "Boundary Confirmed: Epistemic abstention validated under adversarial stress."
            else:
                status = AdversarialEpistemicStatus.ADVERSARIAL_SUPPORTED
                verdict = "Adversarial Supported: Transportability law survived deliberate adversary."
            belief_status = ClaimEpistemicBelief.ACTIVE_SUPPORTED
        else:
            status = AdversarialEpistemicStatus.ADVERSARIAL_FAIL
            verdict = "Adversarial Fail: Model showed uncalibrated false confidence under adversarial selection."
            belief_status = ClaimEpistemicBelief.FALSIFIED_REVERTED

        statement = (
            f"Adversarial Claim [{status.value}]: FCR={fcr_adv*100:.2f}%, BDR={bdr_score*100:.1f}%, "
            f"AR={ar_score*100:.1f}%. {verdict}"
        )

        self.claim_graph.register_claim(
            claim_id=claim_id,
            certificate_id=f"CERT_ADV_{ts[:10]}",
            circuit_or_component_id="ADVERSARIAL_ENGINE",
            behavior_name="independent_adversarial_replication",
            claim_statement=statement,
            dependency_experiment_ids=[(parent_generalization_claim_id, DependencyType.PRIMITIVE_CLAIM)],
        )
        self.claim_graph.claims[claim_id].belief_status = belief_status

        return AdversarialClaimRecord(
            claim_id=claim_id,
            parent_generalization_claim_id=parent_generalization_claim_id,
            status=status,
            fcr_adv=fcr_adv,
            bdr_score=bdr_score,
            ar_score=ar_score,
            timestamp_utc=ts,
            audit_verdict=verdict,
        )
