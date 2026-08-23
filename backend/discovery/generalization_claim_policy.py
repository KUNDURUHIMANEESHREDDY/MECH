r"""Generalization Claim Policy & Granular DAG Revision Engine for MECH.

Manages granular epistemic statuses for held-out generalization:
1. HELDOUT_SUPPORTED: Successfully replicated across held-out combinatorial split.
2. HELDOUT_PARTIAL: Partial transfer with bounded error.
3. OOD_SUPPORTED: Validated out-of-distribution across high novelty gap.
4. BOUNDARY_CONFIRMED: Epistemic abstention successfully verified at architecture boundary.
5. OOD_FALSIFIED: Transfer failed where positive transfer was predicted out-of-domain.
6. GENERALIZATION_UNRESOLVED: Awaiting independent held-out trial data.

Implements two-stage claim revision:
    Held-out divergence downgrades the GENERALIZATION claim specifically,
    without destroying valid in-domain calibration certificates.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief, DependencyType


class GeneralizationEpistemicStatus(str, Enum):
    HELDOUT_SUPPORTED = "HELDOUT_SUPPORTED"
    HELDOUT_PARTIAL = "HELDOUT_PARTIAL"
    OOD_SUPPORTED = "OOD_SUPPORTED"
    BOUNDARY_CONFIRMED = "BOUNDARY_CONFIRMED"
    OOD_FALSIFIED = "OOD_FALSIFIED"
    GENERALIZATION_UNRESOLVED = "GENERALIZATION_UNRESOLVED"


@dataclass
class GeneralizationClaimRecord:
    claim_id: str
    in_domain_claim_id: str
    generalization_status: GeneralizationEpistemicStatus
    hpa_score: float
    transfer_class_accuracy: float
    ece_heldout: float
    fcr_heldout: float
    abstention_precision: float
    ood_distance: float
    timestamp_utc: str
    audit_verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "in_domain_claim_id": self.in_domain_claim_id,
            "generalization_status": self.generalization_status.value,
            "hpa_score": round(self.hpa_score, 4),
            "transfer_class_accuracy": round(self.transfer_class_accuracy, 4),
            "ece_heldout": round(self.ece_heldout, 4),
            "fcr_heldout": round(self.fcr_heldout, 4),
            "abstention_precision": round(self.abstention_precision, 4),
            "ood_distance": round(self.ood_distance, 3),
            "timestamp_utc": self.timestamp_utc,
            "audit_verdict": self.audit_verdict,
        }


class GeneralizationClaimPolicy:
    """Manages the lifecycle of combinatorial generalization claims in the living Claim DAG."""

    def __init__(self, claim_graph: Optional[ClaimDependencyGraphEngine] = None) -> None:
        self.claim_graph = claim_graph or ClaimDependencyGraphEngine()

    def register_heldout_generalization_claim(
        self,
        claim_id: str,
        in_domain_claim_id: str,
        hpa_score: float,
        transfer_class_accuracy: float,
        ece_heldout: float,
        fcr_heldout: float,
        abstention_precision: float,
        ood_distance: float,
    ) -> GeneralizationClaimRecord:
        """Determines granular generalization status and registers to the Claim DAG."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()

        # Epistemic Decision Boundary
        if hpa_score >= 0.90 and transfer_class_accuracy >= 0.90 and ece_heldout <= 0.05 and fcr_heldout <= 0.02:
            if ood_distance >= 0.70 and abstention_precision >= 0.90:
                status = GeneralizationEpistemicStatus.BOUNDARY_CONFIRMED
                verdict = "Boundary Confirmed: Epistemic abstention validated out-of-domain."
            elif ood_distance >= 0.50:
                status = GeneralizationEpistemicStatus.OOD_SUPPORTED
                verdict = "OOD Supported: Generalization verified on high novelty distance."
            else:
                status = GeneralizationEpistemicStatus.HELDOUT_SUPPORTED
                verdict = "Held-Out Supported: Combinatorial prospective generalization certified."
            belief_status = ClaimEpistemicBelief.ACTIVE_SUPPORTED
        elif hpa_score >= 0.75:
            status = GeneralizationEpistemicStatus.HELDOUT_PARTIAL
            verdict = "Held-Out Partial: Partial transfer with bounded error."
            belief_status = ClaimEpistemicBelief.EVIDENCE_WEAKENED
        else:
            status = GeneralizationEpistemicStatus.OOD_FALSIFIED
            verdict = "OOD Falsified: Prospective predictions diverged on held-out split."
            belief_status = ClaimEpistemicBelief.FALSIFIED_REVERTED

        statement = (
            f"Generalization Claim [{status.value}]: HPA={hpa_score*100:.1f}%, "
            f"ClassAcc={transfer_class_accuracy*100:.1f}%, ECE={ece_heldout:.3f}, FCR={fcr_heldout:.3f}. {verdict}"
        )

        self.claim_graph.register_claim(
            claim_id=claim_id,
            certificate_id=f"CERT_GEN_{ts[:10]}",
            circuit_or_component_id="GENERALIZATION_ENGINE",
            behavior_name="combinatorial_prospective_generalization",
            claim_statement=statement,
            dependency_experiment_ids=[(in_domain_claim_id, DependencyType.PRIMITIVE_CLAIM)],
        )
        self.claim_graph.claims[claim_id].belief_status = belief_status

        return GeneralizationClaimRecord(
            claim_id=claim_id,
            in_domain_claim_id=in_domain_claim_id,
            generalization_status=status,
            hpa_score=hpa_score,
            transfer_class_accuracy=transfer_class_accuracy,
            ece_heldout=ece_heldout,
            fcr_heldout=fcr_heldout,
            abstention_precision=abstention_precision,
            ood_distance=ood_distance,
            timestamp_utc=ts,
            audit_verdict=verdict,
        )
