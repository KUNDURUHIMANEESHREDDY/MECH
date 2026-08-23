r"""Theory Extinction & Audit Preservation Engine for MECH.

Manages auditable, non-destructive extinction transitions:
    ACTIVE -> WEAKENED -> SUPERSEDED -> FALSIFIED_REVERTED
Preserves complete historical lineage and reasons for elimination.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief


@dataclass
class TheoryExtinctionRecord:
    eliminated_theory_id: str
    eliminating_experiment_id: str
    winning_competing_theory_id: str
    posterior_before: float
    posterior_after: float
    extinction_reason: str
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eliminated_theory_id": self.eliminated_theory_id,
            "eliminating_experiment_id": self.eliminating_experiment_id,
            "winning_competing_theory_id": self.winning_competing_theory_id,
            "posterior_before": round(self.posterior_before, 4),
            "posterior_after": round(self.posterior_after, 4),
            "extinction_reason": self.extinction_reason,
            "timestamp_utc": self.timestamp_utc,
        }


class TheoryExtinctionEngine:
    """Safely transitions eliminated theories without deleting lineage."""

    def __init__(self, claim_graph: Optional[ClaimDependencyGraphEngine] = None) -> None:
        self.claim_graph = claim_graph or ClaimDependencyGraphEngine()
        self.extinction_audit_log: List[TheoryExtinctionRecord] = []

    def eliminate_theory(
        self,
        theory_id: str,
        experiment_id: str,
        winning_theory_id: str,
        posterior_before: float,
        posterior_after: float,
        reason: str,
    ) -> TheoryExtinctionRecord:
        """Transitions theory in the Claim DAG to SUPERSEDED / FALSIFIED_REVERTED."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        rec = TheoryExtinctionRecord(
            eliminated_theory_id=theory_id,
            eliminating_experiment_id=experiment_id,
            winning_competing_theory_id=winning_theory_id,
            posterior_before=posterior_before,
            posterior_after=posterior_after,
            extinction_reason=reason,
            timestamp_utc=ts,
        )
        self.extinction_audit_log.append(rec)

        claim_id = f"CLAIM_THEORY_{theory_id}"
        if claim_id in self.claim_graph.claims:
            self.claim_graph.claims[claim_id].belief_status = ClaimEpistemicBelief.SUPERSEDED

        return rec
