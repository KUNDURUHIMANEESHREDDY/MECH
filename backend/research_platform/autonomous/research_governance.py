"""Research Governance Engine — Manages scientific claim lifecycles.

Controls transitions between states:
EXPERIMENTAL → UNDER_REVIEW → REVISION_REQUIRED → VALIDATED → SUPERSEDED.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Any, Dict, List, Optional

from .peer_review_panel import PeerReviewPanel, ReviewVerdict


class ClaimState(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    UNDER_REVIEW = "UNDER_REVIEW"
    REVISION_REQUIRED = "REVISION_REQUIRED"
    VALIDATED = "VALIDATED"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"


class ResearchGovernanceEngine:
    """Manages experiment approval workflows and mechanism claim lifecycle."""

    def __init__(self) -> None:
        self.audit_log: List[Dict[str, Any]] = []
        self.claim_states: Dict[str, ClaimState] = {}
        self.review_panel = PeerReviewPanel()

    def register_claim(self, claim_id: str):
        self.claim_states[claim_id] = ClaimState.EXPERIMENTAL

    def submit_for_review(self, claim_id: str, evidence: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Triggers the multi-agent peer review process."""
        self.claim_states[claim_id] = ClaimState.UNDER_REVIEW

        review_result = self.review_panel.review_claim(claim_id, evidence, metadata)

        # Update state based on consensus
        if review_result.final_verdict == ReviewVerdict.PASS:
            self.claim_states[claim_id] = ClaimState.VALIDATED
        elif review_result.final_verdict == ReviewVerdict.FAIL:
            self.claim_states[claim_id] = ClaimState.REJECTED
        else:
            self.claim_states[claim_id] = ClaimState.REVISION_REQUIRED

        audit_entry = {
            "claim_id": claim_id,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "verdict": review_result.final_verdict.value,
            "new_state": self.claim_states[claim_id].value,
            "overall_score": review_result.overall_score_pct,
            "unanimous": review_result.unanimous_pass
        }
        self.audit_log.append(audit_entry)

        return {
            "status": "Review Complete",
            "claim_id": claim_id,
            "current_state": self.claim_states[claim_id].value,
            "review": review_result
        }

    def get_claim_status(self, claim_id: str) -> ClaimState:
        return self.claim_states.get(claim_id, ClaimState.EXPERIMENTAL)

    def validate_and_approve(self, experiment_id: str, requester: str = "ResearchDirector") -> Dict[str, Any]:
        """Legacy approval method."""
        entry = {
            "experiment_id": experiment_id,
            "requester": requester,
            "approved": True,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "provenance_hash": f"sha256_{hash(experiment_id) & 0xffffffff:08x}",
        }
        self.audit_log.append(entry)
        return entry

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        return list(self.audit_log)
