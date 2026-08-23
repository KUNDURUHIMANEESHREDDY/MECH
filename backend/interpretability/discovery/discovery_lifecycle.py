"""Discovery Lifecycle Management.

Defines and enforces the formal discovery lifecycle:
Candidate ➔ Evidence Collection ➔ Validation ➔ Confidence ➔ Knowledge ➔ Publication.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List

VALID_DISCOVERY_STATES = {
    "Candidate",
    "Evidence Collection",
    "Validation",
    "Confidence",
    "Knowledge",
    "Publication",
}


class DiscoveryLifecycleState:
    """Represents the lifecycle state of a single autonomous discovery."""

    def __init__(self, discovery_id: str, title: str = "") -> None:
        self.discovery_id = discovery_id
        self.title = title
        self.state = "Candidate"
        self.created_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.history: List[Dict[str, Any]] = [
            {"state": "Candidate", "timestamp": self.created_at, "note": "Discovery proposed"}
        ]

    def transition_to(self, new_state: str, note: str = "") -> Dict[str, Any]:
        if new_state not in VALID_DISCOVERY_STATES:
            raise ValueError(f"Invalid discovery state transition to {new_state}")

        self.state = new_state
        self.updated_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

        record = {"state": new_state, "timestamp": self.updated_at, "note": note}
        self.history.append(record)
        return record

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discovery_id": self.discovery_id,
            "title": self.title,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": self.history,
        }
