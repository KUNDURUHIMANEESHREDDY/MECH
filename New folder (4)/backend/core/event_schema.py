"""Formal Event Schema.

Defines standard event types: ResearchStarted, ResearchFinished, DiscoveryCreated, HypothesisRejected, ExperimentQueued, CircuitValidated, PublicationGenerated.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict

VALID_EVENT_TYPES = {
    "ResearchStarted",
    "ResearchFinished",
    "DiscoveryCreated",
    "HypothesisRejected",
    "ExperimentQueued",
    "CircuitValidated",
    "PublicationGenerated",
}


class ResearchEvent:
    def __init__(self, event_type: str, payload: Dict[str, Any]) -> None:
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event type: {event_type}")
        self.event_type = event_type
        self.payload = payload
        self.timestamp = _dt.datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }
