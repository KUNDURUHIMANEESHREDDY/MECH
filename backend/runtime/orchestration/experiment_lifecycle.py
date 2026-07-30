"""Experiment Lifecycle Management.

Defines and enforces the formal experiment lifecycle:
Created ➔ Queued ➔ Planning ➔ Scheduled ➔ Running ➔ Checkpointing ➔ Completed / Failed ➔ Archived.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List

VALID_STATES = {
    "Created",
    "Queued",
    "Planning",
    "Scheduled",
    "Running",
    "Checkpointing",
    "Completed",
    "Failed",
    "Archived",
}


class ExperimentLifecycleState:
    """Represents the lifecycle state of a single execution experiment."""

    def __init__(self, experiment_id: str, goal: str = "") -> None:
        self.experiment_id = experiment_id
        self.goal = goal
        self.state = "Created"
        self.created_at = _dt.datetime.utcnow().isoformat() + "Z"
        self.updated_at = self.created_at
        self.history: List[Dict[str, Any]] = [
            {"state": "Created", "timestamp": self.created_at, "note": "Experiment initialized"}
        ]

    def transition_to(self, new_state: str, note: str = "") -> Dict[str, Any]:
        if new_state not in VALID_STATES:
            raise ValueError(f"Invalid state transition to {new_state}")

        self.state = new_state
        self.updated_at = _dt.datetime.utcnow().isoformat() + "Z"
        record = {"state": new_state, "timestamp": self.updated_at, "note": note}
        self.history.append(record)
        return record

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "goal": self.goal,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": self.history,
        }
