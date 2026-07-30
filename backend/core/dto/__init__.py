"""Core Shared DTO Package.

Standardized data transfer objects used across all AI subsystems.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class CoreExperimentDTO:
    def __init__(self, experiment_id: str, goal: str, model: str = "GPT-2 Small") -> None:
        self.experiment_id = experiment_id
        self.goal = goal
        self.model = model
        self.timestamp = _dt.datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "goal": self.goal,
            "model": self.model,
            "timestamp": self.timestamp,
        }


class CoreDiscoveryDTO:
    def __init__(self, discovery_id: str, title: str, confidence: float = 0.95) -> None:
        self.discovery_id = discovery_id
        self.title = title
        self.confidence = confidence
        self.timestamp = _dt.datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discovery_id": self.discovery_id,
            "title": self.title,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }
