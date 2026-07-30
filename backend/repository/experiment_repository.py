"""Experiment Repository — Data access layer for research experiments."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExperimentRecord:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentRepository:
    """Stores research experiments linked to sessions."""

    def __init__(self):
        self._experiments: dict[str, ExperimentRecord] = {}

    def create(self, session_id: str, name: str, description: str = "", tags: list[str] = None) -> ExperimentRecord:
        exp = ExperimentRecord(session_id=session_id, name=name, description=description, tags=tags or [])
        self._experiments[exp.id] = exp
        return exp

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        return self._experiments.get(experiment_id)

    def list_for_session(self, session_id: str) -> list[ExperimentRecord]:
        return [e for e in self._experiments.values() if e.session_id == session_id]

    def search(self, query: str) -> list[ExperimentRecord]:
        q = query.lower()
        return [
            e for e in self._experiments.values()
            if q in e.name.lower() or q in e.description.lower() or any(q in t.lower() for t in e.tags)
        ]


experiment_repo = ExperimentRepository()
