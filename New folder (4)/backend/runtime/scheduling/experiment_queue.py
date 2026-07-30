"""Priority Experiment Queue Manager."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class PriorityExperimentQueue:
    """Manages priority-based execution queue with dependency tracking."""

    def __init__(self) -> None:
        self.queue: List[Dict[str, Any]] = []
        self.completed: List[Dict[str, Any]] = []

    def submit_experiment(
        self,
        experiment_id: str,
        priority: int = 1,
        dependencies: List[str] | None = None,
    ) -> Dict[str, Any]:
        item = {
            "experiment_id": experiment_id,
            "priority": priority,
            "dependencies": dependencies or [],
            "status": "Queued",
            "submitted_at": _dt.datetime.utcnow().isoformat() + "Z",
        }
        self.queue.append(item)
        self.queue.sort(key=lambda x: x["priority"], reverse=True)
        return item

    def pop_next(self) -> Dict[str, Any] | None:
        if not self.queue:
            return None
        item = self.queue.pop(0)
        item["status"] = "Running"
        self.completed.append(item)
        return item

    def list_queue(self) -> List[Dict[str, Any]]:
        return list(self.queue)
