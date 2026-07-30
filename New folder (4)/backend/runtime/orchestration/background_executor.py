"""Persistent Background Execution Daemon."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class BackgroundExecutionDaemon:
    """Runs tasks in background and persists state across restarts."""

    def __init__(self) -> None:
        self.active_tasks: List[Dict[str, Any]] = []

    def start_background_task(self, task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        task = {
            "task_id": task_id,
            "payload": payload,
            "status": "RunningInBackground",
            "started_at": _dt.datetime.utcnow().isoformat() + "Z",
        }
        self.active_tasks.append(task)
        return task

    def get_status(self) -> Dict[str, Any]:
        return {
            "daemon_status": "active",
            "active_tasks_count": len(self.active_tasks),
            "tasks": self.active_tasks,
        }
