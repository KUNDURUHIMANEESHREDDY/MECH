"""Execution Context for Runtime Engine.

Manages shared execution state across sessions, models, hooks, patches,
breakpoints, and profiling metrics.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional


class ExecutionContext:
    """Shared state container for runtime execution."""

    def __init__(self, session_id: str = "default_session", model_name: str = "GPT-2 Small") -> None:
        self.session_id = session_id
        self.model_name = model_name
        self.active_breakpoints: List[int] = []
        self.patches: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.created_at = _dt.datetime.utcnow().isoformat() + "Z"

    def add_breakpoint(self, layer: int) -> None:
        if layer not in self.active_breakpoints:
            self.active_breakpoints.append(layer)

    def remove_breakpoint(self, layer: int) -> None:
        if layer in self.active_breakpoints:
            self.active_breakpoints.remove(layer)

    def get_breakpoints(self) -> List[int]:
        return list(self.active_breakpoints)

    def add_patch(self, patch_info: Dict[str, Any]) -> None:
        self.patches.append(patch_info)
        self.record_intervention(patch_info)

    def record_intervention(self, patch_info: Dict[str, Any]) -> None:
        entry = {
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            **patch_info,
        }
        self.history.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "active_breakpoints": self.active_breakpoints,
            "active_patches_count": len(self.patches),
            "intervention_history_count": len(self.history),
            "created_at": self.created_at,
        }
