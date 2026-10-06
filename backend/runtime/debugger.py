"""Forward Stepping Debugger Engine.

Manages interactive forward debugger sessions, layer stepping, breakpoints,
state inspection mid-inference, and event emissions.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Set


class DebuggerSession:
    """Represents an active forward stepping debug session."""

    def __init__(self, session_id: str, prompt: str, total_layers: int = 12) -> None:
        self.session_id = session_id
        self.prompt = prompt
        self.total_layers = total_layers
        self.current_layer = 0
        self.breakpoints: Set[int] = set()
        self.status = "initialized"  # initialized, paused, running, finished
        self.events: List[Dict[str, Any]] = []
        self._emit_event("DebuggerStarted", {"prompt": prompt, "total_layers": total_layers})

    def set_breakpoint(self, layer: int) -> None:
        self.breakpoints.add(layer)

    def remove_breakpoint(self, layer: int) -> None:
        self.breakpoints.discard(layer)

    def step(self) -> Dict[str, Any]:
        """Advance execution by 1 layer."""
        if self.current_layer >= self.total_layers:
            self.status = "finished"
            self._emit_event("DebuggerFinished", {"total_layers": self.total_layers})
            return self.get_state()

        self.current_layer += 1
        if self.current_layer in self.breakpoints:
            self.status = "paused"
            self._emit_event("BreakpointHit", {"layer": self.current_layer})
        else:
            self.status = "running"
            self._emit_event("StepCompleted", {"layer": self.current_layer})

        if self.current_layer >= self.total_layers:
            self.status = "finished"
            self._emit_event("DebuggerFinished", {"total_layers": self.total_layers})

        return self.get_state()

    def continue_execution(self) -> Dict[str, Any]:
        """Run until next breakpoint or completion."""
        self.status = "running"
        while self.current_layer < self.total_layers:
            self.current_layer += 1
            if self.current_layer in self.breakpoints:
                self.status = "paused"
                self._emit_event("BreakpointHit", {"layer": self.current_layer})
                break

        if self.current_layer >= self.total_layers:
            self.status = "finished"
            self._emit_event("DebuggerFinished", {"total_layers": self.total_layers})

        return self.get_state()

    def stop(self) -> Dict[str, Any]:
        self.status = "stopped"
        self._emit_event("DebuggerStopped", {"layer": self.current_layer})
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "current_layer": self.current_layer,
            "total_layers": self.total_layers,
            "status": self.status,
            "breakpoints": sorted(list(self.breakpoints)),
            "latest_event": self.events[-1] if self.events else None,
        }

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        evt = {
            "type": event_type,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "data": data,
        }
        self.events.append(evt)


_active_sessions: Dict[str, DebuggerSession] = {}


def start_debug_session(session_id: str, prompt: str, total_layers: int = 12) -> DebuggerSession:
    session = DebuggerSession(session_id=session_id, prompt=prompt, total_layers=total_layers)
    _active_sessions[session_id] = session
    return session


def get_debug_session(session_id: str) -> Optional[DebuggerSession]:
    return _active_sessions.get(session_id)
