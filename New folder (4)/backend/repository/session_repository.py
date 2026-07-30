"""Session Repository.

Manages persistence and lookups for research debugging sessions.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional


class SessionRepository:
    """Repository for research session metadata and history."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {
            "sess_1": {
                "id": "sess_1",
                "name": "IOI Circuit Discovery",
                "model": "GPT-2 Small",
                "prompt": "When Mary and John went to the store, John gave a book to Mary",
                "created_at": _dt.datetime.utcnow().isoformat() + "Z",
            }
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        return list(self._sessions.values())

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def save_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        sid = session.get("id") or f"sess_{len(self._sessions) + 1}"
        session["id"] = sid
        session["updated_at"] = _dt.datetime.utcnow().isoformat() + "Z"
        self._sessions[sid] = session
        return session


_session_repo_instance: SessionRepository | None = None


def get_session_repository() -> SessionRepository:
    global _session_repo_instance
    if _session_repo_instance is None:
        _session_repo_instance = SessionRepository()
    return _session_repo_instance
