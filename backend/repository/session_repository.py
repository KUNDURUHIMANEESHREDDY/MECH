"""Session & Workspace Repository — Manages workspace, session, note, and bookmark state.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Note:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class Bookmark:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    target_type: str = ""  # "layer", "neuron", "token", "session"
    target_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionRecord:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: str = "default_workspace"
    model_name: str = "gpt2"
    prompt: str = ""
    generated_text: str = ""
    created_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
    token_ids: list[int] = field(default_factory=list)
    token_texts: list[str] = field(default_factory=list)
    num_layers: int = 12
    num_heads: int = 12
    experiment_ids: list[str] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    bookmarks: list[Bookmark] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.closed_at is None


@dataclass
class WorkspaceRecord:
    id: str = "default_workspace"
    name: str = "Default Workspace"
    description: str = "Primary interpretability workspace"
    models: list[str] = field(default_factory=lambda: ["gpt2"])
    sessions: list[str] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    bookmarks: list[Bookmark] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class SessionRepository:
    """Manages workspaces and sessions."""

    def __init__(self):
        self._workspaces: dict[str, WorkspaceRecord] = {
            "default_workspace": WorkspaceRecord()
        }
        self._sessions: dict[str, SessionRecord] = {}

    def get_workspace(self, workspace_id: str = "default_workspace") -> WorkspaceRecord:
        if workspace_id not in self._workspaces:
            self._workspaces[workspace_id] = WorkspaceRecord(id=workspace_id, name=f"Workspace {workspace_id}")
        return self._workspaces[workspace_id]

    def list_workspaces(self) -> list[WorkspaceRecord]:
        return list(self._workspaces.values())

    def create_session(self, workspace_id: str = "default_workspace", model_name: str = "gpt2", prompt: str = "") -> SessionRecord:
        ws = self.get_workspace(workspace_id)
        session = SessionRecord(workspace_id=workspace_id, model_name=model_name, prompt=prompt)
        self._sessions[session.session_id] = session
        if session.session_id not in ws.sessions:
            ws.sessions.append(session.session_id)
        return session

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        return self._sessions.get(session_id)

    def list_sessions(self, workspace_id: Optional[str] = None) -> list[SessionRecord]:
        if workspace_id:
            return [s for s in self._sessions.values() if s.workspace_id == workspace_id]
        return list(self._sessions.values())

    def add_note(self, session_id: str, title: str, content: str) -> Note:
        session = self.get_session(session_id)
        note = Note(title=title, content=content)
        if session:
            session.notes.append(note)
        return note

    def add_bookmark(self, session_id: str, title: str, target_type: str, target_id: str, metadata: dict = None) -> Bookmark:
        session = self.get_session(session_id)
        bookmark = Bookmark(title=title, target_type=target_type, target_id=target_id, metadata=metadata or {})
        if session:
            session.bookmarks.append(bookmark)
        return bookmark


# Singleton
session_repo = SessionRepository()
