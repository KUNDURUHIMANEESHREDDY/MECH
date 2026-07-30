"""Epic 3: Session Manager — every inference belongs to a session.

Sessions are backed by SQLite for persistence across restarts.
A session bundles:
    - model reference
    - prompt + generated text
    - all captured activations
    - profiling data
    - timestamps
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from .event_bus import bus, SESSION_CREATED, SESSION_CLOSED
from .errors import SessionNotFoundError


_DB_PATH = os.environ.get("MODEL_EXPLORER_DB", str(Path(__file__).parent.parent / "runtime.db"))


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    prompt: str = ""
    generated_text: str = ""
    created_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
    token_ids: list[int] = field(default_factory=list)
    token_texts: list[str] = field(default_factory=list)
    num_layers: int = 0
    num_heads: int = 0
    activations: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    def close(self) -> None:
        self.closed_at = time.time()

    def to_row(self) -> tuple:
        return (
            self.session_id,
            self.model_name,
            self.prompt,
            self.generated_text,
            self.created_at,
            self.closed_at or 0.0,
            json.dumps(self.token_ids),
            json.dumps(self.token_texts),
            self.num_layers,
            self.num_heads,
            json.dumps(self.metadata),
        )

    @classmethod
    def from_row(cls, row: tuple) -> "Session":
        return cls(
            session_id=row[0],
            model_name=row[1],
            prompt=row[2],
            generated_text=row[3],
            created_at=row[4],
            closed_at=row[5] if row[5] else None,
            token_ids=json.loads(row[6]),
            token_texts=json.loads(row[7]),
            num_layers=row[8],
            num_heads=row[9],
            metadata=json.loads(row[10]),
        )


class SessionManager:
    """SQLite-backed session manager."""

    def __init__(self, db_path: str = _DB_PATH):
        self._db_path = db_path
        self._sessions: dict[str, Session] = {}
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    model_name TEXT,
                    prompt TEXT,
                    generated_text TEXT,
                    created_at REAL,
                    closed_at REAL,
                    token_ids TEXT,
                    token_texts TEXT,
                    num_layers INTEGER,
                    num_heads INTEGER,
                    metadata TEXT
                )
            """)
            conn.commit()
        # Load existing sessions from DB
        self._load_from_db()

    def _load_from_db(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute("SELECT * FROM sessions").fetchall()
                for row in rows:
                    s = Session.from_row(row)
                    self._sessions[s.session_id] = s
        except Exception:
            pass

    def _persist(self, session: Session) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO sessions
                       (session_id, model_name, prompt, generated_text, created_at,
                        closed_at, token_ids, token_texts, num_layers, num_heads, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    session.to_row(),
                )
                conn.commit()
        except Exception:
            pass

    def create_session(
        self,
        model_name: str = "",
        prompt: str = "",
        **metadata,
    ) -> Session:
        session = Session(model_name=model_name, prompt=prompt, metadata=metadata)
        self._sessions[session.session_id] = session
        self._persist(session)
        bus.emit(SESSION_CREATED, session_id=session.session_id, model=model_name)
        return session

    def get_session(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} not found")
        return session

    def close_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        session.close()
        self._persist(session)
        bus.emit(SESSION_CLOSED, session_id=session_id)

    def update_session(self, session: Session) -> None:
        self._persist(session)

    def list_sessions(self) -> list[Session]:
        return list(self._sessions.values())

    def list_open_sessions(self) -> list[Session]:
        return [s for s in self._sessions.values() if s.is_open]

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                    conn.commit()
            except Exception:
                pass
            return True
        return False

    def clear(self) -> None:
        self._sessions.clear()
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM sessions")
                conn.commit()
        except Exception:
            pass

    def count(self) -> int:
        return len(self._sessions)


# Global singleton
session_manager = SessionManager()
