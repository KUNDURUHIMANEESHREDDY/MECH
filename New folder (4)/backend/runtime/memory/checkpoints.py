"""Debugger Checkpoint & Portable State System."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional


class DebuggerCheckpoint:
    """Portable snapshot of a forward debugger session."""

    def __init__(
        self,
        checkpoint_id: str,
        session_id: str,
        layer: int,
        activations: Optional[Dict[str, Any]] = None,
        patches: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.session_id = session_id
        self.layer = layer
        self.activations = activations or {}
        self.patches = patches or []
        self.created_at = _dt.datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "layer": self.layer,
            "activations": self.activations,
            "patches": self.patches,
            "created_at": self.created_at,
        }


class CheckpointSystem:
    """Manages saving and restoring portable debugger checkpoints."""

    def __init__(self) -> None:
        self._checkpoints: Dict[str, DebuggerCheckpoint] = {}

    def save_checkpoint(
        self,
        checkpoint_id: str,
        session_id: str,
        layer: int,
        activations: Optional[Dict[str, Any]] = None,
        patches: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        ckpt = DebuggerCheckpoint(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            layer=layer,
            activations=activations,
            patches=patches,
        )
        self._checkpoints[checkpoint_id] = ckpt
        return ckpt.to_dict()

    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        ckpt = self._checkpoints.get(checkpoint_id)
        return ckpt.to_dict() if ckpt else None
