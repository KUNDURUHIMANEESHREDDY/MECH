"""Artifact Manager.

Manages plots, images, reports, and workspace bundles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ArtifactManager:
    """Manager for experiment artifacts."""

    def __init__(self) -> None:
        self._artifacts: Dict[str, Dict[str, Any]] = {}

    def store_artifact(self, artifact_id: str, artifact_type: str, data: Any) -> Dict[str, Any]:
        rec = {
            "artifact_id": artifact_id,
            "type": artifact_type,
            "data": data,
        }
        self._artifacts[artifact_id] = rec
        return rec

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        return self._artifacts.get(artifact_id)
