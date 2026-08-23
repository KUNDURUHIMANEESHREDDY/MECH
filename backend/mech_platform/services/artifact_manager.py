"""Backward-compatible re-export of ArtifactManager from backend.research_platform.services.artifact_manager."""

from backend.research_platform.services.artifact_manager import (
    ArtifactManager,
    Artifact,
)

__all__ = ["ArtifactManager", "Artifact"]
