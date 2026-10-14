"""Artifact Manager.

This file used to be a byte-identical copy of
``backend/mech_platform/services/artifact_manager.py``. Two copies of an
artifact manager is not a convenience: a bug fixed in one and not the other
is a bug that can be triggered by importing the other.

Rather than patch the same bug twice and leave two implementations to drift
again, this module delegates. The implementation lives in
``backend.mech_platform.services.artifact_manager``.
"""

from __future__ import annotations

from backend.mech_platform.services.artifact_manager import ArtifactManager

__all__ = ["ArtifactManager"]