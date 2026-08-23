"""Backward-compatible re-export of ProvenanceService from backend.research_platform.services.provenance_service."""

from backend.research_platform.services.provenance_service import (
    ProvenanceService,
    ProvenanceRecord,
)

__all__ = ["ProvenanceService", "ProvenanceRecord"]
