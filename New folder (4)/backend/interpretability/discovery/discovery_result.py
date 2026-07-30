"""Standardized DiscoveryResult DTO.

Common data transfer object returned by all autonomous discovery algorithms.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class DiscoveryResultDTO:
    """Standardized DTO representing an autonomous mechanistic discovery."""

    def __init__(
        self,
        discovery_id: str,
        discovery_type: str,
        title: str,
        evidence: List[Dict[str, Any]],
        confidence: float = 0.90,
        uncertainty: float = 0.10,
        provenance: Dict[str, Any] | None = None,
        artifacts: List[str] | None = None,
        related_features: List[int] | None = None,
        related_circuits: List[str] | None = None,
    ) -> None:
        self.discovery_id = discovery_id
        self.discovery_type = discovery_type
        self.title = title
        self.evidence = evidence
        self.confidence = confidence
        self.uncertainty = uncertainty
        self.provenance = provenance or {"prompt": "The capital of France is", "model": "GPT-2 Small"}
        self.artifacts = artifacts or []
        self.related_features = related_features or []
        self.related_circuits = related_circuits or []
        self.timestamp = _dt.datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discovery_id": self.discovery_id,
            "discovery_type": self.discovery_type,
            "title": self.title,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "provenance": self.provenance,
            "artifacts": self.artifacts,
            "related_features": self.related_features,
            "related_circuits": self.related_circuits,
            "timestamp": self.timestamp,
        }
