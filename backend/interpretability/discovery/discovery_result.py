"""Standardized DiscoveryResult DTO.

Common data transfer object returned by all autonomous discovery algorithms.

`confidence` and `uncertainty` default to None rather than 0.90 / 0.10. The old
pair was self-consistent -- uncertainty was exactly `1 - confidence` -- which
made it look deliberate rather than invented, but constructing a discovery
without measuring it still asserted 0.90 confidence in it. A caller that omitted
the arguments got a DTO claiming to be 90% sure of a discovery nobody had
measured. None means unmeasured, and is distinguishable from a measured 0.90.
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
        confidence: float | None = None,
        uncertainty: float | None = None,
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
        # No invented default. This used to fall back to a specific prompt and
        # model ("The capital of France is" / "GPT-2 Small") for every caller
        # that omitted provenance, which asserts that a particular prompt was
        # run against a particular model. Provenance that was not supplied is
        # recorded as unavailable, so a downstream evidence gate can tell the
        # difference instead of reading a plausible fiction.
        self.provenance = provenance if provenance is not None else {
            "source": "unavailable",
            "measured": False,
            "reason": (
                "No provenance was supplied when this discovery was "
                "constructed, so its prompt, model, config, and seed are "
                "unknown."
            ),
        }
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
            "field_provenance": {
                "discovery_id": "reference",
                "discovery_type": "reference",
                "title": "reference",
                "evidence": "reference",
                "confidence": "reference",
                "uncertainty": "reference",
                "artifacts": "reference",
                "related_features": "reference",
                "related_circuits": "reference",
            },
            "artifacts": self.artifacts,
            "related_features": self.related_features,
            "related_circuits": self.related_circuits,
            "timestamp": self.timestamp,
        }
