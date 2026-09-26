"""Scientific Validation Engine.

Validation Pipeline: Discovery ➔ Validation ➔ Reproduction ➔ Confidence ➔ Knowledge Base.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.agents.evidence_policy import discovery_is_live, field_map

from .benchmark_runner import MechanisticBenchmarkRunner
from .confidence_engine import ScientificConfidenceEngine
from .peer_review import AutomatedPeerReviewer
from .reproducibility import DiscoveryReproductionEngine


class ScientificValidationEngine:
    """Live held-out validation boundary with fail-closed fallback."""

    def __init__(self) -> None:
        self.reproducibility = DiscoveryReproductionEngine()
        self.confidence_engine = ScientificConfidenceEngine()
        self.peer_reviewer = AutomatedPeerReviewer()
        self.benchmark_runner = MechanisticBenchmarkRunner()

    def validate_discovery(self, discovery_id: str, hypothesis_statement: str,
                           discovery_result: Optional[Dict[str, Any]] = None
                           ) -> Dict[str, Any]:
        """Validate a live discovery by independent held-out re-measurement.

        The legacy reproduction, benchmark, peer-review, and confidence
        helpers return deterministic reference values.  They must not be
        combined into a scientific validation verdict or handed to Society
        publication, so they are not consulted here: the verdict comes only
        from the live validation executor.  Without a live discovery result
        the engine fails closed.
        """
        candidate = discovery_result
        if not discovery_is_live(candidate):
            try:
                from backend.interpretability.discovery.live_discovery import (
                    recall,
                )
                candidate = recall(discovery_id)
            except Exception:
                candidate = None
        if discovery_is_live(candidate):
            try:
                from .live_validation import LiveValidationExecutor
                return LiveValidationExecutor().validate(candidate)
            except Exception as exc:
                return {
                    "discovery_id": discovery_id,
                    "status": "error",
                    "provenance": "unavailable",
                    "field_provenance": field_map(
                        ("discovery_id", "status", "validated", "error"),
                        "unavailable",
                    ),
                    "validation_eligible": False,
                    "publication_eligible": False,
                    "validated": False,
                    "error": str(exc)[:500],
                }
        return {
            "discovery_id": discovery_id,
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(
                ("discovery_id", "status", "validated", "reason"),
                "unavailable",
            ),
            "validation_eligible": False,
            "publication_eligible": False,
            "validated": False,
            "reason": (
                "No live scientific validation executor is connected; "
                "reference validation fields cannot support a verdict."
            ),
        }
