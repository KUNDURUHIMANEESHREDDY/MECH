"""Scientific Validation Engine.

Validation Pipeline: Discovery ➔ Validation ➔ Reproduction ➔ Confidence ➔ Knowledge Base.
"""

from __future__ import annotations

from typing import Any, Dict

from .benchmark_runner import MechanisticBenchmarkRunner
from .confidence_engine import ScientificConfidenceEngine
from .peer_review import AutomatedPeerReviewer
from .reproducibility import DiscoveryReproductionEngine


class ScientificValidationEngine:
    """Fail-closed scientific validation boundary."""

    def __init__(self) -> None:
        self.reproducibility = DiscoveryReproductionEngine()
        self.confidence_engine = ScientificConfidenceEngine()
        self.peer_reviewer = AutomatedPeerReviewer()
        self.benchmark_runner = MechanisticBenchmarkRunner()

    def validate_discovery(self, discovery_id: str, hypothesis_statement: str) -> Dict[str, Any]:
        """Fail closed until a live validation executor is connected.

        The legacy reproduction, benchmark, peer-review, and confidence
        helpers return deterministic reference values.  They must not be
        combined into a scientific validation verdict or handed to Society
        publication.
        """
        return {
            "discovery_id": discovery_id,
            "status": "unavailable",
            "provenance": "unavailable",
            "validation_eligible": False,
            "publication_eligible": False,
            "validated": False,
            "reason": (
                "No live scientific validation executor is connected; "
                "reference validation fields cannot support a verdict."
            ),
        }
