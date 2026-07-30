"""Scientific Validation Engine.

Validation Pipeline: Discovery ➔ Validation ➔ Reproduction ➔ Confidence ➔ Knowledge Base.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List

from .benchmark_runner import MechanisticBenchmarkRunner
from .confidence_engine import ScientificConfidenceEngine
from .peer_review import AutomatedPeerReviewer
from .reproducibility import DiscoveryReproductionEngine


class ScientificValidationEngine:
    """Central scientific validation engine verifying discoveries prior to knowledge base entry."""

    def __init__(self) -> None:
        self.reproducibility = DiscoveryReproductionEngine()
        self.confidence_engine = ScientificConfidenceEngine()
        self.peer_reviewer = AutomatedPeerReviewer()
        self.benchmark_runner = MechanisticBenchmarkRunner()

    def validate_discovery(self, discovery_id: str, hypothesis_statement: str) -> Dict[str, Any]:
        # 1. Validation & Reproduction
        repro = self.reproducibility.reproduce_discovery(discovery_id=discovery_id)

        # 2. Benchmark Execution
        bench = self.benchmark_runner.run_benchmark(benchmark_id=f"bench_{discovery_id}")

        # 3. Peer Review Critique
        review = self.peer_reviewer.review_manuscript(title=hypothesis_statement)

        # 4. Statistical Confidence Scoring
        conf = self.confidence_engine.compute_confidence(reproducibility_score=repro["reproducibility_score"])

        is_validated = conf["confidence_score"] >= 0.85 and review["decision"] == "Accept"

        return {
            "discovery_id": discovery_id,
            "validated": is_validated,
            "reproduction": repro,
            "benchmark": bench,
            "peer_review": review,
            "confidence": conf,
            "validated_at": _dt.datetime.utcnow().isoformat() + "Z",
        }
