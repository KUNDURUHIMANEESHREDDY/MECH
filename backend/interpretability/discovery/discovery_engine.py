"""Central Mechanistic Discovery Engine (Sprint 4 & 5).

The registered extension engines currently provide reference/synthetic fields.
They remain available for non-scientific catalog inspection, but the active
orchestrator fails closed until a live discovery executor is connected.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..semantics.auto_circuit_namer import AutoCircuitNamerEngine
from .auto_hypothesis_tester import AutomaticHypothesisTesterEngine
from .automated_regression_suite import AutomatedRegressionSuite
from .benchmark_registry import BenchmarkRegistry
from .circuit_evolution import CircuitEvolutionEngine
from .confidence_calibration import ConfidenceCalibrationEngine
from .confidence_scorer import PlatformConfidenceEngine
from .cross_model_circuits import CrossModelCircuitsEngine
from .discovery_algorithm_registry import DiscoveryAlgorithmRegistry
from .discovery_lifecycle import DiscoveryLifecycleState
from .discovery_quality_score import DiscoveryQualityScoreEngine
from .evidence_ranker import EvidenceRankerEngine
from .feature_genealogy import FeatureGenealogyEngine
from .feature_universality import FeatureUniversalityEngine
from .induction_head_detector import InductionHeadDetector
from .ioi_benchmark import IOIBenchmarkSuite
from .mechanism_benchmarking import MechanismBenchmarkingEngine
from .mechanism_registry import MechanismRegistry
from .paper_replicator import PublishedPaperReplicator
from .superposition_analyzer import SuperpositionAnalyzerEngine


class DiscoveryEngine:
    """Central orchestrator driving autonomous mechanistic scientific discovery."""

    def __init__(self) -> None:
        self.hypothesis_tester = AutomaticHypothesisTesterEngine()
        self.circuit_evolution = CircuitEvolutionEngine()
        self.cross_model = CrossModelCircuitsEngine()
        self.feature_genealogy = FeatureGenealogyEngine()
        self.circuit_namer = AutoCircuitNamerEngine()
        self.evidence_ranker = EvidenceRankerEngine()
        self.confidence_scorer = PlatformConfidenceEngine()

        # AI 3 Real Mechanistic Science Extensions
        self.ioi_benchmark = IOIBenchmarkSuite()
        self.induction_detector = InductionHeadDetector()
        self.superposition_analyzer = SuperpositionAnalyzerEngine()
        self.universality_engine = FeatureUniversalityEngine()
        self.paper_replicator = PublishedPaperReplicator()
        self.mechanism_benchmarking = MechanismBenchmarkingEngine()
        self.confidence_calibration = ConfidenceCalibrationEngine()

        # AI 3 Research Framework Extensions
        self.benchmark_registry = BenchmarkRegistry()
        self.mechanism_registry = MechanismRegistry()
        self.regression_suite = AutomatedRegressionSuite()
        self.quality_scorer = DiscoveryQualityScoreEngine()
        self.algorithm_registry = DiscoveryAlgorithmRegistry()

        self._register_default_algorithms()
        self.discoveries: Dict[str, DiscoveryLifecycleState] = {}

    def _register_default_algorithms(self) -> None:
        self.algorithm_registry.register_algorithm("IOI", lambda p: self.ioi_benchmark.run_ioi_eval())
        self.algorithm_registry.register_algorithm("Induction", lambda p: {"heads": self.induction_detector.detect_induction_heads()})
        self.algorithm_registry.register_algorithm("Superposition", lambda p: self.superposition_analyzer.analyze_superposition())
        self.algorithm_registry.register_algorithm("PaperReplication", lambda p: self.paper_replicator.replicate_paper())

    def discover_and_orchestrate(self, hypothesis_statement: str) -> Dict[str, Any]:
        """Return an explicit unavailable envelope until a live executor exists.

        The registered discovery algorithms currently return reference or
        synthetic measurements.  They must not be promoted into a lifecycle
        that looks validated or publication-ready, so the orchestrator stops
        at evidence collection and exposes no synthetic result fields.
        """
        disc_id = f"disc_{hash(hypothesis_statement) & 0xffffffff:08x}"
        lifecycle = DiscoveryLifecycleState(discovery_id=disc_id,
                                            title=hypothesis_statement[:50])
        self.discoveries[disc_id] = lifecycle
        lifecycle.transition_to(
            "Evidence Collection",
            "No live discovery executor is connected; reference fields are not scientific evidence",
        )

        return {
            "discovery_id": disc_id,
            "status": "unavailable",
            "provenance": "synthetic",
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "No live discovery executor is connected; synthetic/reference "
                "fields cannot enter Society validation or publication."
            ),
            "lifecycle": lifecycle.to_dict(),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "active",
            "live_discovery_available": False,
            "scientific_evidence_status": "unavailable",
            "discoveries_count": len(self.discoveries),
            "mechanisms_count": len(self.mechanism_registry.list_mechanisms()),
            "registered_algorithms": self.algorithm_registry.list_algorithms(),
        }
