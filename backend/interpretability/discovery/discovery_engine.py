"""Central Mechanistic Discovery Engine (Sprint 4 & 5).

The registered extension engines provide reference/synthetic fields and remain
available for non-scientific catalog inspection only.  The active orchestrator
first tries the live causal discovery executor (real GPT-2 Small forward
passes); only when no live model is connected does it fail closed with an
explicit unavailable envelope.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.agents.evidence_policy import field_map

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
        """Run live causal discovery when a model is connected.

        The registered discovery algorithms return reference or synthetic
        measurements.  They must not be promoted into a lifecycle that looks
        validated or publication-ready, so when no live executor is available
        the orchestrator stops at evidence collection and exposes no synthetic
        result fields.
        """
        # Validate before anything else. An empty or non-string hypothesis
        # previously flowed into the run and produced an ordinary-looking
        # result, or raised an opaque TypeError from a slice operation.
        if not isinstance(hypothesis_statement, str) or not hypothesis_statement.strip():
            raise ValueError(
                "hypothesis_statement must be a non-empty string; got "
                f"{type(hypothesis_statement).__name__}"
            )

        disc_id = f"disc_{hash(hypothesis_statement) & 0xffffffff:08x}"
        lifecycle = DiscoveryLifecycleState(discovery_id=disc_id,
                                            title=hypothesis_statement[:50])
        self.discoveries[disc_id] = lifecycle

        try:
            from .live_discovery import LiveIOIDiscovery
        except Exception:
            LiveIOIDiscovery = None  # type: ignore[assignment]
        if LiveIOIDiscovery is not None and LiveIOIDiscovery.available():
            try:
                result = LiveIOIDiscovery().run(hypothesis_statement)
            except Exception as exc:
                # Record the failure before propagating. A raising executor
                # must not strand the discovery mid-run with no record of why
                # it stopped.
                lifecycle.record_failure(
                    f"live discovery executor raised: {exc}")
                raise
            if (isinstance(result, dict)
                    and result.get("status") == "completed"
                    and result.get("provenance") == "live"):
                lifecycle.transition_to(
                    "Evidence Collection",
                    "measured head effects/edges from live GPT-2 Small",
                )
                lifecycle.transition_to(
                    "Validation",
                    "released to Society validation with live provenance",
                )
                result["lifecycle"] = lifecycle.to_dict()
                return result
            lifecycle.transition_to(
                "Evidence Collection",
                f"live executor returned: {result.get('reason', 'no result')}",
            )
        else:
            lifecycle.transition_to(
                "Evidence Collection",
                "No live discovery executor is connected; reference fields are not scientific evidence",
            )

        return {
            "discovery_id": disc_id,
            "status": "unavailable",
            "provenance": "synthetic",
            "field_provenance": field_map(
                ("discovery_id", "status", "reason", "lifecycle"),
                "unavailable",
            ),
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "No live discovery executor is connected; synthetic/reference "
                "fields cannot enter Society validation or publication."
            ),
            "lifecycle": lifecycle.to_dict(),
        }

    def get_status(self) -> Dict[str, Any]:
        try:
            from .live_discovery import LiveIOIDiscovery
            live_available = LiveIOIDiscovery.available()
        except Exception:
            live_available = False
        evidence = "live" if live_available else "unavailable"
        return {
            "status": "active",
            "live_discovery_available": live_available,
            "scientific_evidence_status": evidence,
            "field_provenance": {
                "live_discovery_available": "live" if live_available else "unavailable",
                "scientific_evidence_status": evidence,
                "discoveries_count": "unavailable",
                "mechanisms_count": "reference",
                "registered_algorithms": "reference",
            },
            "discoveries_count": len(self.discoveries),
            "mechanisms_count": len(self.mechanism_registry.list_mechanisms()),
            "registered_algorithms": self.algorithm_registry.list_algorithms(),
        }
