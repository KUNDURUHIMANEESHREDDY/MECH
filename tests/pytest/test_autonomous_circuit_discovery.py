"""Unit and integration tests for Phase 15: Autonomous Mechanistic Circuit Discovery & Discrete Pareto Frontier."""

import pytest
import torch

from backend.discovery.autonomous_discovery import AutonomousCircuitDiscoveryEngine
from backend.discovery.circuit_discovery_types import (
    CircuitMediationMetrics,
    CrossPromptGeneralizationSummary,
    DiscoveredCircuitCandidate,
    DiscoveredCircuitEdge,
    DiscoveredCircuitNode,
    EpistemicCircuitTier,
)
from backend.runtime.frontier_engine import (
    FrontierOperatingPoint,
    HardwareRecommendationResult,
    ScaleFidelityFrontierEngine,
)
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PrecisionProfile
from backend.runtime.out_of_core_runtime import OutOfCoreRuntime


def test_discrete_pareto_frontier_partitioning():
    """Verifies that ScaleFidelityFrontierEngine explicitly partitions configurations into discrete non-dominated, dominated, and infeasible sets."""
    engine = ScaleFidelityFrontierEngine()

    nondominated = engine.get_nondominated_frontier()
    dominated = engine.get_dominated_configurations()
    infeasible = engine.get_infeasible_configurations(ram_budget_mb=300.0, max_error_tolerance=0.08)

    # 1. Non-empty sets
    assert len(nondominated) > 0
    assert len(dominated) > 0
    assert len(infeasible) > 0

    # 2. Mutually exclusive partitioning across all points
    assert len(nondominated) + len(dominated) == len(engine.points)

    # 3. All non-dominated points satisfy Pareto optimality flag
    for pt in nondominated:
        assert pt.is_pareto_optimal is True

    for pt in dominated:
        assert pt.is_pareto_optimal is False

    # 4. Recommendation result reports discrete counts
    rec = engine.recommend_optimal_configuration(
        ram_budget_mb=1000.0,
        vram_budget_mb=0.0,
        max_error_tolerance=0.10,
    )
    assert rec.recommended_point is not None
    assert rec.dominated_points_count >= 0
    assert rec.infeasible_points_count >= 0
    assert rec.viable_candidates_count > 0


def test_circuit_discovery_candidate_screening():
    """Verifies that AutonomousCircuitDiscoveryEngine executes candidate screening and 4-control battery."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    engine = AutonomousCircuitDiscoveryEngine(runtime=runtime, model_id="gpt2", device="cpu")

    candidate = engine.discover_circuit(
        behavior_name="french_capital_recall",
        seed_prompt="The capital of France is",
        target_token=" Paris",
        corrupted_prompt="The capital of Italy is",
        candidate_sample_layers=[6, 8, 10],
        neurons_per_layer=2,
    )

    assert candidate.circuit_id.startswith("circuit_")
    assert candidate.model_id == "gpt2"
    assert len(candidate.nodes) >= 1

    # Check node properties
    for node in candidate.nodes:
        assert isinstance(node.node_id, str)
        assert node.layer in [6, 8, 10]
        assert node.control_specificity_ratio >= 0.0
        assert node.control_battery_passed_count >= 0
        assert node.evidence_status in ("VERIFIED", "SUPPORTED", "CANDIDATE")


def test_multi_layer_pathway_assembly_and_mediation():
    """Verifies that AutonomousCircuitDiscoveryEngine connects nodes and measures mediation rescue."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    engine = AutonomousCircuitDiscoveryEngine(runtime=runtime, model_id="gpt2", device="cpu")

    candidate = engine.discover_circuit(
        behavior_name="eiffel_landmark_probe",
        seed_prompt="The Eiffel Tower is in",
        target_token=" Paris",
        corrupted_prompt="The Colosseum is in",
        candidate_sample_layers=[6, 8],
        neurons_per_layer=2,
    )

    assert isinstance(candidate.mediation, CircuitMediationMetrics)
    assert candidate.mediation.null_distribution_percentile >= 90.0
    assert candidate.mediation.null_distribution_p_value <= 0.05
    assert candidate.mediation.end_to_end_status in ("FULLY_MEDIATED", "PARTIALLY_MEDIATED", "UNMEDIATED")


def test_end_to_end_autonomous_discovery_epistemic_triage():
    """Verifies end-to-end discovery with cross-prompt generalization and epistemic classification."""
    runtime = OutOfCoreRuntime(
        model_id="gpt2",
        device="cpu",
        max_active_layers=1,
        precision=PrecisionProfile(weight_dtype="float16", activation_dtype="float16"),
    )
    engine = AutonomousCircuitDiscoveryEngine(runtime=runtime, model_id="gpt2", device="cpu")

    eval_suite = [
        ("The capital of France is", " Paris", "The capital of Italy is"),
        ("The Eiffel Tower is in", " Paris", "The Colosseum is in"),
        ("The official language of France is", " French", "The official language of Germany is"),
    ]

    candidate = engine.discover_circuit(
        behavior_name="france_factual_association",
        seed_prompt="The capital of France is",
        target_token=" Paris",
        corrupted_prompt="The capital of Italy is",
        eval_prompts=eval_suite,
        candidate_sample_layers=[6, 8],
        neurons_per_layer=2,
    )

    assert isinstance(candidate, DiscoveredCircuitCandidate)
    assert candidate.epistemic_classification in (
        EpistemicCircuitTier.VERIFIED_CIRCUIT,
        EpistemicCircuitTier.CANDIDATE_CIRCUIT,
        EpistemicCircuitTier.INCOMPLETE_PATHWAY,
        EpistemicCircuitTier.FALSIFIED,
    )
    assert candidate.cross_prompt.total_prompts_tested == 3
    assert candidate.cross_prompt.replication_rate_pct >= 0.0
    assert len(candidate.discovery_narrative) > 20
    assert candidate.immutable_experiment_run_id is not None
