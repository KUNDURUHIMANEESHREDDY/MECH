"""Unit and integration tests for Statistical ACDC Circuit Discovery & Causal Verification."""

import pytest
import torch

from backend.discovery.statistical_acdc import (
    StatisticalACDCEngine,
    ACDCSparseCircuit,
    CircuitGraphEdge,
    CircuitGraphNode,
    EdgePruningStatus,
    GraphComponentType,
)
from backend.discovery.acdc_verification_pipeline import (
    ACDCCausalVerificationOrchestrator,
    VerifiedACDCCircuitReport,
)
from backend.discovery.circuit_discovery_types import EpistemicCircuitTier
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PrecisionProfile
from backend.runtime.out_of_core_runtime import OutOfCoreRuntime


def test_statistical_acdc_computational_graph_initialization():
    """Verifies that StatisticalACDCEngine builds a full computational graph with all heads."""
    engine = StatisticalACDCEngine(model_id="gpt2", device="cpu")
    nodes, edges = engine.build_full_computational_graph(
        layers=[6, 7, 8],
        include_all_heads=True,
        top_neurons_per_layer=8,
    )

    assert len(nodes) > 0
    assert len(edges) > 0

    # Verify presence of Embed, all heads, top neurons, and Output
    node_types = {n.component_type for n in nodes}
    assert GraphComponentType.EMBED in node_types
    assert GraphComponentType.ATTENTION_HEAD in node_types
    assert GraphComponentType.MLP_NEURON in node_types
    assert GraphComponentType.OUTPUT_HEAD in node_types

    # Should have 12 heads per layer * 3 layers = 36 head nodes
    head_nodes = [n for n in nodes if n.component_type == GraphComponentType.ATTENTION_HEAD]
    assert len(head_nodes) == 36

    # Should have 8 neurons per layer * 3 layers = 24 neuron nodes
    neuron_nodes = [n for n in nodes if n.component_type == GraphComponentType.MLP_NEURON]
    assert len(neuron_nodes) == 24

    # Verify all edges are strictly forward directed
    for e in edges:
        assert e.source_layer < e.target_layer


def test_statistical_acdc_edge_significance_testing():
    """Verifies that permutation-based significance testing works on edges."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    engine = StatisticalACDCEngine(runtime=runtime, model_id="gpt2", device="cpu", n_permutations=10)

    # Build a small graph for testing
    nodes, edges = engine.build_full_computational_graph(
        layers=[6, 8],
        include_all_heads=True,
        top_neurons_per_layer=4,
    )

    # Test a single edge
    test_edge = edges[0]
    result_edge = engine.test_edge_significance(
        test_edge,
        clean_prompt="The capital of France is",
        corrupted_prompt="The capital of Italy is",
        target_token=" Paris",
    )

    assert isinstance(result_edge, CircuitGraphEdge)
    assert result_edge.counterfactual_divergence >= 0.0
    assert 0.0 <= result_edge.p_value <= 1.0
    assert result_edge.ci_lower <= result_edge.ci_upper
    assert result_edge.effect_size != 0.0 or result_edge.p_value > 0.05


def test_statistical_acdc_multiple_comparison_correction():
    """Verifies that multiple comparison correction (Holm) is applied correctly."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    engine = StatisticalACDCEngine(
        runtime=runtime,
        model_id="gpt2",
        device="cpu",
        n_permutations=10,
        alpha=0.05,
        correction="holm",
        min_effect_size=0.2,
    )

    nodes, edges = engine.build_full_computational_graph(
        layers=[6],
        include_all_heads=True,
        top_neurons_per_layer=2,
    )

    # Run significance tests on all edges
    for edge in edges:
        engine.test_edge_significance(
            edge,
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Italy is",
            target_token=" Paris",
        )

    # Apply correction
    corrected_edges = engine.apply_multiple_comparison_correction(edges)

    # At least some edges should be tested
    assert len(corrected_edges) == len(edges)
    # Check that significant flag respects correction
    for e in corrected_edges:
        if e.significant:
            assert e.p_value < 0.05
            assert abs(e.effect_size) >= 0.2


def test_statistical_acdc_discover_sparse_circuit():
    """Verifies full statistical ACDC pipeline produces a sparse circuit with stats."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    engine = StatisticalACDCEngine(
        runtime=runtime,
        model_id="gpt2",
        device="cpu",
        n_permutations=5,
        alpha=0.05,
        correction="holm",
        min_effect_size=0.2,
    )

    circuit = engine.discover_sparse_circuit(
        behavior_name="french_capital_statistical",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        corrupted_prompt="The capital of Italy is",
        target_layers=[6],
        pruning_threshold_tau=0.010,
        top_neurons_per_layer=4,
    )

    assert isinstance(circuit, ACDCSparseCircuit)
    assert circuit.initial_edges_count > 0
    assert circuit.n_permutations == 5
    assert circuit.alpha == 0.05
    assert circuit.correction == "holm"
    assert circuit.retained_edges_count >= 0
    assert circuit.pruned_edges_count >= 0
    assert circuit.sparsity_ratio_pct >= 0.0
    assert len(circuit.retained_edges) == circuit.retained_edges_count

    # Check that retained edges have statistical fields populated
    for e in circuit.retained_edges:
        assert e.significant is True
        assert e.p_value < 0.05
        assert e.counterfactual_divergence >= 0.010


def test_statistical_acdc_with_causal_and_mediation_verification():
    """Verifies the unified statistical ACDC discovery + causal intervention + mediation pipeline."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    orchestrator = ACDCCausalVerificationOrchestrator(
        runtime=runtime,
        model_id="gpt2",
        device="cpu",
        n_permutations=5,
        alpha=0.05,
        correction="holm",
        min_effect_size=0.2,
    )

    report = orchestrator.run_acdc_and_causal_verification(
        behavior_name="country_capital_statistical_verification",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        corrupted_prompt="The capital of Italy is",
        target_layers=[6],
        pruning_threshold_tau=0.012,
    )

    assert isinstance(report, VerifiedACDCCircuitReport)
    assert report.active_nodes_count >= 0
    assert report.retained_edges_count >= 0
    assert report.control_specificity_ratio >= 0.0
    assert report.control_specificity_ci[0] <= report.control_specificity_ci[1]
    assert 0.0 <= report.control_p_value <= 1.0
    assert 0.0 <= report.mediation_rescue_fraction <= 1.0
    assert 0.0 <= report.cross_prompt_replication_pct <= 100.0
    assert report.cross_prompt_ci[0] <= report.cross_prompt_ci[1]
    assert report.epistemic_status in (
        EpistemicCircuitTier.VERIFIED_CIRCUIT,
        EpistemicCircuitTier.CANDIDATE_CIRCUIT,
        EpistemicCircuitTier.INCOMPLETE_PATHWAY,
        EpistemicCircuitTier.FALSIFIED,
    )
    assert len(report.verification_narrative) > 30


def test_out_of_core_statistical_acdc_discovery():
    """Verifies that statistical ACDC discovery works with Out-of-Core runtime."""
    runtime = OutOfCoreRuntime(
        model_id="gpt2",
        device="cpu",
        max_active_layers=1,
        precision=PrecisionProfile(weight_dtype="float16", activation_dtype="float16"),
    )
    orchestrator = ACDCCausalVerificationOrchestrator(
        runtime=runtime,
        model_id="gpt2",
        device="cpu",
        n_permutations=3,
        alpha=0.05,
        correction="holm",
        min_effect_size=0.2,
    )

    # Test that engine initializes and can build graph
    nodes, edges = orchestrator.acdc_engine.build_full_computational_graph(
        layers=[6],
        include_all_heads=True,
        top_neurons_per_layer=4,
    )
    assert len(nodes) > 0
    assert len(edges) > 0