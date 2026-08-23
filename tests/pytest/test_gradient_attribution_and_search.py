"""Unit and integration tests for Phase 17: Gradient Attribution & Search-to-Verify Discovery."""

import pytest
import torch

from backend.discovery.gradient_attribution_engine import (
    AttributionMethod,
    GradientAttributionScanner,
    GradientAttributionScore,
)
from backend.discovery.search_to_verify_orchestrator import (
    SearchToVerifyDiscoveryOrchestrator,
    SearchToVerifyDiscoveryReport,
    VerifiedComponentEvidence,
)
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PrecisionProfile
from backend.runtime.out_of_core_runtime import OutOfCoreRuntime


def test_gradient_and_grad_x_act_computation():
    """Verifies that GradientAttributionScanner extracts autograd gradients and computes Grad×Act scores."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    scanner = GradientAttributionScanner(runtime=runtime, model_id="gpt2", device="cpu")

    attributions = scanner.scan_layer_attributions(
        clean_prompt="The capital of France is",
        target_token=" Paris",
        target_layers=[7, 8],
    )

    # Scanned 2 layers of 3072 neurons = 6144 components
    assert len(attributions) == 6144
    assert attributions[0].primary_attribution_rank == 1
    assert attributions[-1].primary_attribution_rank == 6144

    # Top candidate should have non-zero activation and gradient
    top = attributions[0]
    assert isinstance(top.component_id, str)
    assert top.layer in [7, 8]
    assert isinstance(top.raw_activation, float)
    assert isinstance(top.raw_gradient, float)
    assert isinstance(top.grad_x_act_score, float)
    assert top.evidence_tier == "GRADIENT_SEARCH_CANDIDATE"


def test_integrated_gradients_and_attribution_patching():
    """Verifies that Integrated Gradients and Attribution Patching scores are computed accurately."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    scanner = GradientAttributionScanner(runtime=runtime, model_id="gpt2", device="cpu")

    attributions = scanner.scan_layer_attributions(
        clean_prompt="The capital of France is",
        target_token=" Paris",
        target_layers=[8],
        corrupted_prompt="The capital of Italy is",
        compute_integrated_gradients=True,
    )

    assert len(attributions) == 3072
    top = attributions[0]
    assert top.integrated_grad_score is not None
    assert top.attribution_patching_score is not None


def test_search_to_verify_orchestration_and_evidence_hierarchy():
    """Verifies that SearchToVerifyDiscoveryOrchestrator filters candidates and executes causal interventions with controls."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    orchestrator = SearchToVerifyDiscoveryOrchestrator(runtime=runtime, model_id="gpt2", device="cpu")

    report = orchestrator.run_search_to_verify_discovery(
        behavior_name="french_capital_search_to_verify",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        corrupted_prompt="The capital of Italy is",
        target_layers=[7, 8, 9],
        top_k_candidates=4,
    )

    assert isinstance(report, SearchToVerifyDiscoveryReport)
    assert report.total_components_screened == 3 * 3072
    assert report.top_candidates_selected == 4
    assert len(report.verified_components) == 4

    for comp in report.verified_components:
        assert isinstance(comp.component_id, str)
        assert isinstance(comp.causal_delta_z, float)
        assert comp.control_specificity_ratio >= 0.0
        assert comp.controls_passed_count >= 0
        assert comp.epistemic_evidence_tier in (
            "CAUSALLY_VERIFIED_CIRCUIT_NODE",
            "CAUSALLY_SUPPORTED_NODE",
            "FALSIFIED_GRADIENT_HEURISTIC",
        )

    assert len(report.epistemic_summary) > 20


def test_out_of_core_gradient_search_to_verify():
    """Verifies that the search-to-verify pipeline operates seamlessly with Out-of-Core runtimes."""
    runtime = OutOfCoreRuntime(
        model_id="gpt2",
        device="cpu",
        max_active_layers=1,
        precision=PrecisionProfile(weight_dtype="float16", activation_dtype="float16"),
    )
    orchestrator = SearchToVerifyDiscoveryOrchestrator(runtime=runtime, model_id="gpt2", device="cpu")

    report = orchestrator.run_search_to_verify_discovery(
        behavior_name="ooc_search_to_verify",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        target_layers=[8],
        top_k_candidates=2,
    )

    assert isinstance(report, SearchToVerifyDiscoveryReport)
    assert report.model_id == "gpt2"
    assert report.top_candidates_selected == 2
    assert len(report.verified_components) == 2
