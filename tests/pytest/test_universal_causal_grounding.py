"""Unit and integration tests for Phase 63: Universal Causal Grounding & Cross-Modal Physical Invariant Engine.

Updated to use the live-runner API — all engines now require a real or mock GPT-2 runner.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from backend.discovery.causal_invariant_engine import CausalInvariantEngine, InvariantScope
from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.counterfactual_invariance_engine import (
    CounterfactualInvarianceEngine,
    CounterfactualInvarianceRecord,
    CounterfactualInvarianceResult,
)
from backend.discovery.cross_modal_claim_policy import CrossModalClaimPolicy, GroundingClaimRecord, InvariantGroundingTier
from backend.discovery.cross_modal_mechanism_adapter import (
    CanonicalPrimitiveType, CrossModalMechanismAdapter, ModalityMappingRecord, ModalityType
)
from backend.discovery.physical_invariant_probe_engine import PhysicalInvariantProbeEngine, PhysicalProbeResult
from backend.discovery.universal_causal_grounding_orchestrator import (
    Phase63CausalGroundingCertificate, UniversalCausalGroundingOrchestrator
)
from backend.runtime.dynamic_prompt_sampler import DynamicProbe
from backend.runtime.gpt2_live_experiment_runner import (
    PerturbationResult, CounterfactualResult, FunctionalCorrespondenceResult
)


# ── Shared stubs ──────────────────────────────────────────────────────────────

def _make_probe(pid="probe_p63", cat="factual_recall") -> DynamicProbe:
    return DynamicProbe(
        probe_id=pid, category=cat,
        clean_prompt="The capital of France is",
        target_token=" Paris",
        corrupted_prompt="The capital of Germany is",
        distractor_token=" Berlin",
        target_layer_fraction=0.66,
        target_neuron_idx=412,
    )


def _make_runner(csf=0.96, inv_dist=0.01, fc_score=0.91, target_prob=0.72, target_rank=2) -> MagicMock:
    runner = MagicMock()
    runner._baseline_causal_effect.return_value = 1.45

    fwd = MagicMock()
    fwd.target_probability = target_prob
    fwd.target_rank = target_rank
    fwd.target_logit = 7.1
    fwd.layer_residuals = {}
    runner.runtime.forward.return_value = fwd
    runner.runtime.num_layers = 12
    runner.runtime.compute_logit_lens_trajectory.return_value = [
        {"layer": i, "target_logit": float(i) * 0.6, "top_token": " Paris", "target_rank": max(0, 50 - i * 4)}
        for i in range(13)
    ]

    def _phys(probe, ptype):
        return PerturbationResult(
            perturbation_type=ptype,
            baseline_delta_logit=1.45,
            perturbed_delta_logit=1.45 * csf,
            causal_signature_fidelity=csf,
            is_physically_grounded=csf >= 0.90,
            probe_id=probe.probe_id,
            target_token=probe.target_token,
        )
    runner.measure_physical_perturbation.side_effect = _phys
    runner.run_all_physical_perturbations.side_effect = lambda p: [
        _phys(p, t) for t in ["GAUSSIAN_NOISE", "WEIGHT_SCALING_075X", "SEQUENCE_EXTENSION", "LAYER_ABLATION"]
    ]

    def _cf(probe, ptype):
        return CounterfactualResult(
            perturbation_type=ptype,
            baseline_causal_effect=1.45,
            counterfactual_causal_effect=1.45 - inv_dist,
            invariance_distance=inv_dist,
            is_counterfactually_invariant=inv_dist <= 0.05,
            probe_id=probe.probe_id,
        )
    runner.measure_counterfactual_invariance.side_effect = _cf
    runner.run_all_counterfactual_invariances.side_effect = lambda p: [
        _cf(p, t) for t in ["NEURON_PERMUTATION", "ROUTING_SPARSITY", "DISTRACTOR_INSERTION"]
    ]

    fc = FunctionalCorrespondenceResult(
        probe_a_id="a", probe_b_id="b",
        category_a="factual_recall", category_b="arithmetic",
        trajectory_cosine_similarity=fc_score,
        peak_layer_overlap=0.67,
        causal_correspondence_score=fc_score,
        is_functionally_grounded=fc_score >= 0.50,
    )
    runner.measure_functional_correspondence.return_value = fc

    runner.compute_session_metrics.return_value = {
        "csf_mean": csf, "inv_mean": 1.0, "fc_mean": fc_score, "fur": 0.0, "n_probes": 3,
    }
    return runner


# ── Test 1: Cross-modal adapter (text-only) ──────────────────────────────────

def test_causal_mechanism_and_cross_modal_adapter():
    """Verifies CrossModalMechanismAdapter returns live scores for text probe pairs."""
    adapter = CrossModalMechanismAdapter()
    probe_a = _make_probe("pa", "factual_recall")
    probe_b = _make_probe("pb", "arithmetic")
    runner  = _make_runner(fc_score=0.91)

    # TEXT modality — must succeed with live runner
    m_text = adapter.map_modality_component(
        ModalityType.TEXT, "AttentionHead_L10_H4", CanonicalPrimitiveType.RELATIONAL_BINDING,
        probe_a=probe_a, probe_b=probe_b, runner=runner,
    )
    assert isinstance(m_text, ModalityMappingRecord)
    assert m_text.is_functionally_grounded is True
    assert m_text.causal_correspondence_score >= 0.50
    assert m_text.target_primitive == CanonicalPrimitiveType.RELATIONAL_BINDING

    # VISION / AUDIO must be rejected — GPT-2 is text-only
    with pytest.raises(ValueError, match="text-only"):
        adapter.map_modality_component(
            ModalityType.VISION, "ViT_Spatial_CrossAttention", CanonicalPrimitiveType.RELATIONAL_BINDING,
            probe_a=probe_a, probe_b=probe_b, runner=runner,
        )
    with pytest.raises(ValueError, match="text-only"):
        adapter.map_modality_component(
            ModalityType.AUDIO, "Audio_Temporal_ConvFilter", CanonicalPrimitiveType.RELATIONAL_BINDING,
            probe_a=probe_a, probe_b=probe_b, runner=runner,
        )


# ── Test 2: Physical probe & counterfactual invariance (live runner) ──────────

def test_physical_probe_and_counterfactual_invariance():
    """Verifies physical and counterfactual engines delegate to live runner."""
    probe  = _make_probe()
    runner = _make_runner(csf=0.96, inv_dist=0.01)

    # Physical perturbation
    probe_engine  = PhysicalInvariantProbeEngine()
    probe_results = probe_engine.probe_substrate_stability(probe=probe, runner=runner)

    assert len(probe_results) == 4
    for p in probe_results:
        assert isinstance(p, PhysicalProbeResult)
        assert p.is_physically_grounded is True
        assert p.causal_signature_fidelity >= 0.90
        assert p.probe_id == probe.probe_id

    # Counterfactual invariance
    cf_engine = CounterfactualInvarianceEngine()
    cf_results = cf_engine.test_counterfactual_invariance(probe=probe, runner=runner)

    assert len(cf_results) == 3
    for cf in cf_results:
        assert isinstance(cf, (CounterfactualInvarianceResult, CounterfactualInvarianceRecord))
        assert cf.is_counterfactually_invariant is True
        assert cf.invariance_distance <= 0.05
        assert cf.probe_id == probe.probe_id


# ── Test 3: 5-tier promotion ladder (unchanged logic — no runner needed) ──────

def test_5_tier_promotion_ladder_and_anti_universalization():
    """Verifies strict 5-tier ladder promotion and prevention of false universalization (FUR <= 2.0%)."""
    invariant_engine = CausalInvariantEngine()

    scope_univ = invariant_engine.classify_invariant_scope(
        cross_arch_fidelity=0.95, cross_modal_fidelity=0.92,
        is_substrate_divergent=False, is_modality_divergent=False,
    )
    assert scope_univ == InvariantScope.UNIVERSAL_INVARIANT

    scope_modal = invariant_engine.classify_invariant_scope(
        cross_arch_fidelity=0.95, cross_modal_fidelity=0.70,
        is_substrate_divergent=False, is_modality_divergent=True,
    )
    assert scope_modal == InvariantScope.MODALITY_CONDITIONED_INVARIANT

    scope_sub = invariant_engine.classify_invariant_scope(
        cross_arch_fidelity=0.75, cross_modal_fidelity=0.88,
        is_substrate_divergent=True, is_modality_divergent=False,
    )
    assert scope_sub == InvariantScope.SUBSTRATE_CONDITIONED_INVARIANT

    claim_graph = ClaimDependencyGraphEngine()
    policy = CrossModalClaimPolicy(claim_graph=claim_graph)

    claim_rec = policy.register_grounding_claim(
        mechanism_id="RELATIONAL_BINDING",
        target_tier=InvariantGroundingTier.TIER_5_PHYSICAL_GROUNDING_CONFIRMED,
        scope=scope_univ,
        csf=0.965,
    )

    assert isinstance(claim_rec, GroundingClaimRecord)
    assert claim_rec.is_false_universalization_prevented is True


# ── Test 4: Full orchestration with live runner ───────────────────────────────

def test_full_cross_modal_grounding_orchestration():
    """Verifies end-to-end Phase 63 grounding audit with mock live GPT-2 runner."""
    probes = [_make_probe(f"probe_{i}") for i in range(3)]
    runner = _make_runner(csf=0.96, inv_dist=0.01, fc_score=0.91)

    orchestrator = UniversalCausalGroundingOrchestrator()
    cert = orchestrator.run_full_audit(probes=probes, runner=runner, model_id="gpt2")

    assert isinstance(cert, Phase63CausalGroundingCertificate)
    assert cert.model_id == "gpt2"
    assert cert.n_probes_evaluated > 0
    assert cert.metrics["csf_mean"] >= 0.90
    assert cert.metrics["fur"] <= 0.02
    assert "CERT63-" in cert.certificate_id
    assert cert.thresholds_met is True
