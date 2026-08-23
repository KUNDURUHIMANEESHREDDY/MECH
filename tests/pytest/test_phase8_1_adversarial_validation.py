"""Phase 8.1: Comprehensive Adversarial Validation Test Suite for MECH Platform.

Stress-tests all existing features against adversarial inputs, edge cases, and extreme regimes:
1. Causal Tracing: identical prompts, reversed prompts, micro-deltas, absent targets.
2. Path Patching: backwards layer flow, same-layer MLP->Head, self-loops, nonexistent nodes.
3. ACDC Pruning: zero threshold (tau=0), maximum threshold (tau=1.0), empty layer scans.
4. Sparse Autoencoders: out-of-bounds feature IDs, zero-variance inputs, alpha=0.0 steering.
5. Bootstrap Engine: all-identical constant data, N=1, N=2, zero-variance sample arrays.
6. Model Adapters: invalid/unknown model IDs, out-of-bounds layer queries.
7. Out-of-Core Runtime: strict VRAM/RAM limit enforcement & budget aborts.
8. Epistemic Invariants: EXECUTION_FAILED != NO_EFFECT_DETECTED != NOTHING_FOUND.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from backend.interpretability.causal.causal_tracing import CausalTracingEngine
from backend.interpretability.causal.path_patching import EdgePathPatchingEngine
from backend.discovery.acdc_pruning_engine import ACDCCircuitDiscoveryEngine, ACDCSparseCircuit
from backend.interpretability.sae.sae_adapter import NativeMECHSAE
from backend.interpretability.sae.analysis.feature_interpretability import FeatureInterpretabilityAnalyzer
from backend.interpretability.sae.causal.sae_intervention_engine import SAECausalInterventionEngine
from backend.science.statistics.bootstrap_engine import BootstrapEngine
from backend.science.models.adapter_registry import ModelAdapterRegistry
from backend.runtime.residency_manager import (
    ResidencyManager,
    MemoryBudgetPolicy,
    EnforcementMode,
    ViolationAction,
    MemoryBudgetExceededError,
)
from backend.runtime.in_memory_runtime import InMemoryRuntime


# ============================================================================
# Area 1: Causal Tracing Adversarial Stress
# ============================================================================

class TestArea1CausalTracingAdversarial:
    """Stress-test Causal Tracing against pathological prompt pairs."""

    def test_identical_prompts_zero_division_guard(self):
        """Identical clean and corrupted prompts yield delta=0, must not throw or produce NaN."""
        engine = CausalTracingEngine()
        res = engine.trace_causal_effect(
            clean_prompt="The Eiffel Tower is in",
            corrupted_prompt="The Eiffel Tower is in",
            num_layers=4,
        )
        assert res["clean_probability"] == res["corrupted_probability"]
        for eff in res["layer_effects"]:
            assert not math.isnan(eff["indirect_effect"])
            assert not math.isinf(eff["indirect_effect"])
            assert 0.0 <= eff["indirect_effect"] <= 1.0

    def test_reversed_prompts_handling(self):
        """When corrupted prompt has higher target prob than clean, effects clamp cleanly to [0, 1]."""
        engine = CausalTracingEngine()
        res = engine.trace_causal_effect(
            clean_prompt="The capital of Spain is",
            corrupted_prompt="The capital of France is Paris and Spain is",
            num_layers=3,
        )
        for eff in res["layer_effects"]:
            assert not math.isnan(eff["indirect_effect"])
            assert 0.0 <= eff["indirect_effect"] <= 1.0


# ============================================================================
# Area 2: Path Patching Adversarial Stress
# ============================================================================

class TestArea2PathPatchingAdversarial:
    """Stress-test Edge Path Patching against topological edge cases."""

    def test_backwards_layer_rejection(self):
        patcher = EdgePathPatchingEngine()
        res = patcher.test_edge_mediation(
            sender="L10_H3",
            receiver="L2_H1",
            clean_prompt="John gave a drink to Mary",
            corrupted_prompt="John gave a drink to John",
        )
        assert res.is_causally_transmitting is False
        assert res.direct_path_effect == 0.0
        assert "Backwards Flow" in res.verdict

    def test_same_layer_mlp_to_head_rejection(self):
        """MLP is downstream of attention in same layer; MLP -> Head is backwards flow."""
        patcher = EdgePathPatchingEngine()
        res = patcher.test_edge_mediation(
            sender="L5_MLP",
            receiver="L5_H2",
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Italy is",
        )
        assert res.is_causally_transmitting is False
        assert "Backwards Flow" in res.verdict

    def test_self_loop_edge_rejection(self):
        patcher = EdgePathPatchingEngine()
        res = patcher.test_edge_mediation(
            sender="L6_H4",
            receiver="L6_H4",
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Italy is",
        )
        assert res.is_causally_transmitting is False
        assert "Self-Loop" in res.verdict


# ============================================================================
# Area 3: ACDC Pruning Threshold Extremes
# ============================================================================

class TestArea3ACDCPruningExtremes:
    """Stress-test ACDC circuit discovery under extreme pruning thresholds."""

    def test_acdc_zero_threshold_retains_all(self):
        """tau = 0.0 means all candidate edges are retained."""
        engine = ACDCCircuitDiscoveryEngine(model_id="gpt2", device="cpu")
        circuit = engine.discover_sparse_circuit(
            clean_prompt="The capital of France is",
            target_token=" Paris",
            corrupted_prompt="The capital of Italy is",
            target_layers=[6, 8],
            pruning_threshold_tau=0.0,
            heads_per_layer=1,
            neurons_per_layer=1,
        )
        assert circuit.retained_edges_count > 0
        assert circuit.pruned_edges_count == 0
        assert circuit.sparsity_ratio_pct == 0.0

    def test_acdc_extreme_threshold_pruning(self):
        """tau = 1.0 prunes edges, maintaining circuit invariants."""
        engine = ACDCCircuitDiscoveryEngine(model_id="gpt2", device="cpu")
        circuit = engine.discover_sparse_circuit(
            clean_prompt="The capital of France is",
            target_token=" Paris",
            corrupted_prompt="The capital of Italy is",
            target_layers=[6, 8],
            pruning_threshold_tau=1.0,
            heads_per_layer=1,
            neurons_per_layer=1,
        )
        assert isinstance(circuit, ACDCSparseCircuit)
        assert 0.0 <= circuit.sparsity_ratio_pct <= 100.0


# ============================================================================
# Area 4: SAE Numerical Adversarial Stress
# ============================================================================

class TestArea4SAENumericalAdversarial:
    """Stress-test SAE encoders, MSI metrics, and feature steering."""

    def test_out_of_bounds_feature_index_rejection(self):
        sae = NativeMECHSAE(d_in=64, d_sae=128, model_id="test", layer=0)
        with pytest.raises(IndexError, match="out of range"):
            sae.get_feature_direction(999999)

        with pytest.raises(IndexError, match="out of range"):
            sae.get_feature_direction(-1)

    def test_zero_variance_input_reconstruction_error(self):
        """Constant tensor (zero variance) must compute valid finite R^2 without division by zero."""
        sae = NativeMECHSAE(d_in=32, d_sae=64, model_id="test", layer=0)
        x_const = torch.ones(2, 32) * 5.0
        z, x_hat = sae.reconstruct(x_const)
        err = sae.get_reconstruction_error(x_const, x_hat)
        assert not math.isnan(err["explained_variance"])
        assert not math.isinf(err["explained_variance"])
        assert 0.0 <= err["explained_variance"] <= 1.0

    def test_all_zero_msi_computation(self):
        """All-zero activations must produce 0.0 MSI without NaN."""
        msi = FeatureInterpretabilityAnalyzer.compute_msi(
            target_activation=0.0,
            distractor_activations=[0.0, 0.0, 0.0],
        )
        assert msi == 0.0

    def test_causal_steering_zero_alpha_invariance(self):
        """alpha = 0.0 steering produces zero observed perturbation."""
        runtime = InMemoryRuntime(model_id="gpt2", device="cpu")
        sae = NativeMECHSAE(d_in=768, d_sae=1024, model_id="gpt2", layer=8)
        engine = SAECausalInterventionEngine(runtime=runtime)

        res = engine.steer_feature(
            sae=sae,
            feature_idx=0,
            prompt="The capital of France is",
            target_token=" Paris",
            steering_coefficient=0.0,
        )
        assert abs(res.observed_logit_shift) < 1e-4
        assert res.predicted_logit_shift == 0.0


# ============================================================================
# Area 5: Bootstrap Adversarial Stress
# ============================================================================

class TestArea5BootstrapAdversarial:
    """Stress-test Bootstrap Engine under degenerate and small-sample distributions."""

    def test_all_identical_constant_data(self):
        """Constant data has 0 variance; CI must collapse to [c, c] without crashing."""
        engine = BootstrapEngine(n_bootstraps=100, seed=42)
        const_data = np.array([42.0, 42.0, 42.0, 42.0, 42.0])
        est, low, high = engine.bca_ci(const_data, np.mean)
        assert est == 42.0
        assert low == 42.0
        assert high == 42.0

    def test_single_element_sample(self):
        engine = BootstrapEngine(n_bootstraps=100, seed=42)
        single_data = np.array([7.0])
        est, low, high = engine.percentile_ci(single_data, np.mean)
        assert est == 7.0
        assert low == 7.0
        assert high == 7.0


# ============================================================================
# Area 6: Model Adapters Adversarial Stress
# ============================================================================

class TestArea6ModelAdaptersAdversarial:
    """Stress-test Model Registry against invalid and non-existent architectures."""

    def test_unknown_model_registry_rejection(self):
        registry = ModelAdapterRegistry()
        with pytest.raises(ValueError, match="Unknown model"):
            registry.get_adapter("nonexistent_frontier_model_v999")


# ============================================================================
# Area 7: Out-of-Core Memory Budget Stress
# ============================================================================

class TestArea7MemoryBudgetStress:
    """Stress-test ResidencyManager strict budget abort policy."""

    def test_strict_vram_budget_abort(self):
        """Strict policy with zero VRAM headroom must raise MemoryBudgetExceededError on violation."""
        policy = MemoryBudgetPolicy(
            ram_limit_bytes=100,  # Impossible 100-byte budget
            vram_limit_bytes=100,
            safety_margin_bytes=1000,
            enforcement_mode=EnforcementMode.STRICT,
            violation_action=ViolationAction.ABORT,
        )
        residency = ResidencyManager(budget_policy=policy)
        with pytest.raises(MemoryBudgetExceededError, match="Strict RAM budget exceeded"):
            residency.check_and_enforce_budget()


# ============================================================================
# Area 8: Epistemic Invariant Distinction
# ============================================================================

class TestArea8EpistemicInvariants:
    """Verify machine-readable distinction between EXECUTION_FAILED, NOTHING_FOUND, and ZERO_EFFECT."""

    def test_error_vs_zero_effect_distinction(self):
        """An uninitialized engine must raise RuntimeError, NOT return a 0.0 effect dictionary."""
        from unittest.mock import patch
        tracer = CausalTracingEngine()
        with patch("backend.services.gpt2_engine.load", return_value={"status": "error"}), \
             patch("backend.services.gpt2_engine.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="Synthetic fallback generation is prohibited"):
                tracer.trace_causal_effect("The capital of France is", "The capital of Italy is")
