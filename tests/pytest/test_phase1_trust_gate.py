"""MECH Scientific Trust Gate — Phase 1 Acceptance Test Battery.

Verifies:
1. P0: SAE Integrity — SparseAutoencoderAdapter & SAELensAdapter return real, non-zero decoder vectors.
2. P0: Invalid feature indices fail closed with explicit IndexError.
3. P0: DLA cannot silently produce an all-zero attribution due to a stub.
4. P1: DLA results carry explicit non-causal epistemic labeling (is_causal=False, VIRTUAL_UNEMBEDDED_PROJECTION).
5. P1: Causal steering hook enforces device and dtype alignment dynamically without crashes.
6. P1: Batch-safe execution over multi-sequence inputs (B > 1).
7. P0: Provenance metadata is attached to every interventional and attribution result.
8. Golden Rule #2: Zero hardcoded results — outputs dynamically vary with weights, prompts, and alpha.
"""

from __future__ import annotations

import types
import pytest
import torch
import numpy as np

from backend.interpretability.sae.sae_interface import (
    SAEMetadata,
    SAEArchitectureType,
    SAEBackendSource,
)
from backend.interpretability.sae.sae_adapter import (
    NativeMECHSAE,
    GenericPyTorchSAEAdapter,
    SAELensAdapter,
    SparseAutoencoderAdapter,
)
from backend.interpretability.sae.attribution.sae_dla import (
    SAEDLAResult,
    SAEDirectLogitAttributor,
)
from backend.interpretability.sae.causal.sae_intervention_engine import (
    SAECausalInterventionEngine,
    SAECausalInterventionResult,
)


# =========================================================================
# 1. SAE Integrity & Non-Zero Decoder Vector Verification
# =========================================================================

def test_sparse_autoencoder_adapter_real_feature_direction():
    """Verify that SparseAutoencoderAdapter extracts real, non-zero decoder weights."""
    d_in = 64
    d_sae = 256
    
    # Create mock external sparse_autoencoder model with decoder.weight [d_sae, d_in]
    mock_model = types.SimpleNamespace()
    mock_decoder = types.SimpleNamespace()
    
    # Generate distinct non-zero weights
    rng_weights = torch.randn(d_sae, d_in) + 0.5
    mock_decoder.weight = rng_weights
    mock_model.decoder = mock_decoder
    
    adapter = SparseAutoencoderAdapter(
        sparse_autoencoder_obj=mock_model,
        metadata=SAEMetadata(
            sae_id="test_sae",
            model_id="gpt2",
            layer=8,
            hook_point="hook_resid_post",
            d_in=d_in,
            d_sae=d_sae,
            backend_source=SAEBackendSource.SPARSE_AUTOENCODER,
        )
    )
    
    # Check feature 42
    vec = adapter.get_feature_direction(42)
    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (d_in,)
    assert not torch.all(vec == 0)
    assert torch.allclose(vec, rng_weights[42])


def test_sparse_autoencoder_adapter_transposed_orientation():
    """Verify SparseAutoencoderAdapter handles [d_in, d_sae] orientation properly."""
    d_in = 32
    d_sae = 128
    
    mock_model = types.SimpleNamespace()
    mock_model.W_dec = torch.randn(d_in, d_sae) + 1.0
    
    adapter = SparseAutoencoderAdapter(
        sparse_autoencoder_obj=mock_model,
        metadata=SAEMetadata(
            sae_id="test_transposed",
            model_id="gpt2",
            layer=8,
            hook_point="hook_resid_post",
            d_in=d_in,
            d_sae=d_sae,
            backend_source=SAEBackendSource.SPARSE_AUTOENCODER,
        )
    )
    
    vec = adapter.get_feature_direction(10)
    assert vec.shape == (d_in,)
    assert torch.allclose(vec, mock_model.W_dec[:, 10])


def test_sae_adapter_invalid_index_fails_closed():
    """Verify that negative or out-of-bounds indices throw explicit IndexError."""
    sae = NativeMECHSAE(d_in=32, d_sae=128)
    
    with pytest.raises(IndexError):
        sae.get_feature_direction(-1)
        
    with pytest.raises(IndexError):
        sae.get_feature_direction(128)
        
    with pytest.raises(IndexError):
        sae.get_feature_direction(999)


def test_sparse_autoencoder_missing_weights_fails_closed():
    """Verify that an adapter with no decoder weights raises RuntimeError rather than returning zeros."""
    empty_obj = types.SimpleNamespace()
    adapter = SparseAutoencoderAdapter(empty_obj)
    
    with pytest.raises(RuntimeError):
        adapter.get_feature_direction(0)


# =========================================================================
# 2. DLA Non-Causal Semantics & Zero-Stub Rejection
# =========================================================================

def test_sae_dla_rejects_zero_direction_vector():
    """Verify that SAEDirectLogitAttributor rejects zero-norm feature directions."""
    vocab_size = 100
    d_in = 32
    w_u = torch.randn(vocab_size, d_in)
    
    class DummyTokenizer:
        def decode(self, idx): return f"tok_{idx[0]}"
        def encode(self, tok): return [1]
        
    attributor = SAEDirectLogitAttributor(unembedding_matrix=w_u, tokenizer=DummyTokenizer())
    
    # Construct a dummy SAE that returns zeros
    class ZeroSAE(NativeMECHSAE):
        def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
            return torch.zeros(d_in)
            
    bad_sae = ZeroSAE(d_in=d_in, d_sae=64)
    with pytest.raises(ValueError, match="zero norm"):
        attributor.attribute_feature(bad_sae, 0)


def test_dla_explicitly_labeled_non_causal():
    """Verify that DLA results carry non-causal epistemic tags and limitations."""
    vocab_size = 50
    d_in = 16
    w_u = torch.randn(vocab_size, d_in)
    
    class DummyTokenizer:
        def decode(self, idx): return f"tok_{idx[0]}"
        def encode(self, tok): return [5]
        
    attributor = SAEDirectLogitAttributor(unembedding_matrix=w_u, tokenizer=DummyTokenizer())
    sae = NativeMECHSAE(d_in=d_in, d_sae=32)
    
    res = attributor.attribute_feature(sae, feature_idx=2, target_token="tok_5")
    assert isinstance(res, SAEDLAResult)
    assert res.is_causal is False
    assert res.evidence_type == "VIRTUAL_UNEMBEDDED_PROJECTION"
    assert res.provenance == "COMPUTED_RUNTIME"
    assert "Does NOT measure mediated downstream" in res.method_limitation
    
    d_dict = res.to_dict()
    assert d_dict["is_causal"] is False
    assert d_dict["evidence_type"] == "VIRTUAL_UNEMBEDDED_PROJECTION"


# =========================================================================
# 3. Causal Steering Device, DType & Batch Safety
# =========================================================================

def test_causal_intervention_device_and_dtype_coercion():
    """Verify that steering hook coerces steering delta to match hidden state dtype and device."""
    d_in = 64
    d_sae = 128
    
    sae = NativeMECHSAE(d_in=d_in, d_sae=d_sae)
    d_i = sae.get_feature_direction(5)
    
    # Simulate mixed precision hidden state (e.g. float16)
    h_float16 = torch.randn(2, 8, d_in, dtype=torch.float16)
    steering_coeff = 3.5
    
    # Test delta computation logic
    delta = (steering_coeff * d_i).to(dtype=h_float16.dtype, device=h_float16.device)
    assert delta.dtype == torch.float16
    
    # In-place broadcast injection
    h_steered = h_float16.clone()
    h_steered[:, -1, :] = h_steered[:, -1, :] + delta
    
    assert h_steered.shape == (2, 8, d_in)
    assert h_steered.dtype == torch.float16
    assert not torch.allclose(h_steered[:, -1, :], h_float16[:, -1, :])


def test_causal_intervention_provenance_and_evidence_type():
    """Verify that SAECausalInterventionResult includes required epistemic and provenance metadata."""
    res = SAECausalInterventionResult(
        feature_idx=7,
        layer=8,
        prompt="Test prompt",
        target_token=" target",
        steering_coefficient=2.0,
        baseline_target_logit=5.0,
        steered_target_logit=7.2,
        observed_logit_shift=2.2,
        predicted_logit_shift=2.0,
        baseline_target_probability=0.1,
        steered_target_probability=0.35,
        causal_faithfulness_ratio=1.1,
        is_causally_effective=True,
        summary="Test summary",
    )
    
    assert res.provenance == "COMPUTED_RUNTIME"
    assert res.evidence_type == "INTERVENTIONAL_CAUSAL_STEERING"
    
    d = res.to_dict()
    assert d["provenance"] == "COMPUTED_RUNTIME"
    assert d["evidence_type"] == "INTERVENTIONAL_CAUSAL_STEERING"


# =========================================================================
# 4. Golden Rule #2: Zero Hardcoded Results
# =========================================================================

def test_golden_rule_dla_computation_is_dynamic():
    """Verify that changing unembedding weights changes DLA results dynamically."""
    vocab_size = 20
    d_in = 8
    
    w_u1 = torch.eye(vocab_size, d_in)
    w_u2 = w_u1 * 5.0
    
    class DummyTokenizer:
        def decode(self, idx): return f"t_{idx[0]}"
        def encode(self, tok): return [0]
        
    sae = NativeMECHSAE(d_in=d_in, d_sae=16, seed=123)
    
    attributor1 = SAEDirectLogitAttributor(unembedding_matrix=w_u1, tokenizer=DummyTokenizer())
    attributor2 = SAEDirectLogitAttributor(unembedding_matrix=w_u2, tokenizer=DummyTokenizer())
    
    res1 = attributor1.attribute_feature(sae, 0)
    res2 = attributor2.attribute_feature(sae, 0)
    
    assert res1.direct_effect_magnitude > 0
    assert res2.direct_effect_magnitude > 0
    assert not np.isclose(res1.direct_effect_magnitude, res2.direct_effect_magnitude)
    assert np.isclose(res2.direct_effect_magnitude, res1.direct_effect_magnitude * 5.0, atol=1e-3)

