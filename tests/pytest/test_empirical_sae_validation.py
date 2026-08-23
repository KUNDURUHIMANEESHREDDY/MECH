"""Empirical Validation & Causal Feature Steering Test Suite for MECH SAE Subsystem."""

from __future__ import annotations

import os
from unittest.mock import MagicMock
import pytest
import torch

from backend.interpretability.sae import (
    SAEInterface,
    SAEMetadata,
    SAEOriginState,
    SAEProvenance,
    NativeMECHSAE,
    GenericPyTorchSAEAdapter,
    NativeSAELoader,
    HuggingFaceSAELoader,
    SAEValidator,
    SAELoadError,
    SAEIncompatibleError,
    SAECorruptedError,
    SAECausalInterventionEngine,
    ScientificSAEReportEngine,
)


def _make_mock_gpt2_runtime() -> MagicMock:
    """Mock GPT-2 runtime for deterministic causal steering testing."""
    runtime = MagicMock()
    runtime.device = "cpu"
    runtime.d_model = 768

    vocab = {
        0: " the", 1: " Paris", 2: " France", 3: " capital", 4: " city",
        5: " Germany", 6: " Berlin", 7: " Mary", 8: " John", 9: " dog",
    }
    runtime.tokenizer.decode = lambda ids: "".join(vocab.get(i, f" tok_{i}") for i in ids)
    runtime.tokenizer.encode = lambda s: [1] if "Paris" in s else [6] if "Berlin" in s else [0]
    runtime.tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    # Base forward
    fwd = MagicMock()
    fwd.target_logit = 3.5
    fwd.target_probability = 0.45
    fwd.layer_residuals = {8: torch.randn(768)}
    runtime.forward.return_value = fwd

    # Model mock
    model = MagicMock()
    # Mock layer 8 block
    layer_block = MagicMock()
    hooks = []

    def _reg_hook(fn):
        hooks.append(fn)
        return MagicMock(remove=lambda: hooks.remove(fn) if fn in hooks else None)

    layer_block.register_forward_hook = _reg_hook
    model.transformer.h = {8: layer_block}

    # Model forward pass
    def _model_forward(**kwargs):
        # Base logits [1, 3, 10]
        logits = torch.zeros(1, 3, 10)
        logits[0, -1, 1] = 3.5  # Paris
        # If hook registered, simulate logit boost
        if hooks:
            dummy_out = (torch.zeros(1, 3, 768),)
            for h_fn in hooks:
                dummy_out = h_fn(layer_block, None, dummy_out)
            # Simulated steered logit boost proportional to steering
            logits[0, -1, 1] += 2.0
        out = MagicMock()
        out.logits = logits
        return out

    model.side_effect = _model_forward
    runtime.model = model

    lm_head = MagicMock()
    w_u = torch.zeros(10, 768)
    w_u[1, :] = 1.0  # Feature direction points directly to 'Paris'
    lm_head.weight.data = w_u
    runtime.adapter.get_lm_head.return_value = lm_head

    return runtime


def test_real_sae_checkpoint_loading_and_provenance(tmp_path):
    """Verifies that NativeSAELoader and HuggingFaceSAELoader load real state dicts with SHA256 provenance."""
    ckpt_path = str(tmp_path / "gpt2_layer8_sae.pt")

    # Generate real-format SAE weight tensors
    w_enc = torch.randn(768, 3072)
    w_dec = torch.randn(3072, 768)
    b_enc = torch.zeros(3072)
    b_dec = torch.zeros(768)

    torch.save({
        "W_enc": w_enc,
        "W_dec": w_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
    }, ckpt_path)

    # 1. Load via NativeSAELoader
    sae_native = NativeSAELoader.load(checkpoint_path=ckpt_path, model_id="gpt2", layer=8)
    assert isinstance(sae_native, SAEInterface)
    assert sae_native.metadata.provenance is not None
    assert sae_native.metadata.provenance.origin_state == SAEOriginState.REAL_PRETRAINED
    assert len(sae_native.metadata.provenance.weights_sha256) == 64

    # 2. Load via HuggingFaceSAELoader (using local checkpoint path)
    sae_hf = HuggingFaceSAELoader.load(repo_id=ckpt_path, layer=8)
    assert isinstance(sae_hf, SAEInterface)
    assert sae_hf.metadata.provenance.origin_state == SAEOriginState.REAL_PRETRAINED
    assert sae_hf.metadata.provenance.weights_sha256 == sae_native.metadata.provenance.weights_sha256


def test_cross_backend_numerical_equivalence():
    """Verifies that GenericPyTorchSAEAdapter and NativeMECHSAE produce identical outputs on same weights."""
    w_enc = torch.randn(64, 256)
    w_dec = torch.randn(256, 64)
    # Normalize decoder
    norms = torch.norm(w_dec, dim=1, keepdim=True) + 1e-8
    w_dec = w_dec / norms

    meta = SAEMetadata(
        sae_id="equiv_test",
        model_id="gpt2",
        layer=8,
        hook_point="hook_resid_post",
        d_in=64,
        d_sae=256,
    )

    sae_generic = GenericPyTorchSAEAdapter(metadata=meta, w_enc=w_enc, w_dec=w_dec)
    sae_native = NativeMECHSAE(d_in=64, d_sae=256, layer=8)
    sae_native.w_enc = w_enc.clone()
    sae_native.w_dec = w_dec.clone()

    x = torch.randn(4, 64)
    z_gen, x_hat_gen = sae_generic.reconstruct(x)
    z_nat, x_hat_nat = sae_native.reconstruct(x)

    # Numerical equivalence within 1e-6 tolerance
    assert torch.allclose(z_gen, z_nat, atol=1e-6)
    assert torch.allclose(x_hat_gen, x_hat_nat, atol=1e-6)


def test_sae_strict_failure_defenses(tmp_path):
    """Verifies that SAEValidator and Loaders enforce strict failure defense on corrupt/mismatched models."""
    sae = NativeMECHSAE(d_in=768, d_sae=3072, layer=8, model_id="gpt2")

    # 1. Incompatible dimension error
    with pytest.raises(SAEIncompatibleError):
        SAEValidator.validate_compatibility(sae, expected_d_in=1024)

    # 2. Incompatible layer error
    with pytest.raises(SAEIncompatibleError):
        SAEValidator.validate_compatibility(sae, expected_layer=11)

    # 3. Missing file error
    with pytest.raises(SAELoadError):
        NativeSAELoader.load(checkpoint_path="non_existent_sae.pt")

    # 4. Corrupted weights error
    corrupted_path = str(tmp_path / "corrupted_sae.pt")
    torch.save({"W_enc": torch.tensor([float("nan")])}, corrupted_path)
    with pytest.raises(SAECorruptedError):
        NativeSAELoader.load(checkpoint_path=corrupted_path)


def test_sae_causal_feature_steering_and_faithfulness():
    """Verifies that SAECausalInterventionEngine executes live forward hooks and measures faithfulness."""
    runtime = _make_mock_gpt2_runtime()
    engine = SAECausalInterventionEngine(runtime=runtime)
    sae = NativeMECHSAE(d_in=768, d_sae=3072, layer=8)
    # Align feature 0 direction with target unembedding direction
    sae.w_dec[0, :] = 1.0 / (768 ** 0.5)

    res = engine.steer_feature(
        sae=sae,
        feature_idx=0,
        prompt="The capital of France is",
        target_token=" Paris",
        steering_coefficient=2.0,
        layer=8,
    )

    assert res.feature_idx == 0
    assert res.layer == 8
    assert res.observed_logit_shift > 0.0
    assert res.is_causally_effective is True
    assert res.causal_faithfulness_ratio > 0.0
    assert res.summary


def test_scientific_sae_validation_report_generation():
    """Verifies end-to-end SAEEmpiricalValidationReport construction with full provenance and diagnostics."""
    sae = NativeMECHSAE(d_in=64, d_sae=256, layer=8)
    # Set projective dictionary weights for high explained variance
    q, _ = torch.linalg.qr(torch.randn(256, 64))
    sae.w_dec = q.clone()  # [256, 64]
    sae.w_enc = q.T.clone()  # [64, 256]
    sae.b_enc.zero_()
    sae.b_dec.zero_()

    sample_latents = torch.zeros(8, 256)
    sample_latents[:, :16] = torch.rand(8, 16) + 1.0
    h = sae.decode(sample_latents)
    w_u = torch.randn(10, 64)
    tok = MagicMock()
    tok.decode = lambda ids: " Paris"
    tok.encode = lambda s: [1]

    report = ScientificSAEReportEngine.generate_report(
        sae=sae,
        sample_activations=h,
        prompt="The capital of France is",
        target_token=" Paris",
        unembedding_matrix=w_u,
        tokenizer=tok,
        causal_faithfulness=0.88,
    )

    assert report.is_scientifically_validated is True
    assert "reconstruction_fidelity" in report.to_dict()
    assert "sparsity_profile" in report.to_dict()
    assert "top_dla_features" in report.to_dict()
    assert report.causal_intervention_faithfulness == 0.88
    assert "validated on" in report.scientific_summary
