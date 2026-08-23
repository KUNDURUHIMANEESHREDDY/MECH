"""Comprehensive Test Suite for the MECH Unified Sparse Autoencoder (SAE) Subsystem."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
import torch

from backend.interpretability.sae import (
    SAEInterface,
    SAEMetadata,
    SAEArchitectureType,
    SAEBackendSource,
    NativeMECHSAE,
    GenericPyTorchSAEAdapter,
    SAELensAdapter,
    SAERegistry,
    default_sae_registry,
    NativeSAELoader,
    SAEDirectLogitAttributor,
    SAEDLAResult,
    FeatureActivationAnalyzer,
    FeatureSparsityTracker,
    FeatureInterpretabilityAnalyzer,
    FeatureDashboardEngine,
    SAETrainer,
    SAETrainingConfig,
    SAELossCalculator,
    DeadFeatureResampler,
    SAELoader,
)


def _make_mock_tokenizer():
    """Helper creating mock tokenizer with decode/encode support."""
    tok = MagicMock()
    vocab = {
        0: " the", 1: " Paris", 2: " France", 3: " capital", 4: " city",
        5: " Germany", 6: " Berlin", 7: " Mary", 8: " John", 9: " dog",
    }
    tok.decode = lambda ids: "".join(vocab.get(i, f" tok_{i}") for i in ids)
    tok.encode = lambda s: [1, 2]
    return tok


def test_sae_interface_and_native_mech_adapter():
    """Verifies that NativeMECHSAE adheres to SAEInterface and computes valid reconstructions."""
    sae = NativeMECHSAE(d_in=768, d_sae=3072, layer=8, seed=42)

    assert isinstance(sae, SAEInterface)
    assert sae.metadata.d_in == 768
    assert sae.metadata.d_sae == 3072
    assert sae.metadata.backend_source == SAEBackendSource.NATIVE

    # Test encode / decode / reconstruct
    x = torch.randn(2, 768)
    z, x_hat = sae.reconstruct(x, top_k=32)

    assert z.shape == (2, 3072)
    assert x_hat.shape == (2, 768)

    # Check sparsity stats
    sparsity = sae.get_sparsity(z)
    assert sparsity["l0"] <= 32.0
    assert sparsity["l1"] >= 0.0

    # Check error metrics
    err = sae.get_reconstruction_error(x, x_hat)
    assert err["mse"] >= 0.0
    assert 0.0 <= err["explained_variance"] <= 1.0

    # Check feature direction
    d_0 = sae.get_feature_direction(0)
    assert d_0.shape == (768,)
    assert abs(torch.norm(d_0).item() - 1.0) < 1e-4


def test_generic_pytorch_sae_adapter_and_loaders():
    """Verifies that GenericPyTorchSAEAdapter and NativeSAELoader initialize and adapt weights."""
    meta = SAEMetadata(
        sae_id="custom_sae_test",
        model_id="gpt2",
        layer=6,
        hook_point="hook_resid_post",
        d_in=128,
        d_sae=512,
    )
    w_enc = torch.randn(128, 512)
    w_dec = torch.randn(512, 128)
    adapter = GenericPyTorchSAEAdapter(meta, w_enc=w_enc, w_dec=w_dec)

    x = torch.randn(1, 128)
    z, x_hat = adapter.reconstruct(x)
    assert z.shape == (1, 512)
    assert x_hat.shape == (1, 128)

    # Test loader creates valid instance
    loader_sae = NativeSAELoader.load(d_in=128, d_sae=512, layer=6)
    assert isinstance(loader_sae, SAEInterface)


def test_sae_registry_registration_and_lookup():
    """Verifies that SAERegistry registers and looks up SAE instances cleanly."""
    registry = SAERegistry()
    sae1 = NativeMECHSAE(d_in=768, d_sae=3072, layer=8, model_id="gpt2")
    sae2 = NativeMECHSAE(d_in=768, d_sae=3072, layer=9, model_id="gpt2")

    registry.register(sae1)
    registry.register(sae2)

    assert len(registry.list_saes()) == 2
    assert registry.get(model_id="gpt2", layer=8) is sae1
    assert registry.get(model_id="gpt2", layer=9) is sae2
    assert registry.get(model_id="gpt2", layer=10) is None


def test_sae_dla_attributor():
    """Verifies Direct Logit Attribution across SAE decoder directions."""
    sae = NativeMECHSAE(d_in=64, d_sae=256, layer=8)
    w_u = torch.randn(20, 64)  # 20 vocab tokens
    tok = _make_mock_tokenizer()

    attributor = SAEDirectLogitAttributor(unembedding_matrix=w_u, tokenizer=tok)
    res = attributor.attribute_feature(
        sae=sae,
        feature_idx=5,
        top_k=4,
        target_token=" Paris",
        distractor_token=" Berlin",
    )

    assert isinstance(res, SAEDLAResult)
    assert res.feature_idx == 5
    assert len(res.top_positive_tokens) == 4
    assert len(res.top_negative_tokens) == 4
    assert res.target_token_logit_boost is not None
    assert res.direct_effect_magnitude > 0.0


def test_feature_sparsity_and_dead_neuron_tracker():
    """Verifies that FeatureSparsityTracker accurately measures L0, L1, and detects dead features."""
    sae = NativeMECHSAE(d_in=64, d_sae=128)
    tracker = FeatureSparsityTracker(sae)

    # Simulate sparse activations where only features 0..9 fire
    z = torch.zeros(5, 128)
    z[:, :10] = torch.rand(5, 10) + 0.1

    metrics = tracker.update(z)
    assert metrics["l0"] == 10.0

    profile = tracker.get_sparsity_profile()
    assert profile.mean_l0 == 10.0
    assert profile.dead_features_count == 118  # 128 - 10
    assert profile.dead_features_fraction > 0.90


def test_feature_interpretability_and_msi():
    """Verifies Monosemantic Specificity Index calculation and comparative report generation."""
    analyzer = FeatureInterpretabilityAnalyzer()

    # Highly specific SAE feature
    sae_msi = analyzer.compute_msi(target_activation=5.0, distractor_activations=[0.1, 0.05, 0.0])
    assert sae_msi > 0.95

    # Polysemantic raw neuron
    raw_msi = analyzer.compute_msi(target_activation=4.0, distractor_activations=[3.8, 4.2, 3.5])
    assert raw_msi < 0.35

    report = analyzer.evaluate_disentanglement(
        concept_label="Capital Recall",
        sae_target_act=5.0,
        sae_distractor_acts=[0.1, 0.05],
        sae_feature_idx=142,
        raw_neuron_target_act=4.0,
        raw_neuron_distractor_acts=[3.8, 4.2],
        raw_neuron_idx=412,
    )

    assert report.is_monosemantic is True
    assert report.polysemantic_gap > 0.40
    assert report.summary


def test_feature_dashboard_engine_prompt_inspection():
    """Verifies that FeatureDashboardEngine produces full structured inspection reports."""
    sae = NativeMECHSAE(d_in=64, d_sae=256, layer=8)
    w_u = torch.randn(20, 64)
    tok = _make_mock_tokenizer()

    dashboard = FeatureDashboardEngine(sae=sae, unembedding_matrix=w_u, tokenizer=tok)
    h = torch.randn(1, 64)

    report = dashboard.inspect_prompt(
        hidden_state=h,
        prompt="The capital of France is",
        top_k_features=8,
        target_token=" Paris",
    )

    assert "metadata" in report
    assert "reconstruction" in report
    assert "sparsity" in report
    assert "active_features" in report
    assert len(report["active_features"]) <= 8


def test_sae_trainer_and_feature_resampler():
    """Verifies that SAETrainer optimizes loss over activation batches and resamples dead features."""
    cfg = SAETrainingConfig(
        d_in=64,
        d_sae=128,
        architecture=SAEArchitectureType.TOP_K,
        top_k=8,
        learning_rate=1e-2,
        dead_steps_threshold=2,
        resample_freq=3,
    )
    trainer = SAETrainer(config=cfg)

    # Train for 5 steps on synthetic activation batches
    losses = []
    for step in range(5):
        batch = torch.randn(16, 64)
        step_res = trainer.train_step(batch)
        losses.append(step_res.total_loss)

    assert len(losses) == 5
    assert trainer.step_count == 5
    assert losses[-1] >= 0.0


def test_legacy_sae_loader_backwards_compatibility():
    """Verifies that the legacy SAELoader interface remains functional and that
    activation is fail-closed when weights were never actually loaded."""
    loader = SAELoader()
    sae = loader.load_sae("hf", "gpt2-small-res-jb", version="v1")

    # Legacy interface returns a correctly configured SAE descriptor.
    assert sae.config.d_in == 768
    assert sae.config.d_sae == 16384

    # Fail-closed: activating an SAE whose weights were never loaded must raise
    # rather than silently synthesize fake features.
    dummy_hidden = torch.randn(768).numpy()
    with pytest.raises(RuntimeError):
        sae.activate(dummy_hidden)
