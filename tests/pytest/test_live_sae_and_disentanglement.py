"""Tests for Live Sparse Autoencoder (SAE) Engine & Polysemantic Disentanglement."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
import torch

from backend.interpretability.sae.live_sae_engine import (
    LiveSparseAutoencoder,
    LiveSAEEngine,
    SAEFeatureActivation,
    SAEDecompositionResult,
    DisentanglementComparison,
)
from backend.discovery.sae_features import SAEFeatureDiscovery


def _make_mock_sae_runtime() -> MagicMock:
    """Mock runtime for testing SAE feature extraction without downloading weights."""
    runtime = MagicMock()
    runtime.device = "cpu"
    runtime.d_model = 768

    # Mock tokenizer
    def _decode(tokens):
        vocab_map = {
            0: " the", 1: " Paris", 2: " France", 3: " capital", 4: " city",
            5: " Germany", 6: " Berlin", 7: " Mary", 8: " John", 9: " dog",
        }
        return "".join(vocab_map.get(t, f" tok_{t}") for t in tokens)

    runtime.tokenizer.decode = _decode
    runtime.tokenizer.encode = lambda s: [1, 2]
    runtime.tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    # Mock forward output with layer residuals
    fwd = MagicMock()
    fwd.layer_residuals = {
        i: torch.randn(768) for i in range(13)
    }
    runtime.forward.return_value = fwd

    # Mock lm_head unembedding weights [50257, 768]
    lm_head = MagicMock()
    lm_head.weight.data = torch.randn(50, 768)
    runtime.adapter.get_lm_head.return_value = lm_head

    # Mock model hook for neuron activation
    runtime.model.transformer.h = {
        8: MagicMock()
    }
    mlp = MagicMock()
    mlp.c_fc.register_forward_hook = lambda h: MagicMock(remove=lambda: None)
    runtime.model.transformer.h[8].mlp = mlp

    return runtime


def test_live_sparse_autoencoder_shape_and_sparsity():
    """Verifies that LiveSparseAutoencoder correctly encodes and decodes hidden state vectors."""
    sae = LiveSparseAutoencoder(d_model=768, d_sae=3072, layer=8, seed=42)
    x = torch.randn(1, 768)

    # Encode with Top-K=16
    z = sae.encode(x, top_k=16)
    assert z.shape == (1, 3072)
    active_count = (z > 0).sum().item()
    assert active_count == 16, f"Expected 16 active features, got {active_count}"

    # Decode back to d_model
    x_rec = sae.decode(z)
    assert x_rec.shape == (1, 768)


def test_live_sae_decompose_prompt_activations():
    """Verifies that LiveSAEEngine decomposes live residual states and computes Direct Logit Attribution."""
    runtime = _make_mock_sae_runtime()
    engine = LiveSAEEngine(runtime=runtime, d_sae_factor=4)

    result = engine.decompose_prompt_activations(
        prompt="The capital of France is",
        layer=8,
        top_k_features=10,
        top_k_tokens=3,
    )

    assert isinstance(result, SAEDecompositionResult)
    assert result.layer == 8
    assert result.prompt == "The capital of France is"
    assert result.l0_norm <= 10
    assert len(result.active_features) > 0

    feat = result.active_features[0]
    assert isinstance(feat, SAEFeatureActivation)
    assert feat.activation > 0
    assert len(feat.top_positive_tokens) == 3
    assert len(feat.top_negative_tokens) == 3
    assert feat.monosemantic_label.startswith("SAE_L8_F")


def test_live_sae_polysemantic_disentanglement_measurement():
    """Verifies that LiveSAEEngine measures specificity gain from raw neurons to SAE features."""
    runtime = _make_mock_sae_runtime()
    engine = LiveSAEEngine(runtime=runtime)

    comp = engine.measure_polysemantic_disentanglement(
        target_prompt="The capital of France is",
        distractor_prompts=[
            "Two plus two equals",
            "When Mary gave the book to John, John thanked",
        ],
        layer=8,
        target_neuron_idx=412,
    )

    assert isinstance(comp, DisentanglementComparison)
    assert 0.0 <= comp.raw_neuron_specificity <= 1.0
    assert 0.0 <= comp.sae_feature_specificity <= 1.0
    assert comp.summary


def test_sae_feature_discovery_integration():
    """Verifies that SAEFeatureDiscovery runs on live runtime activations."""
    runtime = _make_mock_sae_runtime()
    discovery = SAEFeatureDiscovery(experiment_id="exp_sae_001", runtime=runtime)

    res = discovery.analyze_latents(prompt="The capital of France is", layer=8)

    assert "feature_id" in res
    assert "interpretation" in res
    assert "top_positive_tokens" in res
    assert "discovery_confidence" in res
    assert res["discovery_confidence"] > 0.0
