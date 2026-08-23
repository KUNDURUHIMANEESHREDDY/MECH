"""Established Interpretability Libraries Integration Test Suite.

Validates:
1. TransformerLens: HookedTransformer execution, canonical head & MLP activation patching.
2. SAELens: StandardSAE encoding/decoding, feature direction extraction, L0 & R^2 metrics.
3. CircuitsVis: Attention patterns and attention heads canonical HTML rendering.
4. Fail-Closed & Epistemic Contracts: Rejection of invalid/empty prompts, heuristic caveats.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from backend.interpretability.causal.transformer_lens_patching import TransformerLensPatchingEngine
from backend.interpretability.sae.sae_lens_adapter import SAELensAdapter, HAS_SAELENS
from backend.services.circuitsvis_service import CircuitsVisService


class TestTransformerLensPatchingIntegration:
    """Validate TransformerLens activation patching operations."""


    def test_transformer_lens_head_patching_live(self):
        engine = TransformerLensPatchingEngine(model_name="gpt2-small", device="cpu")
        res = engine.patch_attention_heads(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Germany is",
            target_token=" Paris",
        )

        assert res["status"] == "success"
        assert res["patch_type"] == "ATTENTION_HEADS"
        assert res["provenance"] == "TRANSFORMER_LENS_PATCHING"
        assert len(res["all_head_effects"]) == 144  # 12 layers * 12 heads
        assert len(res["top_heads"]) <= 10
        assert "statistical_caveat" in res
        assert "heuristic" in res["statistical_caveat"].lower()

        # Invariant: Head effects are finite
        for h in res["top_heads"]:
            assert not math.isnan(h["effect"])
            assert 0.0 <= h["heuristic_p_score"] <= 1.0

    def test_transformer_lens_mlp_patching_live(self):
        engine = TransformerLensPatchingEngine(model_name="gpt2-small", device="cpu")
        res = engine.patch_mlp_layers(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Germany is",
            target_token=" Paris",
        )

        assert res["status"] == "success"
        assert res["patch_type"] == "MLP_OUT"
        assert len(res["layer_effects"]) == 12
        for l in res["layer_effects"]:
            assert not math.isnan(l["effect"])

    def test_transformer_lens_fail_closed_on_empty_prompt(self):
        engine = TransformerLensPatchingEngine(model_name="gpt2-small", device="cpu")
        with pytest.raises(ValueError) as excinfo:
            engine.patch_attention_heads(
                clean_prompt="   ",
                corrupted_prompt="The capital of Germany is",
                target_token=" Paris",
            )
        assert "INVALID_INPUT" in str(excinfo.value)


@pytest.mark.skipif(not HAS_SAELENS, reason="sae_lens not installed")
class TestSAELensAdapterIntegration:
    """Validate SAELens StandardSAE adapter execution and metrics."""

    def test_sae_lens_adapter_forward_and_directions(self):
        sae = SAELensAdapter(d_in=768, d_sae=1024, model_name="gpt2-small", layer=8, device="cpu")
        
        # Test feature direction extraction
        d_0 = sae.get_feature_direction(0)
        assert d_0.shape == (768,)
        assert not torch.isnan(d_0).any()

        # Test encoding and decoding
        x = torch.randn(2, 4, 768)
        acts = sae.encode(x)
        recon = sae.decode(acts)

        assert acts.shape == (2, 4, 1024)
        assert recon.shape == (2, 4, 768)

    def test_sae_lens_adapter_metrics(self):
        sae = SAELensAdapter(d_in=768, d_sae=1024, model_name="gpt2-small", layer=8, device="cpu")
        x = torch.randn(4, 8, 768)
        res = sae.forward(x)

        assert res.l0_norm >= 0
        assert res.l1_norm >= 0.0
        assert res.reconstruction_mse >= 0.0
        assert 0.0 <= res.explained_variance <= 1.0

    def test_sae_lens_out_of_bounds_feature_rejection(self):
        sae = SAELensAdapter(d_in=768, d_sae=1024, model_name="gpt2-small", layer=8, device="cpu")
        with pytest.raises(IndexError):
            sae.get_feature_direction(999999)

    def test_sae_lens_provenance_and_pretrained_fail_closed(self):
        """Scratch initialized SAE must be labeled SYNTHETIC_INITIALIZED, and invalid pretrained IDs must fail closed."""
        sae = SAELensAdapter(d_in=768, d_sae=1024, model_name="gpt2-small", layer=8, device="cpu")
        assert sae.metadata.provenance.origin_state.value == "SYNTHETIC_INITIALIZED"

        # Loading non-existent pretrained release must fail closed with RuntimeError
        with pytest.raises(RuntimeError) as excinfo:
            SAELensAdapter.from_pretrained(release="non_existent_release_xyz", sae_id="layer_8")
        assert "EXECUTION_FAILED" in str(excinfo.value)

    def test_legacy_sae_loader_fails_closed_when_weights_missing(self):
        """Legacy loader activate() must raise RuntimeError instead of generating synthetic random weights."""
        from backend.interpretability.sae.loader import SAE, SAEConfig
        cfg = SAEConfig(
            d_in=768,
            d_sae=1024,
            model_name="gpt2-small",
            layer=8,
            repo_id="test/repo",
            version="1.0",
        )
        uninitialized_sae = SAE(cfg, weights=None)
        with pytest.raises(RuntimeError) as excinfo:
            uninitialized_sae.activate(torch.randn(768))
        assert "Synthetic fallback generation is prohibited" in str(excinfo.value)



class TestCircuitsVisServiceIntegration:
    """Validate CircuitsVis attention visualization HTML generation."""

    def test_circuitsvis_attention_patterns_live(self):
        service = CircuitsVisService(model_name="gpt2")
        res = service.get_attention_patterns(prompt="The capital of France is", layer=8)

        assert res["status"] == "success"
        assert res["view_type"] == "circuitsvis_attention_patterns"
        assert res["library"] == "circuitsvis"
        assert res["provenance"] == "OFFICIAL_CIRCUITSVIS_LIVE_PYTORCH"
        assert len(res["tokens"]) == 5
        assert len(res["html"]) > 1000

    def test_circuitsvis_attention_heads_live(self):
        service = CircuitsVisService(model_name="gpt2")
        res = service.get_attention_heads(prompt="The capital of France is")

        assert res["status"] == "success"
        assert res["view_type"] == "circuitsvis_attention_heads"
        assert len(res["html"]) > 1000

    def test_circuitsvis_fail_closed_on_empty_prompt(self):
        service = CircuitsVisService(model_name="gpt2")
        with pytest.raises(ValueError) as excinfo:
            service.get_attention_patterns(prompt="   ")
        assert "INVALID_INPUT" in str(excinfo.value)
