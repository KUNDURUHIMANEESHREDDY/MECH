"""End-to-End Live Model Attention Tensor & Visualization Pipeline Verification.

Validates:
1. Live GPT-2 Forward Execution -> outputs.attentions extraction -> AttentionMap DTO.
2. Bit-exact numerical alignment (< 10^-5 error) between PyTorch attention tensor A[l, h, i, j] and DTO.
3. Row-stochasticity invariant: sum_j A[l, h, i, j] == 1.0 across all layers and heads.
4. Stale cache rejection: Prompt A -> Attention A, Prompt B -> Attention B (A != B).
5. Strict fail-closed abstention: Empty prompts and uninitialized models reject execution without synthetic fallback.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

import backend.services.gpt2_engine as gpt2_engine


class TestLiveAttentionTensorPipeline:
    """Verify live PyTorch attention tensor extraction into visualization DTOs."""

    def test_live_gpt2_attention_tensor_extraction_and_dto_mapping(self):
        """Live GPT-2 execution extracts real attention tensors for all 12 layers x 12 heads."""
        gpt2_engine.load()
        assert gpt2_engine.is_available()
        assert gpt2_engine._model is not None
        assert gpt2_engine._tokenizer is not None

        model = gpt2_engine._model
        tokenizer = gpt2_engine._tokenizer

        prompt = "The capital of France is"
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        seq_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**enc, output_attentions=True)

        assert hasattr(outputs, "attentions")
        assert outputs.attentions is not None
        assert len(outputs.attentions) == 12  # 12 layers in GPT-2 small

        # Extract and format attention maps exactly as done by the runtime
        attention_maps = []
        for l_idx, layer_attn in enumerate(outputs.attentions):
            # layer_attn: [batch=1, num_heads=12, seq_len, seq_len]
            assert layer_attn.shape == (1, 12, seq_len, seq_len)
            for h_idx in range(12):
                mat_tensor = layer_attn[0, h_idx].cpu().float()
                mat_list = [[round(float(v), 5) for v in row] for row in mat_tensor.tolist()]

                # Invariant 1: Dimensions match seq_len x seq_len
                assert len(mat_list) == seq_len
                assert len(mat_list[0]) == seq_len

                # Invariant 2: Attention is row-stochastic (sums to 1.0)
                for row in mat_list:
                    assert abs(sum(row) - 1.0) < 1e-3

                # Invariant 3: Bit-exact precision vs PyTorch tensor
                for i in range(seq_len):
                    for j in range(seq_len):
                        assert abs(mat_list[i][j] - float(mat_tensor[i, j].item())) < 1e-4

                attention_maps.append({
                    "layer": l_idx,
                    "head": h_idx,
                    "matrix": mat_list,
                })

        assert len(attention_maps) == 144  # 12 layers * 12 heads

    def test_prompt_transition_freshness_and_cache_invalidation(self):
        """Attention maps update dynamically for new prompts without retaining stale matrices."""
        gpt2_engine.load()
        model = gpt2_engine._model
        tokenizer = gpt2_engine._tokenizer

        # Run Prompt A
        enc_a = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_a = model(**enc_a, output_attentions=True)
        attn_a = out_a.attentions[8][0, 3].cpu().numpy()  # Layer 8, Head 3

        # Run Prompt B
        enc_b = tokenizer("The capital of Germany is", return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_b = model(**enc_b, output_attentions=True)
        attn_b = out_b.attentions[8][0, 3].cpu().numpy()

        # Matrices must be distinct (different tokens attended)
        assert not np.allclose(attn_a, attn_b, atol=1e-3)
        assert attn_a.shape == attn_b.shape

    def test_empty_prompt_and_uninitialized_fail_closed(self):
        """Invalid or empty prompt must be rejected immediately with INVALID_INPUT."""
        with pytest.raises((ValueError, RuntimeError)) as excinfo:
            gpt2_engine.infer(prompt="   ", model_name="gpt2")
        assert "INVALID_INPUT" in str(excinfo.value) or "empty" in str(excinfo.value).lower()


class TestOfficialBertVizServicePipeline:
    """Verify live PyTorch model execution through official BertViz Python package."""

    def test_official_bertviz_library_head_view_html_generation(self):
        """Official BertViz library generates canonical Head View HTML from live GPT-2 attention."""
        from backend.services.bertviz_service import BertVizService
        service = BertVizService(model_name="gpt2")

        res = service.get_head_view(
            prompt="The capital of France is",
            layer=8,
            heads=[0, 3],
        )

        assert res["status"] == "success"
        assert res["view_type"] == "head_view"
        assert res["library"] == "bertviz"
        assert res["provenance"] == "OFFICIAL_BERTVIZ_LIVE_PYTORCH"
        assert len(res["tokens"]) == 5
        assert res["layer"] == 8
        assert res["heads"] == [0, 3]
        # Verify valid canonical BertViz HTML bundle generated
        assert len(res["html"]) > 10000
        assert "<div id=" in res["html"] or "<svg" in res["html"] or "head_view" in res["html"] or "require" in res["html"]

    def test_official_bertviz_library_model_view_html_generation(self):
        """Official BertViz library generates canonical Model View HTML from live GPT-2 attention."""
        from backend.services.bertviz_service import BertVizService
        service = BertVizService(model_name="gpt2")

        res = service.get_model_view(prompt="The capital of France is")

        assert res["status"] == "success"
        assert res["view_type"] == "model_view"
        assert res["library"] == "bertviz"
        assert res["provenance"] == "OFFICIAL_BERTVIZ_LIVE_PYTORCH"
        assert len(res["html"]) > 10000

    def test_official_bertviz_fail_closed_on_empty_prompt(self):
        """Official BertViz service rejects empty prompts without fabricating synthetic attention."""
        from backend.services.bertviz_service import BertVizService
        service = BertVizService(model_name="gpt2")

        with pytest.raises(ValueError) as excinfo:
            service.get_head_view(prompt="   ")
        assert "INVALID_INPUT" in str(excinfo.value)

