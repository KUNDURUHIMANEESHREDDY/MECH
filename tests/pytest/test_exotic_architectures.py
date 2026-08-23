"""Tests for Exotic Transformer Architectures: MoE Paging, Encoder-Decoder Execution, and Introspection.

Verifies:
1. MoE Architecture Introspection & Selective Top-K Expert Paging with Routing Persistence.
2. MoE Expert-Level Causal Interventions and Ablations.
3. Encoder-Decoder Progressive Execution (T5) with Cross-Attention DAG Tracking.
4. GQA/MQA Attention Asymmetry & Parallel Attention-MLP Introspection.
"""

import gc
import tempfile
import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from backend.runtime.artifacts.cas_store import ArtifactStore
from backend.runtime.memory.enc_dec_pager import EncoderDecoderPager
from backend.runtime.memory.moe_pager import MoEExpertPager
from backend.runtime.memory.model_introspector import ModelIntrospector, PositionStrategy


def test_moe_architecture_introspection():
    """Verifies that MoE architectures are dynamically discovered with expert metadata."""
    moe_models = [
        ("Mixtral-8x7B", "mistralai/Mixtral-8x7B-v0.1", 8),
        ("Qwen-MoE", "Qwen/Qwen1.5-MoE-A2.7B", 4),
    ]

    for name, model_id, expected_min_experts in moe_models:
        config = AutoConfig.from_pretrained(model_id)
        with init_empty_weights():
            meta_model = AutoModelForCausalLM.from_config(config)

        spec = ModelIntrospector.introspect(meta_model)

        assert spec.is_moe is True, f"{name}: failed to detect is_moe=True"
        assert spec.num_experts is not None and spec.num_experts >= expected_min_experts, (
            f"{name}: expected at least {expected_min_experts} experts, got {spec.num_experts}"
        )
        assert spec.num_layers > 0
        assert spec.layer_stack is not None
        assert len(spec.layer_stack) == spec.num_layers


def test_moe_selective_expert_paging_and_ablation():
    """Verifies that MoE pager selectively computes only active top-k experts and enables expert ablation."""
    hidden_dim = 64
    num_experts = 8
    top_k = 2
    batch_size = 2
    seq_len = 4

    # Build a minimal synthetic MoE layer block
    class SyntheticMoEBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(hidden_dim, num_experts)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_experts)
            ])

        def forward(self, x):
            return x

    block = SyntheticMoEBlock()
    x = torch.randn(batch_size, seq_len, hidden_dim)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = ArtifactStore(storage_dir=f"{tmpdir}/cas", db_path=f"{tmpdir}/cas.db")
        pager = MoEExpertPager(store=store)

        # ── RUN 1: Standard Selective MoE Forward Pass ──
        out, decision, trace = pager.execute_selective_moe_layer(
            layer_idx=0,
            layer_block=block,
            hidden_states=x,
            top_k=top_k,
        )

        assert out.shape == x.shape
        assert decision.total_experts == 8
        assert decision.top_k == 2
        assert len(decision.active_expert_ids) <= batch_size * seq_len * top_k
        assert trace.memory_savings_ratio >= 1.0

        # Check that routing decision was persisted in CAS
        artifacts = store.search()
        assert len(artifacts) >= 1
        assert any(a["component"] == "moe_routing" for a in artifacts)

        # ── RUN 2: Expert-Level Causal Intervention (Zero Ablation on active expert) ──
        target_expert = decision.active_expert_ids[0]
        def zero_ablation(h: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(h)

        ablated_out, ablated_decision, _ = pager.execute_selective_moe_layer(
            layer_idx=0,
            layer_block=block,
            hidden_states=x,
            top_k=top_k,
            expert_interventions={target_expert: zero_ablation},
        )

        # Output must diverge for tokens routed to the ablated expert
        assert not torch.allclose(out, ablated_out)

        del pager, store
        gc.collect()


def test_encoder_decoder_t5_execution():
    """Verifies that Encoder-Decoder architectures execute through EncoderDecoderPager with cross-attention tracking."""
    model_id = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    encoder_prompt = "translate English to German: The cat sits on the mat."
    decoder_prompt = "Die Katze"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = ArtifactStore(storage_dir=f"{tmpdir}/cas", db_path=f"{tmpdir}/cas.db")
        pager = EncoderDecoderPager(store=store)

        output = pager.run_sequential_forward(
            model=model,
            tokenizer=tokenizer,
            encoder_prompt=encoder_prompt,
            decoder_prompt=decoder_prompt,
        )

        assert output.encoder_hidden_states is not None
        assert output.decoder_hidden_states is not None
        assert output.logits is not None
        assert output.encoder_layers_executed == 6
        assert output.decoder_layers_executed == 6
        assert len(output.encoder_residuals) == 6
        assert len(output.decoder_residuals) == 6

        # Check CAS artifacts recorded for both encoder and decoder
        cas_artifacts = store.search()
        assert len(cas_artifacts) >= 7  # 1 encoder final + 6 decoder layers
        assert any(a["component"] == "encoder_hidden_states" for a in cas_artifacts)
        assert any(a["component"] == "decoder_residual" for a in cas_artifacts)

        del pager, store
        gc.collect()


def test_gqa_head_hierarchy_discovery():
    """Verifies that GQA/MQA attention head asymmetry is dynamically detected."""
    gqa_models = [
        ("TinyLlama GQA", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 32, 4),
        ("Qwen2.5 GQA", "Qwen/Qwen2.5-0.5B", 14, 2),
        ("Mixtral GQA", "mistralai/Mixtral-8x7B-v0.1", 32, 8),
    ]

    for name, model_id, expected_q_heads, expected_kv_heads in gqa_models:
        config = AutoConfig.from_pretrained(model_id)
        with init_empty_weights():
            meta_model = AutoModelForCausalLM.from_config(config)

        spec = ModelIntrospector.introspect(meta_model)

        assert spec.num_heads == expected_q_heads, f"{name}: expected {expected_q_heads} Q heads, got {spec.num_heads}"
        assert spec.num_key_value_heads == expected_kv_heads, (
            f"{name}: expected {expected_kv_heads} KV heads, got {spec.num_key_value_heads}"
        )


def test_parallel_attention_mlp_phi_discovery():
    """Verifies that parallel attention/MLP architectures like Phi-2 are introspected correctly."""
    config = AutoConfig.from_pretrained("microsoft/phi-2")
    with init_empty_weights():
        meta_model = AutoModelForCausalLM.from_config(config)

    spec = ModelIntrospector.introspect(meta_model)

    assert spec.num_layers == 32
    assert spec.hidden_size == 2560
    assert spec.num_heads == 32
    assert spec.position_strategy == PositionStrategy.ROTARY_ROPE
    assert spec.layer_block_cls.__name__ == "PhiDecoderLayer"
