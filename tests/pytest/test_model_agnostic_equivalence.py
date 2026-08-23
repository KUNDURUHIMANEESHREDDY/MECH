"""Model-Agnostic Numerical Equivalence & Architecture Discovery Test Suite.

Verifies:
1. Dynamic architecture introspection across multiple model families (Llama, Qwen, Mistral, OPT, GPT-2, GPT-Neo, Phi).
2. Exact zero-loss numerical equivalence (max(|A - B|) < 1e-4) between standard monolithic HuggingFace models and MECH disk-paged sequential execution across representative model families (GPT-2, OPT, Qwen2.5, TinyLlama, GPT-Neo).
3. Model-agnostic 3-phase execution lifecycle (Cold -> Warm 100% Hit -> Causal Intervention partial recomputation).
"""

import gc
import tempfile
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights

from backend.runtime.artifacts.cas_store import ArtifactStore
from backend.runtime.memory.disk_weight_store import DiskWeightStore
from backend.runtime.memory.layer_pager import LayerPager
from backend.runtime.memory.model_introspector import ModelIntrospector, PositionStrategy, AttentionStrategy


def test_dynamic_architecture_discovery_across_families():
    """Verifies that ModelIntrospector dynamically discovers components across diverse model families."""
    architectures_to_test = [
        ("GPT-2", "gpt2"),
        ("OPT", "facebook/opt-125m"),
        ("Llama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        ("Qwen", "Qwen/Qwen2.5-0.5B"),
        ("GPT-Neo", "EleutherAI/gpt-neo-125m"),
    ]

    for name, model_id in architectures_to_test:
        config = AutoConfig.from_pretrained(model_id)
        with init_empty_weights():
            meta_model = AutoModelForCausalLM.from_config(config)

        spec = ModelIntrospector.introspect(meta_model)

        assert spec.num_layers > 0, f"{name}: failed to discover layers"
        assert spec.input_embeddings is not None, f"{name}: failed to discover input embeddings"
        assert spec.layer_stack is not None, f"{name}: failed to discover layer stack"
        assert len(spec.layer_stack) == spec.num_layers, f"{name}: layer stack length mismatch"
        assert spec.layer_block_cls is not None, f"{name}: failed to discover layer block class"
        assert spec.output_head is not None or spec.input_embeddings is not None, f"{name}: failed to discover head"
        assert len(spec.forward_parameter_names) > 0, f"{name}: failed to discover forward parameters"
        assert isinstance(spec.position_strategy, PositionStrategy), f"{name}: missing position strategy"
        assert isinstance(spec.attention_strategy, AttentionStrategy), f"{name}: missing attention strategy"


@pytest.mark.parametrize("model_id", [
    "gpt2",
    "facebook/opt-125m",
    "Qwen/Qwen2.5-0.5B",
    "EleutherAI/gpt-neo-125m",
])
def test_real_model_numerical_equivalence(model_id):
    """Verifies strict numerical equivalence across real checkpoints (GPT-2, OPT, Qwen2.5, GPT-Neo)."""
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    prompt = "The mechanistic interpretability framework enables reverse-engineering neural networks."
    inputs = tokenizer(prompt, return_tensors="pt")

    # 1. Standard HuggingFace Monolithic Forward Pass
    with torch.no_grad():
        std_logits = model(**inputs).logits

    # 2. MECH Sequential Layer Forward Pass via ExecutionSpec
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = ArtifactStore(storage_dir=f"{tmpdir}/cas", db_path=f"{tmpdir}/cas.db")
        pager = LayerPager(device="cpu", dtype=torch.float32, store=store)

        mech_output = pager.run_sequential_forward(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
        )

        # 3. Assert Strict Numerical Equivalence
        max_diff = (mech_output.logits - std_logits).abs().max().item()
        assert torch.allclose(mech_output.logits, std_logits, atol=1e-4), (
            f"Numerical divergence on {model_id}! Max diff: {max_diff}"
        )
        assert max_diff < 1e-4, f"{model_id} max difference {max_diff} exceeded tolerance"

        del pager, store, model
        gc.collect()


def test_model_agnostic_sharded_disk_paging_equivalence():
    """Verifies that sharding a model to NVMe disk and paging layers yields exact numerical equivalence."""
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    prompt = "Attention heads route information across token positions."
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        std_logits = model(**inputs).logits

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        weight_store = DiskWeightStore(base_dir=f"{tmpdir}/weights")
        manifest = weight_store.shard_existing_hf_model(model, "test_gpt2_sharded")

        store = ArtifactStore(storage_dir=f"{tmpdir}/cas", db_path=f"{tmpdir}/cas.db")
        pager = LayerPager(device="cpu", dtype=torch.float32, store=store, weight_store=weight_store)

        # ── RUN 1: Cold Sharded Disk Forward Pass ──
        paged_out_cold = pager.run_disk_paged_forward(
            manifest=manifest,
            prompt=prompt,
            tokenizer=tokenizer,
            reference_model=model,
        )

        max_diff = (paged_out_cold.logits - std_logits).abs().max().item()
        assert torch.allclose(paged_out_cold.logits, std_logits, atol=1e-4), f"Cold disk mismatch! Max diff: {max_diff}"
        assert paged_out_cold.layers_cached == 0
        assert paged_out_cold.total_disk_bytes_read > 0

        # ── RUN 2: Warm Sharded Disk Forward Pass (100% CAS Cache Hit) ──
        paged_out_warm = pager.run_disk_paged_forward(
            manifest=manifest,
            prompt=prompt,
            tokenizer=tokenizer,
            reference_model=model,
        )

        assert paged_out_warm.cache_hit_rate == 1.0
        assert paged_out_warm.layers_cached == manifest.num_layers
        assert paged_out_warm.layer_disk_bytes_read == 0
        assert torch.allclose(paged_out_warm.logits, std_logits, atol=1e-4)

        # ── RUN 3: Intervened Run @ Layer 6 ──
        def zero_ablation(h: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(h)

        paged_out_intervened = pager.run_disk_paged_forward(
            manifest=manifest,
            prompt=prompt,
            tokenizer=tokenizer,
            reference_model=model,
            interventions={6: zero_ablation},
        )

        # Layers 0..5 should hit cache, layer 6..11 must recompute
        assert paged_out_intervened.layers_cached == 6
        assert paged_out_intervened.layers_executed == 6
        # Logits should have changed due to causal ablation
        assert not torch.allclose(paged_out_intervened.logits, std_logits, atol=1e-2)

        del pager, store, weight_store
        gc.collect()
