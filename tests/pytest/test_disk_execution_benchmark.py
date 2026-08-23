"""Test suite for DiskWeightStore and Large-Model Progressive Disk-Execution Engine."""

from __future__ import annotations

import os
import shutil
import tempfile
import pytest
import torch

from backend.runtime.artifacts.cas_store import ArtifactStore
from backend.runtime.memory.disk_weight_store import DiskWeightStore
from backend.runtime.memory.layer_pager import LayerPager
from backend.runtime.memory.benchmark import MockTokenizer
from backend.runtime.model_manager import load_model, get_model_and_tokenizer


@pytest.fixture
def temp_dirs():
    """Provides temporary directories for weights and CAS artifacts."""
    w_dir = tempfile.mkdtemp(prefix="mech_weights_test_")
    a_dir = tempfile.mkdtemp(prefix="mech_artifacts_test_")
    db_path = os.path.join(a_dir, "test_artifacts.db")

    w_store = DiskWeightStore(base_dir=w_dir)
    a_store = ArtifactStore(storage_dir=os.path.join(a_dir, "cas"), db_path=db_path)

    yield w_store, a_store

    shutil.rmtree(w_dir, ignore_errors=True)
    shutil.rmtree(a_dir, ignore_errors=True)


def test_disk_weight_store_sharding_and_loading(temp_dirs):
    """Verifies that layer weights can be sharded and loaded from disk."""
    w_store, _ = temp_dirs
    model_id = "test-shard-model"

    state_dict = {
        "weight_a": torch.randn(64, 64),
        "bias_b": torch.zeros(64),
    }

    meta = w_store.save_layer_weights(model_id, layer_idx=0, state_dict=state_dict, component="block")
    assert meta.num_parameters == (64 * 64 + 64)
    assert meta.size_bytes > 0
    assert os.path.exists(meta.file_path)

    loaded = w_store.load_layer_weights(model_id, layer_idx=0, component="block")
    assert "weight_a" in loaded
    assert "bias_b" in loaded
    assert torch.allclose(loaded["weight_a"], state_dict["weight_a"])


def test_progressive_disk_execution_phases(temp_dirs):
    """Verifies: Cold run (0% cache hit) -> Warm run (100% cache hit) -> Intervention (partial recompute)."""
    w_store, a_store = temp_dirs
    model_id = "test-3-phase-model"
    num_layers = 8
    hidden_size = 128
    num_heads = 4

    manifest = w_store.create_synthetic_scaled_sharded_model(
        model_id=model_id,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        vocab_size=1000,
        dtype=torch.float32,
    )

    pager = LayerPager(device="cpu", dtype=torch.float32, store=a_store, weight_store=w_store)
    tokenizer = MockTokenizer(vocab_size=1000)
    prompt = "Mechanistic interpretability test prompt"

    # 1. Cold Run
    cold_res = pager.run_disk_paged_forward(
        manifest=manifest,
        prompt=prompt,
        tokenizer=tokenizer,
        session_id="s1",
    )
    assert cold_res.layers_executed == num_layers
    assert cold_res.layers_cached == 0
    assert cold_res.cache_hit_rate == 0.0
    assert cold_res.total_disk_bytes_read > 0

    # 2. Warm Run
    warm_res = pager.run_disk_paged_forward(
        manifest=manifest,
        prompt=prompt,
        tokenizer=tokenizer,
        session_id="s2",
    )
    assert warm_res.layers_cached == num_layers
    assert warm_res.layers_executed == 0
    assert warm_res.cache_hit_rate == 1.0
    assert warm_res.total_disk_bytes_read == 0  # 0 disk reads across entire model!
    assert warm_res.layer_disk_bytes_read == 0
    assert warm_res.execution_time_seconds <= cold_res.execution_time_seconds
    assert torch.allclose(cold_res.logits, warm_res.logits, atol=1e-4)

    # 3. Intervention Run at Layer 4
    intervention_layer = 4
    interv_res = pager.run_disk_paged_forward(
        manifest=manifest,
        prompt=prompt,
        tokenizer=tokenizer,
        interventions={intervention_layer: lambda t: t + 10.0},
        session_id="s3",
    )
    # Layers 0..3 hit CAS cache (4 layers cached)
    assert interv_res.layers_cached == 4
    # Layers 4..7 recompute (4 layers executed)
    assert interv_res.layers_executed == 4
    assert interv_res.cache_hit_rate == 4 / num_layers
    # Trace validation
    for i in range(4):
        assert interv_res.traces[i].status == "CACHE_HIT"
    assert interv_res.traces[4].status == "INTERVENTION"
    for i in range(5, 8):
        assert interv_res.traces[i].status == "COMPUTE"


def test_real_hf_model_sharded_disk_execution(temp_dirs):
    """Verifies that a real HuggingFace model can be sharded to disk and executed layer-by-layer."""
    w_store, a_store = temp_dirs
    load_model("gpt2")
    model, tokenizer = get_model_and_tokenizer()

    manifest = w_store.shard_existing_hf_model(model, "gpt2-sharded")
    assert manifest.num_layers == 12
    assert len(manifest.layers) == 12

    pager = LayerPager(device="cpu", store=a_store, weight_store=w_store)
    prompt = "The Eiffel Tower is in"

    # Run disk-paged forward
    out = pager.run_disk_paged_forward(
        manifest=manifest,
        prompt=prompt,
        tokenizer=tokenizer,
        session_id="hf_sharded_test",
    )
    assert out.layers_executed == 12
    assert len(out.tokens) > 0
    assert out.logits is not None
