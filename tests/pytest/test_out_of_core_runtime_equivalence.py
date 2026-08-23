"""Equivalence and Numerical Tolerance Test Suite for Phase 12 Out-of-Core Runtime.

Proves that OutOfCoreRuntime produces identical outputs to the reference InMemoryRuntime
within strict scientific numerical tolerances (abs <= 1e-4, exact integer ranks).
"""

import math
import pytest
import torch

from backend.runtime.activation_artifact import ActivationArtifact
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.out_of_core_runtime import OutOfCoreRuntime
from backend.runtime.virtual_unembedding import VirtualUnembeddingEngine
from backend.science.reproducibility.tolerance_engine import DEFAULT_ABS_TOLERANCE


@pytest.fixture(scope="module")
def runtimes():
    """Initializes in-memory and out-of-core runtimes for testing."""
    in_mem = InMemoryRuntime(model_id="gpt2", device="cpu")
    out_core = OutOfCoreRuntime(model_id="gpt2", device="cpu", vram_budget_mb=512.0)
    return in_mem, out_core


def test_virtual_unembedding_exact_topk(runtimes):
    in_mem, _ = runtimes
    engine = VirtualUnembeddingEngine(chunk_size=1024)

    # Random dummy hidden state
    h = torch.randn(768)
    w_u = in_mem.model.lm_head.weight.data

    # Monolithic reference
    full_logits = torch.matmul(h, w_u.T)
    ref_top_vals, ref_top_indices = torch.topk(full_logits, k=10)

    # Virtual chunked projection
    res = engine.project_hidden_state(
        hidden_state=h,
        unembedding_weights=w_u,
        tokenizer=in_mem.tokenizer,
        top_k=10,
    )

    assert len(res["top_candidates"]) == 10
    for i in range(10):
        ref_id = int(ref_top_indices[i].item())
        ref_val = float(ref_top_vals[i].item())
        obs = res["top_candidates"][i]

        assert obs["token_id"] == ref_id
        assert abs(obs["logit"] - ref_val) <= 1e-3


def test_runtime_forward_equivalence(runtimes):
    in_mem, out_core = runtimes
    prompt = "The capital of France is"
    target_token = " Paris"

    res_mem = in_mem.forward(prompt, target_token=target_token)
    res_core = out_core.forward(prompt, target_token=target_token)

    assert res_mem.top_predicted_token == res_core.top_predicted_token
    assert res_mem.top_predicted_id == res_core.top_predicted_id

    # Exact target rank equality
    assert res_mem.target_rank == res_core.target_rank

    # Floating point tolerance
    assert abs(res_mem.target_logit - res_core.target_logit) <= DEFAULT_ABS_TOLERANCE
    assert abs(res_mem.target_probability - res_core.target_probability) <= DEFAULT_ABS_TOLERANCE


def test_logit_lens_trajectory_equivalence(runtimes):
    in_mem, out_core = runtimes
    prompt = "The capital of France is"
    target_token = " Paris"

    traj_mem = in_mem.compute_logit_lens_trajectory(prompt, target_token=target_token)
    traj_core = out_core.compute_logit_lens_trajectory(prompt, target_token=target_token)

    assert len(traj_mem) == len(traj_core)

    for step_mem, step_core in zip(traj_mem, traj_core):
        assert step_mem["layer"] == step_core["layer"]
        assert step_mem["top_token_id"] == step_core["top_token_id"]
        assert step_mem["target_rank"] == step_core["target_rank"]
        assert abs(step_mem["target_logit"] - step_core["target_logit"]) <= DEFAULT_ABS_TOLERANCE


def test_causal_intervention_equivalence(runtimes):
    in_mem, out_core = runtimes
    prompt = "The capital of France is"
    target_token = " Paris"

    # Zero-ablate L8_N412
    res_mem = in_mem.apply_intervention(
        prompt=prompt,
        target_token=target_token,
        layer=8,
        component_type="neuron",
        component_index=412,
        ablation_scale=0.0,
    )

    res_core = out_core.apply_intervention(
        prompt=prompt,
        target_token=target_token,
        layer=8,
        component_type="neuron",
        component_index=412,
        ablation_scale=0.0,
    )

    assert abs(res_mem.clean_logit - res_core.clean_logit) <= DEFAULT_ABS_TOLERANCE
    assert abs(res_mem.intervened_logit - res_core.intervened_logit) <= DEFAULT_ABS_TOLERANCE
    assert abs(res_mem.delta_logit - res_core.delta_logit) <= DEFAULT_ABS_TOLERANCE
    assert res_mem.clean_rank == res_core.clean_rank
    assert res_mem.intervened_rank == res_core.intervened_rank


def test_activation_artifact_lifecycle(tmp_path):
    t = torch.randn(1, 5, 768)
    art = ActivationArtifact.from_tensor(t, model_id="gpt2", layer=8)

    assert art.current_location == "ram"
    art.spill_to_disk(cache_dir=tmp_path)
    assert art.current_location == "disk"
    assert art.tensor_payload is None

    # Restore
    restored = art.get_tensor(target_device="cpu")
    assert restored.shape == (1, 5, 768)
    assert torch.allclose(t, restored)
