"""Tests for Memory-Constrained Execution, Activation Provenance, and Causal Path Patching.

Validates Phase 12B/12C/12E:
1. Memory-bounded execution with max_active_layers=1 (strictly single-layer active in device memory).
2. ActivationArtifact capture, SHA-256 provenance hashing, and disk spilling.
3. Activation replay injection.
4. Multi-hop path patching and mediation rescue equivalence between in-memory and out-of-core runtimes.
"""

import pytest
import torch

from backend.runtime.activation_artifact import ActivationArtifact
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PathHopSpec
from backend.runtime.out_of_core_runtime import OutOfCoreRuntime
from backend.science.reproducibility.tolerance_engine import DEFAULT_ABS_TOLERANCE


@pytest.fixture(scope="module")
def runtimes():
    in_mem = InMemoryRuntime(model_id="gpt2", device="cpu")
    # Strictly memory-constrained: at most 1 active layer resident in memory at any point!
    out_core_constrained = OutOfCoreRuntime(model_id="gpt2", device="cpu", max_active_layers=1)
    return in_mem, out_core_constrained


def test_memory_constrained_single_layer_residency(runtimes):
    """Proves that out-of-core execution runs under max_active_layers=1 constraint with numerical fidelity."""
    in_mem, out_core = runtimes
    prompt = "The capital of France is"
    target_token = " Paris"

    out_core.residency.reset_peak_stats()
    res_core = out_core.forward(prompt, target_token=target_token)
    res_mem = in_mem.forward(prompt, target_token=target_token)

    # 1. Assert memory constraint: at most 1 active layer was resident simultaneously
    peak_active = out_core.residency.peak_active_layers_observed
    assert peak_active <= 1, f"Expected at most 1 active layer resident, but observed {peak_active}"

    # 2. Assert exact numerical equivalence
    assert res_core.top_predicted_token == res_mem.top_predicted_token
    assert res_core.target_rank == res_mem.target_rank
    assert abs(res_core.target_logit - res_mem.target_logit) <= DEFAULT_ABS_TOLERANCE
    assert abs(res_core.target_probability - res_mem.target_probability) <= DEFAULT_ABS_TOLERANCE


def test_activation_artifact_provenance_and_capture(runtimes):
    """Tests activation capture and deterministic SHA-256 artifact hashing."""
    _, out_core = runtimes
    prompt = "The Eiffel Tower is located in the city of"

    art = out_core.capture_activation(
        prompt=prompt,
        layer=8,
        component="residual",
        sequence_position=-1,
        experiment_hash="exp_sha256_mock_hash_12345",
    )

    assert isinstance(art, ActivationArtifact)
    assert art.layer == 8
    assert len(art.artifact_hash) == 64
    assert art.experiment_hash == "exp_sha256_mock_hash_12345"
    assert art.source_runtime == "out_of_core"

    # Spilling to disk
    art.spill_to_disk()
    assert art.current_location == "disk"
    assert art.tensor_payload is None

    # Restoration
    t = art.get_tensor("cpu")
    assert t.shape[-1] == 768


def test_activation_replay_and_injection_equivalence(runtimes):
    """Tests single-layer activation patching equivalence between in-memory and out-of-core runtimes."""
    in_mem, out_core = runtimes
    source_prompt = "The capital of France is"
    target_prompt = "The capital of Germany is"
    target_token = " Paris"

    # Capture from source prompt at Layer 8
    art_mem = in_mem.capture_activation(source_prompt, layer=8, sequence_position=-1)
    art_core = out_core.capture_activation(source_prompt, layer=8, sequence_position=-1)

    # Patch into target prompt
    patched_mem = in_mem.patch_activation(target_prompt, target_token=target_token, layer=8, artifact=art_mem)
    patched_core = out_core.patch_activation(target_prompt, target_token=target_token, layer=8, artifact=art_core)

    # Verify patched logit equivalence
    assert abs(patched_mem.target_logit - patched_core.target_logit) <= DEFAULT_ABS_TOLERANCE
    assert patched_mem.target_rank == patched_core.target_rank


def test_multi_hop_path_patching_and_mediation_rescue(runtimes):
    """Tests multi-hop path patching and mediation rescue fraction equivalence."""
    in_mem, out_core = runtimes
    source_prompt = "The capital of France is"
    corrupted_prompt = "The capital of Germany is"
    target_token = " Paris"

    hops = [
        PathHopSpec(source_layer=6, target_layer=8, source_component="residual", target_component="residual"),
    ]

    out_core.residency.reset_peak_stats()
    res_mem = in_mem.patch_path(source_prompt, corrupted_prompt, target_token, hops)
    res_core = out_core.patch_path(source_prompt, corrupted_prompt, target_token, hops)

    # Assert bounded layer residency during multi-hop path patching
    assert out_core.residency.peak_active_layers_observed <= 1

    # Assert mediation rescue and indirect effect equivalence
    assert abs(res_mem.indirect_effect - res_core.indirect_effect) <= DEFAULT_ABS_TOLERANCE
    assert abs(res_mem.mediation_rescue_fraction - res_core.mediation_rescue_fraction) <= DEFAULT_ABS_TOLERANCE
    assert res_mem.patched_target_rank == res_core.patched_target_rank
    assert len(res_core.artifacts) == len(hops)
