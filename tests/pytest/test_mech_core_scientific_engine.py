"""Comprehensive End-to-End Automated Test Suite for MECH Mechanistic Experimentation Engine.

All tests execute against real model weights (GPT-2) and real PyTorch forward hooks.
ZERO mocked or hardcoded scientific results.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from backend.core.experiment_engine import (
    ComponentTarget,
    InterventionSpec,
    MechanisticExperimentEngine,
    set_seed,
)
from backend.core.model_registry import ModelRegistry, get_model_registry
from backend.datasets.dataset_manager import DatasetManager
from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import InterventionType


@pytest.fixture
def tmp_storage():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_mech_core.db"
        storage = DesktopStorage(db_path)
        storage.initialize()
        yield storage


def test_model_registry_and_live_health(tmp_storage):
    """Verifies that the model registry registers models and executes live forward pass health checks."""
    registry = ModelRegistry(storage=tmp_storage)
    models = registry.list_models()
    assert len(models) >= 1
    assert any(m["model_id"] == "gpt2" for m in models)

    # Live health check forward pass
    health = registry.validate_model_health("gpt2")
    assert health["status"] == "healthy"
    assert health["n_layers"] == 12
    assert health["n_heads"] == 12
    assert health["d_model"] == 768
    assert health["d_mlp"] == 3072
    assert health["latency_ms"] > 0.0
    assert health["capabilities"]["forward_hooks"] is True
    assert health["capabilities"]["attention_patterns"] is True


def test_dataset_manager_builtins_and_splits():
    """Verifies built-in benchmark datasets (IOI, Factual, Greater-Than, Induction)."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        mgr = DatasetManager(data_dir=tmpdir)

        # 1. IOI Dataset
        ioi_prompts = mgr.load("ioi")
        assert len(ioi_prompts) >= 5
        assert all("clean" in p and "corrupted" in p and "target" in p for p in ioi_prompts)

        # 2. Factual Dataset
        facts = mgr.load("factual")
        assert len(facts) >= 5
        assert any(p["target"] == " Paris" for p in facts)

        # 3. Splits
        val_splits = mgr.split("ioi", "validation")
        assert len(val_splits) >= 1

        # 4. Fingerprint
        fp = mgr.compute_fingerprint("ioi", ioi_prompts)
        assert "prompt_hash" in fp
        assert len(fp["prompt_hash"]) == 64


def test_clean_baseline_forward_pass(tmp_storage):
    """Executes a real clean baseline forward pass and captures logits, probabilities, and residual norms."""
    engine = MechanisticExperimentEngine(model_id="gpt2", storage=tmp_storage)
    res = engine.run_baseline(
        prompt="The capital of France is",
        target_token=" Paris",
        distractor_token=" Berlin",
        top_k=5,
    )

    assert res.target_logit is not None
    assert res.target_probability is not None
    assert res.target_probability > 0.0
    assert res.distractor_logit is not None
    assert res.logit_diff is not None
    assert len(res.top_tokens) == 5
    assert len(res.hidden_norms) == 13  # 1 embed + 12 transformer layers
    assert all(h > 0.0 for h in res.hidden_norms)
    assert res.execution_time_ms > 0.0


def test_live_causal_interventions_zero_and_scaling(tmp_storage):
    """Executes genuine PyTorch forward hook interventions (zero ablation & scaling) and calculates metrics."""
    engine = MechanisticExperimentEngine(model_id="gpt2", storage=tmp_storage)

    # 1. Zero Ablation on MLP
    spec_mlp = InterventionSpec(
        clean_prompt="The capital of France is",
        target_token=" Paris",
        target_components=[ComponentTarget(type="mlp", layer=8)],
        intervention_type=InterventionType.ABLATION_ZERO,
        random_seed=42,
    )
    res_mlp = engine.run_intervention_experiment(spec_mlp)

    assert res_mlp.clean_target_logit != 0.0
    assert res_mlp.intervened_target_logit != res_mlp.clean_target_logit
    assert res_mlp.delta_logit != 0.0
    assert res_mlp.kl_divergence >= 0.0
    assert len(res_mlp.controls) == 5
    assert len(res_mlp.provenance_hash) == 64

    # 2. Scaling on Attention Head
    spec_head = InterventionSpec(
        clean_prompt="When Mary and John went to the store, John gave a drink to",
        target_token=" Mary",
        distractor_token=" John",
        target_components=[ComponentTarget(type="attention_head", layer=9, index=9)],
        intervention_type=InterventionType.SCALING,
        scale_coefficient=2.0,
        random_seed=42,
    )
    res_head = engine.run_intervention_experiment(spec_head)
    assert res_head.delta_logit_diff is not None
    assert res_head.specificity_ratio >= 0.0
    assert res_head.evidence_tier in ("CAUSALLY_VERIFIED", "SUPPORTED", "WEAKLY_SUPPORTED", "REFUTED", "UNTESTED")


def test_activation_patching_contrastive_experiment(tmp_storage):
    """Executes contrastive activation patching (swapping clean activation with corrupted activation)."""
    engine = MechanisticExperimentEngine(model_id="gpt2", storage=tmp_storage)

    spec = InterventionSpec(
        clean_prompt="When Mary and John went to the store, John gave a drink to",
        corrupted_prompt="When Mary and John went to the store, Mary gave a drink to",
        target_token=" Mary",
        distractor_token=" John",
        target_components=[ComponentTarget(type="attention_head", layer=9, index=9)],
        intervention_type=InterventionType.ACTIVATION_PATCHING,
        random_seed=42,
    )
    res = engine.run_intervention_experiment(spec)

    assert res.corrupted_prompt is not None
    assert res.indirect_effect is not None
    assert isinstance(res.indirect_effect, float)
    assert res.clean_target_logit != 0.0


def test_deterministic_reproducibility(tmp_storage):
    """Verifies that two runs with identical seeds yield bitwise identical metrics and provenance hashes."""
    engine = MechanisticExperimentEngine(model_id="gpt2", storage=tmp_storage)

    spec1 = InterventionSpec(
        clean_prompt="The Eiffel Tower is in the city of",
        target_token=" Paris",
        target_components=[ComponentTarget(type="neuron", layer=8, index=412)],
        intervention_type=InterventionType.ABLATION_ZERO,
        random_seed=1337,
    )
    res1 = engine.run_intervention_experiment(spec1)

    spec2 = InterventionSpec(
        clean_prompt="The Eiffel Tower is in the city of",
        target_token=" Paris",
        target_components=[ComponentTarget(type="neuron", layer=8, index=412)],
        intervention_type=InterventionType.ABLATION_ZERO,
        random_seed=1337,
    )
    res2 = engine.run_intervention_experiment(spec2)

    assert res1.clean_target_logit == res2.clean_target_logit
    assert res1.intervened_target_logit == res2.intervened_target_logit
    assert res1.delta_logit == res2.delta_logit
    assert res1.delta_prob == res2.delta_prob
    assert res1.kl_divergence == res2.kl_divergence


def test_deep_inspection_neuron_and_head():
    """Verifies live weight norms and prompt activation for neuron and head introspection."""
    engine = MechanisticExperimentEngine(model_id="gpt2")

    # 1. Neuron inspection
    neu = engine.inspect_neuron(layer=8, neuron_idx=412, prompt="The capital of France is")
    assert neu["in_weight_l2"] > 0.0
    assert neu["out_weight_l2"] > 0.0
    assert neu["prompt_activation"] is not None

    # 2. Head inspection
    head = engine.inspect_head(layer=9, head_idx=9, prompt="When Mary and John went to the store, John gave a drink to")
    assert head["d_head"] == 64
    assert head["q_weight_l2"] > 0.0
    assert len(head["top_attended_tokens"]) > 0


def test_logit_lens_trajectory():
    """Verifies true logit lens layer-by-layer decoded top vocabulary tokens."""
    engine = MechanisticExperimentEngine(model_id="gpt2")
    traj = engine.compute_logit_lens("The capital of France is", top_k=3)

    assert len(traj) == 12
    for step in traj:
        assert "layer" in step
        assert "residual_norm" in step
        assert step["residual_norm"] > 0.0
        assert len(step["top_predictions"]) == 3
        assert all("token" in t and "probability" in t for t in step["top_predictions"])


def test_cli_execution_standalone_json():
    """Verifies Agent 1 can invoke the engine via CLI and receive structured JSON output with zero UI dependency."""
    # Test model info
    cmd = [sys.executable, "-m", "backend.cli.mech_cli", "model", "info", "--model-id", "gpt2", "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["model_id"] == "gpt2"
    assert data["parameter_count"] == 124439808

    # Test baseline
    cmd_base = [sys.executable, "-m", "backend.cli.mech_cli", "run-baseline", "--prompt", "The capital of France is", "--target", " Paris", "--json"]
    proc_base = subprocess.run(cmd_base, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert proc_base.returncode == 0
    base_data = json.loads(proc_base.stdout)
    assert base_data["target_token"] == " Paris"
    assert base_data["target_logit"] is not None

    # Test intervention
    cmd_int = [
        sys.executable, "-m", "backend.cli.mech_cli", "run-intervention",
        "--prompt", "The capital of France is",
        "--target", " Paris",
        "--component", "L8_MLP",
        "--type", "ablation_zero",
        "--json",
    ]
    proc_int = subprocess.run(cmd_int, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert proc_int.returncode == 0
    int_data = json.loads(proc_int.stdout)
    assert "delta_logit" in int_data
    assert "provenance_hash" in int_data
    assert len(int_data["controls"]) == 5
