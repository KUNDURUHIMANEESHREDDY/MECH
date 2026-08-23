"""Comprehensive Test Suite for Model-Agnostic Discovery Engine & Agent 1 -> Agent 2 API Contract.

All tests execute against live GPT-2 weights with PyTorch tensor hooks.
Zero mocked or hardcoded scientific results.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from backend.core.discovery_engine import AutomatedDiscoveryEngine
from backend.core.experiment_engine import (
    ComponentTarget,
    InterventionSpec,
    MechanisticExperimentEngine,
)
from backend.core.model_adapter import GPT2Adapter, get_model_adapter
from backend.main import app
from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import InterventionType


@pytest.fixture
def tmp_storage():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_discovery.db"
        storage = DesktopStorage(db_path)
        storage.initialize()
        yield storage


def test_model_adapter_introspection():
    """Verifies that ModelAdapter dynamically introspects model architecture without hardcoding."""
    adapter = get_model_adapter(model_id="gpt2")
    adapter.load()

    assert adapter.n_layers == 12
    assert adapter.n_heads == 12
    assert adapter.d_model == 768
    assert adapter.d_mlp == 3072
    assert adapter.d_head == 64
    assert adapter.vocab_size == 50257
    assert adapter.parameter_count == 124439808
    assert len(adapter.model_hash) == 64

    # Unembedding matrix
    W_U = adapter.get_unembedding_weight()
    assert W_U.shape == (50257, 768)

    # Sub-module retrieval
    block0 = adapter.get_layer_block(0)
    attn0 = adapter.get_attention_module(0)
    mlp0 = adapter.get_mlp_fc_module(0)
    assert block0 is not None
    assert attn0 is not None
    assert mlp0 is not None


def test_multi_trial_repeated_runs_and_statistics(tmp_storage):
    """Verifies that multi-trial experiments calculate sample variance, SE, and 95% Confidence Intervals."""
    engine = MechanisticExperimentEngine(model_id="gpt2", storage=tmp_storage)

    spec = InterventionSpec(
        clean_prompt="When Mary and John went to the store, John gave a drink to",
        target_token=" Mary",
        target_components=[ComponentTarget(type="attention_head", layer=9, index=9)],
        intervention_type=InterventionType.ABLATION_NOISE,  # Noise intervention introduces stochasticity across seeds
        random_seed=42,
        repeats=4,
    )
    res = engine.run_intervention_experiment(spec)

    assert res.repeats == 4
    assert len(res.multi_trial_stats.seeds) == 4
    assert len(res.multi_trial_stats.delta_logits) == 4
    assert res.multi_trial_stats.ci95_low <= res.multi_trial_stats.mean_delta_logit <= res.multi_trial_stats.ci95_high
    assert res.multi_trial_stats.se_delta_logit >= 0.0
    assert len(res.provenance_hash) == 64


def test_5tier_control_battery(tmp_storage):
    """Verifies all 5 tiers of the enhanced control battery (Positive, Sham, Matched, Same-Layer, Random)."""
    engine = MechanisticExperimentEngine(model_id="gpt2", storage=tmp_storage)

    spec = InterventionSpec(
        clean_prompt="The capital of France is",
        target_token=" Paris",
        target_components=[ComponentTarget(type="mlp", layer=8)],
        intervention_type=InterventionType.ABLATION_ZERO,
        random_seed=42,
        repeats=1,
    )
    res = engine.run_intervention_experiment(spec)

    assert len(res.controls) == 5
    cat_names = [c.control_category for c in res.controls]
    assert "POSITIVE" in cat_names
    assert "SHAM" in cat_names
    assert "MATCHED_NORM" in cat_names
    assert "SAME_LAYER" in cat_names
    assert "RANDOM_GLOBAL" in cat_names

    # Positive control must pass (destroying target unembedding direction drops logit)
    assert res.positive_control_passed is True

    # Sham control must pass (coeff=1.0 identity hook must not distort tensor)
    assert res.sham_control_passed is True


def test_automated_layer_scan():
    """Verifies that layer-wise activation patching identifies causal processing windows."""
    discovery = AutomatedDiscoveryEngine(model_id="gpt2")

    layers = discovery.scan_layers(
        clean_prompt="When Mary and John went to the store, John gave a drink to",
        corrupted_prompt="When Mary and John went to the store, Mary gave a drink to",
        target_token=" Mary",
    )

    assert len(layers) == 12
    assert all(isinstance(l.delta_logit, float) for l in layers)
    assert all(isinstance(l.indirect_effect, float) for l in layers)
    assert any(l.is_causal_window for l in layers)


def test_automated_head_scan_and_circuit_discovery(tmp_storage):
    """Verifies that the automated discovery engine scans attention heads and synthesizes validated circuits."""
    discovery = AutomatedDiscoveryEngine(model_id="gpt2", storage=tmp_storage)

    report = discovery.discover_and_validate_circuit(
        clean_prompt="When Mary and John went to the store, John gave a drink to",
        corrupted_prompt="When Mary and John went to the store, Mary gave a drink to",
        target_token=" Mary",
        distractor_token=" John",
        max_candidates_to_validate=3,
        validation_repeats=2,
    )

    assert report.total_heads_scanned == 144
    assert len(report.top_candidates) >= 3
    assert len(report.layer_scan) == 12
    assert report.execution_time_ms > 0.0

    # Top candidate should have causal rank assigned and validation status evaluated
    top_cand = report.top_candidates[0]
    assert top_cand.causal_rank == 1
    assert top_cand.validation_status in ("CAUSALLY_VERIFIED", "SUPPORTED", "WEAKLY_SUPPORTED", "REFUTED")


def test_api_endpoints_agent1_agent2_contract():
    """Verifies the complete Agent 1 -> Agent 2 REST API contract via FastAPI TestClient."""
    client = TestClient(app)

    # 1. GET /models
    res_models = client.get("/api/v1/models")
    assert res_models.status_code == 200
    models_data = res_models.json()
    assert len(models_data) >= 1

    # 2. GET /models/{id}/architecture
    res_arch = client.get("/api/v1/models/gpt2/architecture")
    assert res_arch.status_code == 200
    arch_data = res_arch.json()
    assert arch_data["n_layers"] == 12
    assert arch_data["n_heads"] == 12

    # 3. POST /hypotheses
    res_hyp = client.post(
        "/api/v1/hypotheses",
        json={
            "title": "L9H9 Name Mover Hypothesis",
            "statement": "Head L9H9 causally mediates indirect object token copying",
            "target_component": "L9H9",
        },
    )
    assert res_hyp.status_code == 201
    hyp_data = res_hyp.json()
    hyp_id = hyp_data["id"]

    # 4. POST /experiments
    res_exp = client.post(
        "/api/v1/experiments",
        json={
            "name": "IOI L9H9 Causal Test",
            "clean_prompt": "When Mary and John went to the store, John gave a drink to",
            "corrupted_prompt": "When Mary and John went to the store, Mary gave a drink to",
            "target_token": " Mary",
            "distractor_token": " John",
            "source_component": "L9H9",
            "intervention_type": "ablation_zero",
            "repeats": 2,
            "hypothesis_id": hyp_id,
        },
    )
    assert res_exp.status_code in (200, 201)
    exp_data = res_exp.json()
    exp_id = exp_data["id"]

    # 5. POST /experiments/{id}/run
    res_run = client.post(f"/api/v1/experiments/{exp_id}/run", json={"repeats": 2, "seed": 42})
    assert res_run.status_code == 200
    run_data = res_run.json()
    assert "delta_logit" in run_data
    assert "provenance_hash" in run_data
    assert len(run_data["controls"]) == 5

    # 6. GET /experiments/{id}/runs
    res_runs = client.get(f"/api/v1/experiments/{exp_id}/runs")
    assert res_runs.status_code == 200
    assert len(res_runs.json()) >= 1

    # 7. POST /interventions (Ad-hoc)
    res_int = client.post(
        "/api/v1/interventions",
        json={
            "prompt": "The capital of France is",
            "target": " Paris",
            "component": "L8_MLP",
            "type": "ablation_zero",
            "repeats": 2,
        },
    )
    assert res_int.status_code == 200
    assert "delta_logit" in res_int.json()
