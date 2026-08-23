"""Unit and integration tests for Live Tensor Forward Hook Intervention Engine."""

import pytest
from backend.interpretability.causal.live_intervention_engine import (
    LiveInterventionEngine,
    LiveInterventionResult,
)
from backend.core.capability_registry import CapabilityRegistry
from fastapi.testclient import TestClient
from backend.main import app


def test_live_intervention_engine_execution():
    engine = LiveInterventionEngine()

    # 1. Clean unperturbed run
    clean_res = engine.run_live_intervention(
        prompt="The Eiffel Tower is in the city of",
        target_token=" Paris",
        node_interventions={},
    )
    assert isinstance(clean_res, LiveInterventionResult)
    assert clean_res.clean_target_prob > 0.01
    assert clean_res.intervened_target_prob == clean_res.clean_target_prob
    assert len(clean_res.clean_top_tokens) >= 1
    assert clean_res.clean_top_tokens[0].token == " Paris"

    # 2. Live Knockout intervention (scale = 0.0)
    ablation_res = engine.run_live_intervention(
        prompt="The Eiffel Tower is in the city of",
        target_token=" Paris",
        node_interventions={"L6_MLP": 0.0},
    )
    assert ablation_res.intervened_logit_margin != ablation_res.clean_logit_margin
    assert ablation_res.execution_backend in (
        "PyTorch Live Tensor Forward Pass (Hook Intervened)",
        "Attribution-Scaled Dynamic Estimate (Offline Fallback)",
    )
    assert len(ablation_res.intervened_top_tokens) >= 1

    # 3. Multi-node intervention (Ablate MLP + Modify Attention Head)
    multi_res = engine.run_live_intervention(
        prompt="The Eiffel Tower is in the city of",
        target_token=" Paris",
        node_interventions={"L6_MLP": 0.0, "L8_H5": 2.5},
    )
    assert multi_res.intervened_logit_margin != clean_res.clean_logit_margin
    assert len(multi_res.intervened_top_tokens) >= 1



def test_live_intervention_tool_in_capability_registry():
    registry = CapabilityRegistry()
    assert registry.get_tool("run_live_tensor_intervention") is not None

    tool_res = registry.execute_tool("run_live_tensor_intervention", {
        "prompt": "The capital of France is",
        "target_token": " Paris",
        "node_interventions": {"L6_MLP": 0.0},
    })

    assert tool_res["status"] == "success"
    assert "intervention_result" in tool_res
    ir = tool_res["intervention_result"]
    assert "clean_target_prob" in ir
    assert "intervened_target_prob" in ir
    assert "execution_backend" in ir


def test_live_intervention_api_endpoint():
    client = TestClient(app)
    headers = {"X-API-Key": "mech_dev_key_default"}

    resp = client.post(
        "/api/assistant/intervene",
        json={
            "prompt": "The Eiffel Tower is in the city of",
            "target_token": " Paris",
            "node_interventions": {"L6_MLP": 0.0, "L8_H5": 1.5},
        },
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "is_live_tensor_execution" in data
    assert "clean_target_prob" in data
    assert "intervened_target_prob" in data
    assert len(data["intervened_top_tokens"]) >= 1
