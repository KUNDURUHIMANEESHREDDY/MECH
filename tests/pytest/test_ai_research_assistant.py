"""Unit and integration tests for AI Research Assistant Layer & Verified Capability Registry."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.core.capability_registry import CapabilityRegistry, VerifiedTool
from backend.research_platform.autonomous.ai_research_assistant import AIResearchAssistant


def test_capability_registry_tools_and_execution():
    registry = CapabilityRegistry()
    
    # 1. Test capability checks
    assert registry.supports("sae")
    assert registry.supports("patching")
    assert registry.supports("logit_lens")
    assert registry.supports("ai_research_assistant")

    # 2. Test tool listings
    tools = registry.list_tools()
    tool_names = [t["name"] for t in tools]
    assert "run_logit_lens" in tool_names
    assert "run_activation_patching" in tool_names
    assert "extract_sae_features" in tool_names
    assert "discover_circuit" in tool_names
    assert "run_path_patching" in tool_names

    # 3. Test Logit Lens tool execution
    ll_res = registry.execute_tool("run_logit_lens", {"prompt": "The Eiffel Tower is in", "model_name": "gpt2"})
    assert ll_res["status"] == "success"
    assert "layer_predictions" in ll_res
    assert len(ll_res["layer_predictions"]) == 12

    # 4. Test Activation Patching tool execution
    ap_res = registry.execute_tool("run_activation_patching", {
        "clean_prompt": "The Eiffel Tower is in",
        "corrupted_prompt": "The Colosseum is in",
        "target_token": " Paris",
    })
    assert ap_res["status"] == "success"
    assert "critical_components" in ap_res
    assert ap_res["total_clean_restoration"] > 0.4

    # 5. Test Edge-Level Path Patching tool execution
    pp_res = registry.execute_tool("run_path_patching", {
        "sender": "L5_H2",
        "receiver": "L6_MLP",
        "clean_prompt": "The Eiffel Tower is in",
        "corrupted_prompt": "The Colosseum is in",
        "target_token": " Paris",
    })
    assert pp_res["status"] == "success"
    assert "edge_mediation" in pp_res
    assert pp_res["edge_mediation"]["direct_path_effect"] > 0.0
    assert pp_res["edge_mediation"]["is_causally_transmitting"] is True



def test_ai_research_assistant_end_to_end_investigation():
    assistant = AIResearchAssistant()
    
    # Run investigation on canonical fact retrieval
    res = assistant.investigate(
        goal="Investigate why GPT-2 predicts Paris for the Eiffel Tower",
        model_name="gpt2",
    )
    
    assert res["status"] == "completed"
    assert "investigation_id" in res
    assert len(res["stages"]) >= 7
    
    # Verify circuit structure
    assert "circuit" in res
    nodes = res["circuit"]["nodes"]
    edges = res["circuit"]["edges"]
    assert len(nodes) >= 4
    assert len(edges) >= 3
    
    # Verify scientific validation report
    assert "validation_report" in res
    vr = res["validation_report"]
    assert "held_out_faithfulness" in vr
    assert "ablation_specificity_score" in vr
    assert "overall_evidence_tier" in vr
    assert vr["overall_evidence_tier"] in ("STRONG", "MODERATE")

    # Verify reflection and statistical significance
    reflection = res["reflection"]
    assert "causal_effect_drop_percentage" in reflection
    assert len(reflection["findings"]) >= 3
    
    # Verify history
    history = assistant.get_history()
    assert len(history) >= 1
    assert history[0]["investigation_id"] == res["investigation_id"]



def test_ai_research_assistant_api_endpoints():
    client = TestClient(app)
    headers = {"X-API-Key": "mech_dev_key_default"}
    
    # 1. GET /api/assistant/tools
    resp = client.get("/api/assistant/tools", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data
    assert data["count"] >= 4
    
    # 2. POST /api/assistant/execute-tool
    tool_resp = client.post(
        "/api/assistant/execute-tool",
        json={"tool_name": "extract_sae_features", "params": {"layer": 8, "top_k": 3}},
        headers=headers,
    )
    assert tool_resp.status_code == 200
    tool_data = tool_resp.json()
    assert tool_data["status"] == "success"
    assert len(tool_data["features"]) >= 3
    
    # 3. POST /api/assistant/investigate
    inv_resp = client.post(
        "/api/assistant/investigate",
        json={"goal": "Indirect Object Identification (IOI) Circuit", "model_name": "gpt2"},
        headers=headers,
    )
    assert inv_resp.status_code == 200
    inv_data = inv_resp.json()
    assert inv_data["status"] == "completed"
    assert "circuit" in inv_data
    assert "reflection" in inv_data
    
    # 4. GET /api/assistant/history
    hist_resp = client.get("/api/assistant/history", headers=headers)
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert len(hist_data["history"]) >= 1
