import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_research_router_investigations_and_hypotheses():
    # 1. List investigations
    res = client.get("/api/v1/research/investigations")
    assert res.status_code == 200
    data = res.json()
    assert "investigations" in data
    assert len(data["investigations"]) > 0

    inv_id = data["investigations"][0]["id"]

    # 2. Get specific investigation
    res_inv = client.get(f"/api/v1/research/investigations/{inv_id}")
    assert res_inv.status_code == 200
    inv_data = res_inv.json()
    assert inv_data["investigation"]["id"] == inv_id

    # 3. Create hypothesis
    new_hyp = {
        "investigation_id": inv_id,
        "title": "L8H3 Induction Test",
        "statement": "L8H3 acts as an induction head.",
        "target_component": "L8H3",
        "prediction": "Ablation reduces induction score.",
        "expected_evidence": "Positive delta prob.",
        "falsification_condition": "Delta prob <= 0",
    }
    res_hyp = client.post("/api/v1/research/hypotheses", json=new_hyp)
    assert res_hyp.status_code == 200
    hyp_id = res_hyp.json()["hypothesis"]["id"]

    # 4. Run Causal Experiment
    exp_payload = {
        "investigation_id": inv_id,
        "hypothesis_id": hyp_id,
        "name": "L8H3 Ablation Run",
        "clean_prompt": "When Mary and John went to the store, John gave a drink to",
        "target_token": " Mary",
        "intervention_type": "ABLATION_ZERO",
        "source_component": "L8H3",
        "control_component": "L0H0",
    }
    res_run = client.post("/api/v1/research/experiments/run", json=exp_payload)
    assert res_run.status_code == 200
    run_data = res_run.json()
    assert run_data["status"] == "success"
    assert "run" in run_data
    assert run_data["run"]["delta_logit"] is not None

    # 5. Evaluate Hypothesis
    res_eval = client.post(f"/api/v1/research/hypotheses/{hyp_id}/evaluate?investigation_id={inv_id}")
    assert res_eval.status_code == 200
    eval_data = res_eval.json()
    assert "status" in eval_data
    assert "verdict" in eval_data

    # 6. Generate Research Report
    res_rep = client.post("/api/v1/research/artifacts/generate-report", json={"investigation_id": inv_id})
    assert res_rep.status_code == 200
    rep_data = res_rep.json()
    assert "report_markdown" in rep_data
    assert "# Scientific Research Report" in rep_data["report_markdown"]
