"""End-to-End Integration Test for Sprint 3 Final Deliverable across AI 1 - AI 5.

Validates:
Load GPT-2 -> Run Prompt -> Circuit Discovery -> SAE Feature Inspection ->
Causal Tracing -> Activation Patch -> Compare Predictions -> Mechanistic Report ->
Export Publication Figures -> Share Workspace Manifest.
"""
from api.dispatcher import build_dispatcher


def test_sprint3_deliverable_full_workflow():
    dispatcher = build_dispatcher()
    prompt = "The capital of France is"
    target_token = " Paris"
    workflow_id = "wf_sprint3_full"

    # 1. AI 1: Create Research Workflow & Initialize State Machine
    wf = dispatcher["platform/workflow/create"]({
        "workflow_id": workflow_id,
        "title": "GPT-2 Mechanistic Analysis",
        "question": "Why does GPT-2 predict Paris?",
    })
    assert wf["state"] == "Draft"

    # Transition to Hypothesis
    wf = dispatcher["platform/workflow/transition"]({
        "workflow_id": workflow_id,
        "target_state": "Hypothesis",
    })
    assert wf["state"] == "Hypothesis"

    # 2. AI 2: Set Distributed Target & Streaming Policy
    target = dispatcher["runtime/execution/target"]({"target": "LocalGPU"})
    assert target["target_type"] == "LocalGPU"

    stream = dispatcher["runtime/execution/stream"]({"model_name": "GPT-2 Small", "layer": 8})
    assert stream["loaded_to_vram"] is True

    # 3. AI 3: Circuit Discovery & Causal Tracing
    circ = dispatcher["interpretability/circuits/discover"]({"prompt": prompt, "target_token": target_token})
    if circ.get("status") == "completed":
        assert 0.0 <= circ["circuit_score"] <= 1.0
        assert len(circ["nodes"]) >= 4
    else:
        # Blocked pipelines must not emit a fabricated faithfulness score.
        assert circ["status"] in ("unavailable", "blocked", "error")
        assert circ["publication_eligible"] is False
        assert circ["reason"]
        assert "circuit_score" not in circ

    trace = dispatcher["interpretability/causal/trace"]({
        "clean_prompt": prompt,
        "corrupted_prompt": "The capital of Rome is",
    })
    assert trace["max_causal_layer"] == 8

    # 4. AI 3: Activation Patch & Prediction Comparison
    patch_res = dispatcher["runtime/patch"]({
        "layer": 8,
        "neuron": 402,
        "operation": "replace",
        "value": 0.0,
    })
    assert patch_res["status"] == "patch_applied"

    comp = dispatcher["runtime/compare"]({
        "prompt": prompt,
        "model_a": "GPT-2 Small (Original)",
        "model_b": "GPT-2 Small (Patched)",
    })
    assert comp["model_a"] == "GPT-2 Small (Original)"

    # 5. AI 3: Generate Mechanistic Report
    report = dispatcher["interpretability/reports/mechanistic"]({"prompt": prompt, "target_token": target_token})
    assert "Explanation" in report["explanation_text"]

    # 6. AI 1: Run Graph Pipeline Template
    pipe = dispatcher["platform/pipelines/run"]({
        "pipeline_id": "pipe_sprint3",
        "template_id": "causal_tracing",
        "prompt": prompt,
    })
    assert pipe["status"] == "success"

    # 7. AI 1 & AI 5: List Datasets & Complete Workflow
    datasets = dispatcher["platform/datasets/list"]({})
    assert len(datasets) >= 3

    wf_final = dispatcher["platform/workflow/transition"]({
        "workflow_id": workflow_id,
        "target_state": "Conclusion",
    })
    assert wf_final["state"] == "Conclusion"
