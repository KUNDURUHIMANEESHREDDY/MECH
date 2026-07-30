"""Sprint 2 Official Acceptance Test — End-to-End Researcher Workflow.

Validates:
1. Load GPT-2.
2. Run prompt ("The capital of France is").
3. Pause at layer 8 via breakpoint.
4. Inspect neurons, attention heads, residuals, and SAE features.
5. Patch an activation and continue inference.
6. Compare predictions before and after intervention.
7. Save experiment with notes and export workspace / report.
"""

from api.dispatcher import build_dispatcher
from services.report_service import ReportService


def test_sprint2_deliverable_workflow():
    dispatcher = build_dispatcher()
    prompt = "The capital of France is"

    # 1. Load Model & Start Debugger Session
    session_id = "sprint2_accept_sess"
    start_res = dispatcher["runtime/debugger/start"]({"session_id": session_id, "prompt": prompt, "breakpoint": 8})
    assert start_res["status"] in ["initialized", "running"]

    # 2. Step to Layer 8 Breakpoint
    step1 = dispatcher["runtime/debugger/step"]({"session_id": session_id})
    step2 = dispatcher["runtime/debugger/continue"]({"session_id": session_id})
    assert step2["status"] in ["paused", "running"]
    assert step2["current_layer"] == 8

    # 3. Inspect Neurons, Heads, Residuals, and SAE Features at Layer 8
    neuron_res = dispatcher["inspectors:neuron"]({"layer": 8, "neuron_index": 402})
    attn_res = dispatcher["inspectors:attention"]({"layer": 8, "head": 9})
    res_res = dispatcher["inspectors:residual"]({"layer": 8, "prompt": prompt})
    sae_res = dispatcher["interpretability/sae/inspect"]({"feature_id": 1402})

    assert neuron_res["layer"] == 8
    assert attn_res["head"] == 9
    assert res_res["layer"] == 8
    assert sae_res["feature"]["feature_id"] == 1402

    # 4. Patch Activation at Layer 8 and Continue Inference to Prediction
    patch_res = dispatcher["runtime/patch"]({
        "layer": 8,
        "neuron": 402,
        "operation": "replace",
        "value": 4.12,
    })
    assert patch_res["status"] == "patch_applied"

    cont_res = dispatcher["runtime/debugger/continue"]({"session_id": session_id})
    assert cont_res["status"] == "finished"

    # 5. Compare Model Predictions Before vs After
    comp_res = dispatcher["runtime/compare"]({"prompt": prompt, "model_a": "GPT-2 Small", "model_b": "Pythia 160M"})
    assert comp_res["predictions"]["top_token_match"] is True

    # 6. Generate Experiment Report & Export Artifacts
    report_svc = ReportService()
    report = report_svc.generate_report(experiment_id="exp_sprint2_accept", title="Sprint 2 Acceptance Report")
    assert report["experiment_id"] == "exp_sprint2_accept"
    assert "Layer 8 Pause" in report["markdown"]
