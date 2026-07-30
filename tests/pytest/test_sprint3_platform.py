"""Tests for AI 1 Chief Architect Sprint 3 Research Platform."""
from api.dispatcher import build_dispatcher


def test_research_workflow_state_machine():
    dispatcher = build_dispatcher()
    # 1. Create Workflow
    wf = dispatcher["platform/workflow/create"]({
        "workflow_id": "wf_sprint3_1",
        "title": "IOI Circuit Discovery Workflow",
        "question": "Which heads mediate IOI in GPT-2?",
    })
    assert wf["state"] == "Draft"

    # 2. Transition State Machine
    t1 = dispatcher["platform/workflow/transition"]({"workflow_id": "wf_sprint3_1", "target_state": "Question"})
    assert t1["state"] == "Question"

    t2 = dispatcher["platform/workflow/transition"]({"workflow_id": "wf_sprint3_1", "target_state": "Hypothesis"})
    assert t2["state"] == "Hypothesis"


def test_graph_pipeline_engine():
    dispatcher = build_dispatcher()
    templates = dispatcher["platform/pipelines/templates"]({})
    assert len(templates) >= 5

    res = dispatcher["platform/pipelines/run"]({
        "pipeline_id": "pipe_ioi_1",
        "template_id": "causal_tracing",
        "prompt": "The capital of France is",
    })
    assert res["status"] == "success"
    assert len(res["stages"]) >= 4


def test_dataset_manager():
    dispatcher = build_dispatcher()
    ds_list = dispatcher["platform/datasets/list"]({})
    assert len(ds_list) >= 4

    samples = dispatcher["platform/datasets/stream"]({"dataset_id": "openwebtext", "limit": 3})
    assert len(samples) == 3
    assert samples[0]["dataset"] == "OpenWebText"


def test_plugin_sdk_registration():
    dispatcher = build_dispatcher()
    reg = dispatcher["platform/sdk/register"]({
        "plugin_id": "custom_probe_v1",
        "name": "Custom Linear Probe Plugin",
        "version": "1.0.0",
    })
    assert reg["status"] == "registered"
    assert reg["plugin_id"] == "custom_probe_v1"
