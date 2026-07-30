"""Tests for all Runtime Engineer epics (Patching, Debugger, Comparison, Logits, Batch Runs)."""
from api.dispatcher import build_dispatcher
from runtime.engine import get_engine


def test_activation_patching():
    dispatcher = build_dispatcher()
    res = dispatcher["runtime/patch"]({
        "session_id": "sess_test",
        "layer": 8,
        "component": "mlp",
        "neuron_index": 402,
        "operation": "replace",
        "value": 3.5,
    })
    assert res["status"] == "patch_applied"
    assert res["patch"]["value"] == 3.5


def test_forward_debugger_stepping():
    dispatcher = build_dispatcher()
    start = dispatcher["runtime/debugger/start"]({
        "session_id": "dbg_test_1",
        "prompt": "Test prompt",
        "breakpoint": 4,
    })
    assert start["status"] == "initialized"

    step = dispatcher["runtime/debugger/step"]({"session_id": "dbg_test_1"})
    assert step["current_layer"] == 1

    cont = dispatcher["runtime/debugger/continue"]({"session_id": "dbg_test_1"})
    assert cont["status"] == "paused"
    assert cont["current_layer"] == 4


def test_model_comparison():
    dispatcher = build_dispatcher()
    res = dispatcher["runtime/compare"]({
        "prompt": "Test prompt",
        "model_a": "GPT-2 Small",
        "model_b": "Pythia 160M",
    })
    assert "activations" in res
    assert "predictions" in res
    assert res["predictions"]["top_token_match"] is True


def test_intermediate_logits():
    dispatcher = build_dispatcher()
    res = dispatcher["runtime/logits"]({"prompt": "France is", "layers": 12})
    assert res["total_layers"] == 12
    assert len(res["layer_projections"]) == 12


def test_batch_experiments():
    dispatcher = build_dispatcher()
    res = dispatcher["runtime/experiments"]({
        "experiment_id": "exp_batch_1",
        "model_name": "GPT-2 Small",
        "prompts": ["Prompt A", "Prompt B", "Prompt C"],
    })
    assert res["total_prompts"] == 3
    assert res["completed_prompts"] == 3
