"""Integration Test: Patch + Continue Workflow.

Validates the full workflow:
Run -> Pause -> Patch -> Continue -> Prediction / State Updated.
"""
from api.dispatcher import build_dispatcher


def test_patch_and_continue_workflow():
    dispatcher = build_dispatcher()
    session_id = "integration_sess_1"

    # 1. Start debug session with breakpoint at Layer 5
    start_res = dispatcher["runtime/debugger/start"]({
        "session_id": session_id,
        "prompt": "When Mary and John went to the store, John gave a book to",
        "breakpoint": 5,
    })
    assert start_res["status"] == "initialized"

    # 2. Continue until breakpoint is hit at Layer 5
    cont_res = dispatcher["runtime/debugger/continue"]({"session_id": session_id})
    assert cont_res["status"] == "paused"
    assert cont_res["current_layer"] == 5

    # 3. Apply activation patch at Layer 5
    patch_res = dispatcher["runtime/patch"]({
        "session_id": session_id,
        "layer": 5,
        "component": "mlp",
        "neuron_index": 402,
        "operation": "replace",
        "value": 5.0,
    })
    assert patch_res["status"] == "patch_applied"

    # 4. Continue execution to completion
    finish_res = dispatcher["runtime/debugger/continue"]({"session_id": session_id})
    assert finish_res["status"] == "finished"
    assert finish_res["current_layer"] == 12
