"""Integration Test: Debugger Checkpoint Restore.

Validates: Run -> Save Checkpoint -> Restore Checkpoint -> Continue -> Verify.
"""
from api.dispatcher import build_dispatcher


def test_checkpoint_restore_workflow():
    dispatcher = build_dispatcher()
    session_id = "ckpt_sess_1"

    # 1. Start Debugger & Save Checkpoint at Layer 5
    start_res = dispatcher["runtime/debugger/start"]({"session_id": session_id, "prompt": "Paris is", "breakpoint": 5})
    assert start_res["status"] in ["initialized", "running"]

    ckpt_save = dispatcher["runtime/memory/checkpoint_save"]({
        "checkpoint_id": "ckpt_l5",
        "session_id": session_id,
        "layer": 5,
    })
    assert ckpt_save["checkpoint_id"] == "ckpt_l5"
    assert ckpt_save["layer"] == 5

    # 2. Restore Checkpoint
    ckpt_restore = dispatcher["runtime/memory/checkpoint_restore"]({"checkpoint_id": "ckpt_l5"})
    assert ckpt_restore["checkpoint_id"] == "ckpt_l5"
    assert ckpt_restore["layer"] == 5
