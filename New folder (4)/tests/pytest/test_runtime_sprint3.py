"""Tests for AI 2 Runtime Engineer Sprint 3 Execution Platform epics."""
from api.dispatcher import build_dispatcher


def test_distributed_execution_target():
    dispatcher = build_dispatcher()
    target_res = dispatcher["runtime/execution/target"]({"target": "RemoteGPU"})
    assert target_res["target_type"] == "RemoteGPU"
    assert target_res["status"] == "ready"


def test_multi_gpu_planner():
    dispatcher = build_dispatcher()
    plan = dispatcher["runtime/execution/multi_gpu"]({"model_name": "Gemma-7B", "num_gpus": 4, "strategy": "TensorParallel"})
    assert plan["model_name"] == "Gemma-7B"
    assert plan["num_gpus"] == 4
    assert plan["strategy"] == "TensorParallel"


def test_on_demand_layer_streaming():
    dispatcher = build_dispatcher()
    stream_res = dispatcher["runtime/execution/stream"]({"model_name": "GPT-2 Small", "layer": 5})
    assert stream_res["layer"] == 5
    assert stream_res["loaded_to_vram"] is True


def test_experiment_scheduler():
    dispatcher = build_dispatcher()
    job = dispatcher["runtime/scheduler/submit"]({"job_id": "job_prio_1", "priority": 10})
    assert job["job_id"] == "job_prio_1"
    assert job["status"] == "scheduled"

    workers = dispatcher["runtime/scheduler/workers"]({})
    assert len(workers) >= 2
