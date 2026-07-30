"""Tests for AI 2 Runtime Sprint 4 Large Scale Execution epics."""
from api.dispatcher import build_dispatcher


def test_runtime_orchestration_submit():
    dispatcher = build_dispatcher()
    res = dispatcher["runtime/orchestration/submit"]({
        "experiment_id": "exp_s4_test_1",
        "goal": "Test Parallel Analysis",
        "priority": 2,
        "strategy": "Speed",
    })
    assert res["experiment"]["state"] == "Completed"
    assert res["resource_plan"]["gpu_target"] == "NVIDIA H100"
    assert res["execution_result"]["status"] == "Completed"


def test_experiment_lifecycle_history():
    dispatcher = build_dispatcher()
    events = dispatcher["runtime/orchestration/events"]({})
    assert len(events) >= 1
    assert events[0]["state"] == "Completed"


def test_cloud_runtime_providers():
    dispatcher = build_dispatcher()
    providers = dispatcher["runtime/orchestration/providers"]({})
    assert len(providers) >= 5
    provider_names = [p["name"] for p in providers]
    assert "AWS" in provider_names
    assert "RunPod" in provider_names


def test_priority_experiment_queue():
    dispatcher = build_dispatcher()
    item = dispatcher["runtime/scheduler/queue_submit"]({
        "experiment_id": "exp_q_test",
        "priority": 5,
    })
    assert item["status"] == "Queued"
    assert item["priority"] == 5


def test_resource_optimizer_strategies():
    dispatcher = build_dispatcher()
    plan_mem = dispatcher["runtime/orchestration/optimize_resources"]({"strategy": "Memory"})
    assert plan_mem["precision"] == "INT8"

    plan_cost = dispatcher["runtime/orchestration/optimize_resources"]({"strategy": "Cost"})
    assert "RunPod" in plan_cost["gpu_target"]


def test_runtime_benchmark_run():
    dispatcher = build_dispatcher()
    bench = dispatcher["runtime/orchestration/benchmark_run"]({"model_name": "Gemma-7B"})
    assert bench["throughput_tok_per_sec"] > 1000.0
    assert bench["compression_ratio"] == 0.50


def test_checkpoint_recovery():
    dispatcher = build_dispatcher()
    rec = dispatcher["runtime/memory/checkpoint_recover"]({"checkpoint_id": "ckpt_test"})
    assert rec["recovery_status"] == "Restored"
    assert rec["restored_step"] == 4200
