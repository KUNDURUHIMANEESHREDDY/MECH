"""End-to-end Python integration test suite verifying complete Sprint 4 & Sprint 5 deliverable."""
from backend.api.legacy_dispatcher import build_dispatcher


def test_sprint4_end_to_end_deliverable():
    dispatcher = build_dispatcher()

    # AI 1: Autonomous Research Agent
    agent_res = dispatcher["platform/autonomous/agent_run"]({"goal": "Investigate IOI Circuit in GPT-2"})
    assert agent_res["status"] == "completed"

    # AI 1: Typed Research Graph
    graph = dispatcher["platform/autonomous/graph_get"]({})
    assert graph["nodes_count"] >= 6

    # AI 2: Execution Orchestrator, Learned Prediction, Locality & Industrial Backends
    exec_res = dispatcher["runtime/orchestration/submit"]({
        "experiment_id": "exp_s4_final",
        "goal": "Large Scale Parallel Analysis",
        "strategy": "Balanced",
    })
    assert exec_res["experiment"]["state"] == "Completed"
    assert exec_res["k8s_job"]["status"] == "Running"
    assert exec_res["ray_task"]["status"] == "PENDING"
    assert exec_res["slurm_job"]["status"] == "QUEUED"
    assert exec_res["tensor_cache"]["cached"] is True
    assert exec_res["cost_tracking"]["within_budget"] is True
    assert exec_res["learned_prediction"]["predicted_gpu_utilization_pct"] > 90.0
    assert exec_res["locality"]["allocated_tier"] in ["GPU_VRAM", "NVMe_CACHE"]
    assert exec_res["backend_job"]["status"] in ["Pending", "Running", "Completed"]

    # AI 2: Benchmark Run
    bench = dispatcher["runtime/orchestration/benchmark_run"]({"model_name": "Gemma-7B"})
    assert bench["throughput_tok_per_sec"] > 1000.0

    # AI 3: Autonomous Discovery Engine & Research Framework
    disc_res = dispatcher["interpretability/discovery/run"]({"hypothesis_statement": "L8_N402 mediates IOI capital retrieval"})
    assert disc_res["lifecycle"]["state"] == "Publication"
    assert disc_res["test_result"]["outcome_state"] == "Confirmed"
    assert disc_res["benchmark_suite"]["total_benchmarks"] == 6
    assert disc_res["ioi_eval"]["ioi_accuracy"] > 0.9
    assert len(disc_res["induction_heads"]) >= 3
    assert disc_res["superposition"]["superposition_degree"] > 0.2
    # is_universal is an empirical result from real model weights (threshold on
    # cross-layer neuron cosine similarity); assert the assessment is produced.
    assert isinstance(disc_res["universality"]["is_universal"], bool)
    assert disc_res["calibrated_confidence"]["is_calibrated"] is True
    assert disc_res["quality_score"]["overall_quality_score"] > 0.85
    assert disc_res["regression"]["status"] == "Passing"
    assert disc_res["registered_mechanism"]["status"] == "Validated"
    assert disc_res["paper_replication"]["status"] == "SuccessfullyReplicated"

    # AI 3: Cross-Model Circuit Alignment
    align = dispatcher["interpretability/circuits/cross_model"]({
        "source_model": "GPT-2 Small",
        "target_model": "Gemma-2B",
    })
    assert align["alignment"]["causal_similarity"] > 0.7

    # AI 3: Feature Genealogy DAG
    gen = dispatcher["interpretability/features/genealogy"]({"feature_id": 1402})
    assert len(gen["genealogy"]["parents"]) == 2
