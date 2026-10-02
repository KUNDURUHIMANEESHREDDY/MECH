"""End-to-end Python integration test suite verifying complete Sprint 4 & Sprint 5 deliverable."""
from api.dispatcher import build_dispatcher


def test_sprint4_end_to_end_deliverable():
    dispatcher = build_dispatcher()

    # AI 1: Autonomous Research Agent
    #
    # This asserted `status == "completed"`. It completed only because the
    # publish step used to succeed unconditionally. All seven workflow steps
    # now run live and report provenance "live", but the reproduction gate
    # measures ~62% fidelity against an 0.85 threshold, so Scribe refuses to
    # publish. `blocked` is the correct outcome: the system ran the science and
    # the science did not clear the bar.
    agent_res = dispatcher["platform/autonomous/agent_run"]({"goal": "Investigate IOI Circuit in GPT-2"})
    assert agent_res["status"] == "blocked"
    assert agent_res["publication_eligible"] is False
    assert agent_res["validation_eligible"] is False

    # The block must be explained and attributable to the gate, not a shrug.
    assert "gate" in str(agent_res["publication"]["reason"]).lower()

    # Crucially, every step really did execute against live weights. A block
    # that hides un-run steps would look identical from the outside.
    steps = {n.get("node"): n for n in agent_res["trace"]}
    assert set(steps) >= {"load", "reproduce", "inspect", "patch",
                          "discover", "validate", "publish"}
    for node, step in steps.items():
        assert step["status"] in ("loaded", "completed", "ok"), node
        assert step["provenance"] == "live", node

    # The reproduction measurement is real and is what blocked publication.
    repro = next(n for n in agent_res["trace"] if n["node"] == "reproduce")
    assert repro["result"]["provenance"] == "live"
    assert repro["result"]["observed_metrics"]["circuit_faithfulness"] > 0.0

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
    # GPU utilisation was a constant reported under a "predicted_" name.
    assert exec_res["learned_prediction"]["gpu_utilization_is_estimate"] is True
    assert exec_res["learned_prediction"]["validation_eligible"] is False
    assert exec_res["locality"]["allocated_tier"] in ["GPU_VRAM", "NVMe_CACHE"]
    assert exec_res["backend_job"]["status"] in ["Pending", "Running", "Completed"]

    # AI 2: Benchmark Run
    bench = dispatcher["runtime/orchestration/benchmark_run"]({"model_name": "Gemma-7B"})
    assert bench["throughput_tok_per_sec"] > 1000.0

    # AI 3: Autonomous Discovery Engine & Research Framework
    #
    # This previously asserted a wall of successes for any input: lifecycle
    # Publication, a Confirmed hypothesis, 6 benchmarks, IOI accuracy > 0.9,
    # a "Validated" mechanism and a "SuccessfullyReplicated" paper -- none of
    # which depended on a measurement. The orchestrator now runs live causal
    # discovery and stops at Validation.
    disc_res = dispatcher["interpretability/discovery/run"]({"hypothesis_statement": "L8_N402 mediates IOI capital retrieval"})

    assert disc_res["lifecycle"]["state"] != "Publication"

    if disc_res.get("status") == "completed" and disc_res.get("provenance") == "live":
        # Live measurements must be real, structural measurements.
        assert disc_res["method"]
        assert disc_res["model_id"]
        assert isinstance(disc_res["heads"], list)
        assert isinstance(disc_res["head_effects"], list)
        assert disc_res["n_prompts"] >= 1
        assert isinstance(disc_res["faithfulness"], (int, float))
    else:
        assert disc_res["status"] in ("unavailable", "blocked", "error")
        assert disc_res["validation_eligible"] is False
        assert disc_res["publication_eligible"] is False
        assert disc_res["reason"]

    # The synthetic finding fields must not be present at all any more.
    for removed in ("test_result", "ioi_eval", "induction_heads",
                    "superposition", "universality", "calibrated_confidence",
                    "quality_score", "regression", "registered_mechanism",
                    "paper_replication", "benchmark_suite"):
        assert removed not in disc_res, (
            f"{removed} is a synthetic field and must not be returned as a "
            "discovery result")

    # AI 3: Cross-Model Circuit Alignment -- unmeasured, so must say so
    align = dispatcher["interpretability/circuits/cross_model"]({
        "source_model": "GPT-2 Small",
        "target_model": "Gemma-2B",
    })
    assert align["alignment_measured"] is False
    assert align["publication_eligible"] is False

    # AI 3: Feature Genealogy DAG -- reference fixture, not an inferred lineage
    gen = dispatcher["interpretability/features/genealogy"]({"feature_id": 1402})
    assert gen["genealogy_measured"] is False
    assert gen["publication_eligible"] is False
