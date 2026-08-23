"""Unit and integration tests for Phase 14: Real Model Zoo Validation & Empirical Frontier Engine."""

import pytest
import torch

from backend.runtime.frontier_engine import (
    FrontierOperatingPoint,
    HardwareRecommendationResult,
    ScaleFidelityFrontierEngine,
)
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PrecisionProfile
from backend.runtime.model_zoo import (
    MODEL_ZOO_REGISTRY,
    ModelZooEntry,
    ScaleTier,
    get_zoo_entry,
    list_zoo_models,
)
from backend.runtime.out_of_core_runtime import OutOfCoreRuntime
from backend.runtime.zoo_evaluator import ModelZooEvaluator
from backend.runtime.zoo_probe_suite import (
    STANDARD_PROBES,
    StandardMechanisticProbe,
    StandardProbeExecutionResult,
    run_probe_on_runtime,
)


def test_model_zoo_registry_and_taxonomy():
    """Verifies that the Model Zoo registry contains complete taxonomy and metadata across all tiers."""
    assert len(MODEL_ZOO_REGISTRY) >= 8

    # Check Tier 0
    t0 = list_zoo_models(tier=ScaleTier.TIER_0_NANO)
    assert len(t0) >= 2
    assert any(e.model_id == "EleutherAI/pythia-70m" for e in t0)

    # Check Tier 1
    t1 = list_zoo_models(tier=ScaleTier.TIER_1_SMALL)
    assert len(t1) >= 3
    assert any(e.model_id == "gpt2" for e in t1)
    assert any(e.model_id == "facebook/opt-125m" for e in t1)

    # Check Tier 2
    t2 = list_zoo_models(tier=ScaleTier.TIER_2_MEDIUM)
    assert len(t2) >= 2
    assert any(e.model_id == "Qwen/Qwen2.5-0.5B" for e in t2)

    # Check lookup by ID
    entry = get_zoo_entry("gpt2")
    assert entry is not None
    assert entry.parameter_count == 124_439_808
    assert entry.num_layers == 12
    assert entry.hidden_dim == 768
    assert entry.norm_type == "layernorm"


def test_standard_probe_execution_across_model_runtimes():
    """Verifies that standard mechanistic probes execute identically on both In-Memory and Out-of-Core runtimes."""
    probe = STANDARD_PROBES[0]  # "The capital of France is" -> " Paris"

    # 1. In-Memory Runtime
    rt_inmem = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    res_inmem = run_probe_on_runtime(rt_inmem, probe)

    assert res_inmem.probe_id == probe.probe_id
    assert res_inmem.model_id == "gpt2"
    assert res_inmem.runtime_type == "in_memory"
    assert isinstance(res_inmem.clean_target_logit, float)
    assert res_inmem.clean_target_probability > 0.0
    assert len(res_inmem.logit_lens_trajectory) == 13
    assert isinstance(res_inmem.causal_delta_z, float)

    # 2. Out-of-Core Runtime
    rt_ooc = OutOfCoreRuntime(
        model_id="gpt2",
        device="cpu",
        max_active_layers=1,
        precision=PrecisionProfile(weight_dtype="float16", activation_dtype="float16"),
    )
    res_ooc = run_probe_on_runtime(rt_ooc, probe)

    assert res_ooc.probe_id == probe.probe_id
    assert res_ooc.model_id == "gpt2"
    assert res_ooc.runtime_type == "out_of_core"
    assert isinstance(res_ooc.clean_target_logit, float)
    assert res_ooc.clean_target_probability > 0.0
    assert len(res_ooc.logit_lens_trajectory) == 13


def test_real_checkpoint_model_zoo_evaluator():
    """Verifies that ModelZooEvaluator evaluates real checkpoints and builds cross-model matrix reports."""
    evaluator = ModelZooEvaluator(device="cpu")

    # Evaluate gpt2 and facebook/opt-125m under Out-of-Core FP16
    report = evaluator.evaluate_model_zoo(
        model_ids=["gpt2", "facebook/opt-125m"],
        strategies=[("OutOfCore_1Layer_FP16", PrecisionProfile(weight_dtype="float16", activation_dtype="float16"))],
        probes=[STANDARD_PROBES[0]],
    )

    assert report.total_checkpoints_evaluated >= 1
    assert len(report.records) >= 1

    rec = report.records[0]
    assert rec.model_id == "gpt2"
    assert rec.num_layers == 12
    assert len(rec.weights_hash) > 0
    assert len(rec.tokenizer_hash) > 0
    assert rec.telemetry.max_active_layers == 1
    assert rec.telemetry.memory_budget_satisfied is True
    assert len(rec.probe_results) == 1


def test_scale_fidelity_frontier_pareto_and_recommendation():
    """Verifies that ScaleFidelityFrontierEngine computes the Pareto frontier and recommends optimal configurations."""
    engine = ScaleFidelityFrontierEngine()

    # 1. Check points and Pareto frontier
    assert len(engine.points) >= 15
    frontier = engine.get_pareto_frontier()
    assert len(frontier) >= 3

    # All Pareto optimal points must not be dominated
    for pt in frontier:
        assert pt.is_pareto_optimal is True

    # 2. Query hardware recommendation for tight 500MB RAM budget
    rec_500mb = engine.recommend_optimal_configuration(
        ram_budget_mb=500.0,
        vram_budget_mb=0.0,
        max_error_tolerance=0.10,
    )
    assert rec_500mb.recommended_point is not None
    assert rec_500mb.recommended_point.peak_ram_mb <= 500.0
    assert rec_500mb.viable_candidates_count > 0
    assert "Recommended" in rec_500mb.recommendation_rationale

    # 3. Query hardware recommendation for larger 2500MB RAM budget
    rec_2500mb = engine.recommend_optimal_configuration(
        ram_budget_mb=2500.0,
        vram_budget_mb=0.0,
        max_error_tolerance=0.15,
    )
    assert rec_2500mb.recommended_point is not None
    assert rec_2500mb.recommended_point.peak_ram_mb <= 2500.0
    # Higher RAM budget should allow a larger model (e.g. >= 1B parameters)
    assert rec_2500mb.recommended_point.parameter_count >= rec_500mb.recommended_point.parameter_count
