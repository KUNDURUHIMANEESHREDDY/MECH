"""Tests for Execution Strategy Benchmarking, Strict Budget Policy Enforcement, and Archive Telemetry.

Validates Phase 12I:
1. Multi-strategy scientific fidelity vs hardware efficiency benchmark.
2. Strict memory budget policy enforcement and violation action handling (abort / evict).
3. Immutable experiment archive serialization with runtime strategy and execution telemetry.
"""

import pytest
import torch

from backend.runtime.residency_manager import (
    EnforcementMode,
    MemoryBudgetExceededError,
    MemoryBudgetPolicy,
    ResidencyManager,
    ViolationAction,
)
from backend.runtime.strategy_benchmark import ExecutionStrategyBenchmarkRunner
from backend.science.reproducibility.immutable_store import ImmutableExperimentStore
from backend.science.reproducibility.types import (
    ExecutionEnvironment,
    ExecutionTelemetry,
    ExperimentSpecification,
    ImmutableExperimentRun,
    ModelIdentity,
    ProvenanceChain,
    RuntimeStrategy,
)


def test_multi_strategy_benchmark_execution():
    """Runs identical causal experiment across In-Memory, Out-of-Core Pipelined, and Strict-Budget strategies."""
    runner = ExecutionStrategyBenchmarkRunner(model_id="gpt2", device="cpu")
    report = runner.run_benchmark(
        clean_prompt="The Eiffel Tower is in",
        target_token=" Paris",
        corrupted_prompt="The Colosseum is in",
        intervene_layer=8,
        intervene_neuron=412,
    )

    assert report.fidelity_verdict == "ALL_STRATEGIES_FAITHFUL"
    assert len(report.strategy_records) == 3

    for record in report.strategy_records:
        assert record.fidelity_status in ("IDENTICAL", "WITHIN_TOLERANCE")
        assert record.rank_shift == 0
        assert record.delta_logit_from_ref <= 1e-4

        if record.strategy_name != "Reference_InMemory":
            assert record.telemetry.peak_active_layers_observed <= 1
            assert record.telemetry.prefetch_hits > 0
            assert record.telemetry.memory_budget_satisfied is True


def test_strict_budget_policy_enforcement_and_abort():
    """Verifies that strict budget policy triggers MemoryBudgetExceededError when memory exceeds hard limit."""
    impossible_policy = MemoryBudgetPolicy(
        ram_limit_bytes=1024,  # 1 KB (impossible for any Python process)
        vram_limit_bytes=1024,
        safety_margin_bytes=0,
        enforcement_mode=EnforcementMode.STRICT,
        violation_action=ViolationAction.ABORT,
    )

    mgr = ResidencyManager(
        budget_policy=impossible_policy,
        default_compute_device="cpu",
    )

    # Register dummy layer loader
    mgr.register_layer_loader(0, lambda: torch.nn.Linear(10, 10))

    with pytest.raises(MemoryBudgetExceededError) as exc_info:
        mgr.load_layer_to_device(0)

    assert "Strict RAM budget exceeded" in str(exc_info.value)


def test_immutable_archive_includes_strategy_and_telemetry(tmp_path):
    """Verifies that ImmutableExperimentStore preserves RuntimeStrategy and ExecutionTelemetry across roundtrips."""
    db_path = tmp_path / "test_telemetry_repro.db"
    store = ImmutableExperimentStore(db_path=db_path)

    strategy = RuntimeStrategy(
        execution_runtime="out_of_core",
        max_active_layers=1,
        ram_budget_mb=8192.0,
        vram_budget_mb=4096.0,
        enforcement_mode="strict",
        violation_action="evict",
        prefetch_enabled=True,
    )

    telemetry = ExecutionTelemetry(
        peak_ram_mb=412.5,
        peak_vram_mb=0.0,
        total_load_time_ms=45.2,
        total_eviction_time_ms=12.1,
        total_compute_time_ms=120.4,
        prefetch_requests=11,
        prefetch_hits=10,
        prefetch_misses=1,
        prefetch_hit_rate_pct=90.9,
        compute_utilization_pct=67.7,
        io_stall_pct=32.3,
        memory_budget_satisfied=True,
    )

    run = ImmutableExperimentRun(
        run_id="run_strat_001",
        parent_run_id=None,
        experiment_type="ORIGINAL",
        title="Strategy & Telemetry Preserved Experiment",
        timestamp_utc="2026-08-15T22:00:00Z",
        model=ModelIdentity(
            model_id="gpt2",
            architecture="GPT2LMHeadModel",
            parameter_count=124439808,
            weights_hash="hash_gpt2_weights",
            revision_or_commit="main",
            tokenizer_hash="hash_gpt2_tok",
            config_hash="hash_gpt2_cfg",
        ),
        environment=ExecutionEnvironment(
            python_version="3.13.14",
            pytorch_version="2.6.0",
            transformers_version="4.49.0",
            cuda_version=None,
            gpu_name=None,
            gpu_compute_capability=None,
            os_platform="Windows",
            os_release="11",
            mech_version="2.0.0",
            git_commit_sha="commit_sha_123",
            dependency_lock_hash="lock_hash_abc",
        ),
        specification=ExperimentSpecification(
            clean_prompt="The Eiffel Tower is in",
            target_token=" Paris",
        ),
        provenance_chain=ProvenanceChain(
            logit_lens_divergence_layer=8,
            maximum_predictive_gain_layer=9,
            active_sae_candidates=["L8_N412"],
            dense_substrate_anchors=["L8_N412"],
            linear_projection_delta={"Paris": 0.42},
            edge_causal_effect=0.31,
            four_control_results=[],
            cross_prompt_stability=0.92,
            mediation_rescue_fraction=0.88,
            null_distribution_percentile=99.5,
            null_distribution_p_value=0.005,
            final_evidence_tier="CAUSALLY_VERIFIED",
        ),
        measurements={"delta_z": 0.31, "target_logit": 14.82},
        verdict="VERIFIED",
        runtime_strategy=strategy,
        execution_telemetry=telemetry,
    )

    saved = store.save_run(run)
    retrieved = store.get_run("run_strat_001")

    assert retrieved is not None
    assert retrieved.runtime_strategy.execution_runtime == "out_of_core"
    assert retrieved.runtime_strategy.max_active_layers == 1
    assert retrieved.execution_telemetry is not None
    assert retrieved.execution_telemetry.prefetch_hits == 10
    assert retrieved.execution_telemetry.memory_budget_satisfied is True
    assert retrieved.manifest_sha256 == saved.manifest_sha256
