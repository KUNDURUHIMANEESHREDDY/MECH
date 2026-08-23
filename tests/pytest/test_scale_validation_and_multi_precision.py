"""Unit and integration tests for Phase 13: Empirical Scale Validation & Multi-Precision Benchmarking."""

import pytest
import torch
from transformers import AutoConfig, GPT2Config, LlamaConfig

from backend.runtime.interfaces import PrecisionProfile
from backend.runtime.out_of_core_runtime import OutOfCoreRuntime
from backend.runtime.scale_benchmark import ScaleValidationBenchmarkRunner
from backend.runtime.universal_adapter import UniversalModelAdapter
from backend.science.reproducibility.tolerance_engine import ReproductionToleranceEngine
from backend.science.reproducibility.types import (
    ExecutionEnvironment,
    ExecutionTelemetry,
    ExperimentSpecification,
    ImmutableExperimentRun,
    ModelIdentity,
    ProvenanceChain,
    RuntimeStrategy,
)




def test_universal_architecture_adapter_detection():
    """Verifies that UniversalModelAdapter correctly introspects GPT-2 and LLaMA topologies."""
    # 1. GPT-2
    gpt2_cfg = GPT2Config(n_layer=12, n_embd=768, n_head=12, vocab_size=50257)
    adapter_gpt2 = UniversalModelAdapter(config=gpt2_cfg)
    assert adapter_gpt2.topology.architecture_family == "gpt2"
    assert adapter_gpt2.topology.num_layers == 12
    assert adapter_gpt2.topology.hidden_dim == 768
    assert adapter_gpt2.topology.norm_type == "layernorm"
    assert adapter_gpt2.topology.has_absolute_pos_embeddings is True
    assert adapter_gpt2.topology.has_rotary_embeddings is False

    # 2. LLaMA
    llama_cfg = LlamaConfig(num_hidden_layers=32, hidden_size=4096, num_attention_heads=32, vocab_size=32000)
    adapter_llama = UniversalModelAdapter(config=llama_cfg)
    assert adapter_llama.topology.architecture_family == "llama"
    assert adapter_llama.topology.num_layers == 32
    assert adapter_llama.topology.hidden_dim == 4096
    assert adapter_llama.topology.norm_type == "rmsnorm"
    assert adapter_llama.topology.has_rotary_embeddings is True
    assert adapter_llama.topology.has_absolute_pos_embeddings is False


def test_multi_precision_out_of_core_execution():
    """Verifies that OutOfCoreRuntime executes across FP32, FP16, and INT8 precision profiles."""
    prompt = "The Eiffel Tower is in"
    target = " Paris"

    # FP32
    rt_fp32 = OutOfCoreRuntime(
        model_id="gpt2",
        device="cpu",
        max_active_layers=1,
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    fwd_fp32 = rt_fp32.forward(prompt, target_token=target)
    assert fwd_fp32.target_logit is not None
    assert fwd_fp32.target_rank is not None
    assert fwd_fp32.target_rank <= 10


    # FP16 / BF16 casting
    rt_fp16 = OutOfCoreRuntime(
        model_id="gpt2",
        device="cpu",
        max_active_layers=1,
        precision=PrecisionProfile(weight_dtype="float16", activation_dtype="float16"),
    )
    fwd_fp16 = rt_fp16.forward(prompt, target_token=target)
    assert fwd_fp16.target_logit is not None
    # Check that FP16 logit matches FP32 logit within 0.25 (empirical float16 precision delta)
    assert abs(fwd_fp16.target_logit - fwd_fp32.target_logit) < 0.25


    # INT8 Dynamic Quantization
    rt_int8 = OutOfCoreRuntime(
        model_id="gpt2",
        device="cpu",
        max_active_layers=1,
        precision=PrecisionProfile(weight_dtype="int8", activation_dtype="float32", quantization_scheme="dynamic_int8"),
    )
    fwd_int8 = rt_int8.forward(prompt, target_token=target)
    assert fwd_int8.target_logit is not None
    assert abs(fwd_int8.target_logit - fwd_fp32.target_logit) < 0.2


def test_scale_validation_benchmark_runner():
    """Verifies that ScaleValidationBenchmarkRunner executes multi-strategy precision benchmark."""
    runner = ScaleValidationBenchmarkRunner(model_id="gpt2", device="cpu")
    report = runner.run_precision_matrix(
        clean_prompt="The Eiffel Tower is in",
        target_token=" Paris",
        corrupted_prompt="The Colosseum is in",
        intervene_layer=8,
        intervene_neuron=412,
        include_quantized=True,
    )

    assert report.model_id == "gpt2"
    assert report.num_layers == 12
    assert len(report.profiles) >= 3

    profile_names = [p.profile_name for p in report.profiles]
    assert "Reference_InMemory_FP32" in profile_names
    assert "OutOfCore_1Layer_FP32" in profile_names
    assert "OutOfCore_1Layer_FP16" in profile_names

    # Check that all profiles maintained low rank shift
    for p in report.profiles:
        assert p.rank_shift_from_ref <= 1
        assert isinstance(p.scientific_result.target_logit, float)
        assert p.scientific_result.target_probability > 0.0
        assert p.hardware_telemetry.memory_budget_satisfied is True




def test_cross_strategy_replication_provenance():
    """Verifies that ReproductionToleranceEngine categorizes STRICT vs CROSS_STRATEGY replications."""
    base_model = ModelIdentity(
        model_id="gpt2",
        architecture="GPT2LMHeadModel",
        parameter_count=124439808,
        weights_hash="hash_gpt2_weights",
        revision_or_commit="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        tokenizer_hash="hash_gpt2_tok",
        config_hash="config_hash_sample",
    )
    base_env = ExecutionEnvironment(
        python_version="3.13.14",
        pytorch_version=torch.__version__,
        transformers_version="4.40.0",
        cuda_version=None,
        gpu_name=None,
        gpu_compute_capability=None,
        os_platform="Windows",
        os_release="11",
        mech_version="2.0.0",
        git_commit_sha="c6f9b28a",
        dependency_lock_hash="lock_abc123",
        deterministic_mode=True,
    )
    base_spec = ExperimentSpecification(
        clean_prompt="The Eiffel Tower is in",
        target_token=" Paris",
        layer=8,
        component_type="neuron",
        component_index=412,
        intervention_type="zero_ablation",
    )
    prov = ProvenanceChain(
        logit_lens_divergence_layer=6,
        maximum_predictive_gain_layer=8,
        active_sae_candidates=["SAE_L8_F1842"],
        dense_substrate_anchors=["L8_N412"],
        linear_projection_delta={" Paris": 2.14},
        edge_causal_effect=1.07,
        four_control_results=[],
        cross_prompt_stability=0.94,
        mediation_rescue_fraction=0.88,
        null_distribution_percentile=99.4,
        null_distribution_p_value=0.006,
        final_evidence_tier="CAUSALLY_VERIFIED",
    )
    meas = {
        "clean_logit": 14.50,
        "intervened_logit": 11.20,
        "delta_logit": -3.30,
        "clean_probability": 0.72,
        "intervened_probability": 0.15,
        "clean_rank": 0,
        "intervened_rank": 2,
    }

    # Run 1: Reference In-Memory FP32
    run1 = ImmutableExperimentRun(
        run_id="run_001",
        parent_run_id=None,
        experiment_type="ORIGINAL",
        title="Original In-Memory FP32",
        timestamp_utc="2026-08-15T22:00:00Z",
        model=base_model,
        environment=base_env,
        specification=base_spec,
        provenance_chain=prov,
        measurements=meas,
        verdict="VERIFIED",
        runtime_strategy=RuntimeStrategy(
            execution_runtime="in_memory",
            max_active_layers=12,
            quantization="fp32",
        ),
        execution_telemetry=ExecutionTelemetry(peak_ram_mb=450.0),
    )

    # Run 2: Exact Strict Reproduction (Same Runtime Strategy)
    run2 = ImmutableExperimentRun(
        run_id="run_002",
        parent_run_id="run_001",
        experiment_type="REPRODUCTION",
        title="Direct Reproduction",
        timestamp_utc="2026-08-15T22:05:00Z",
        model=base_model,
        environment=base_env,
        specification=base_spec,
        provenance_chain=prov,
        measurements=meas,
        verdict="VERIFIED",
        runtime_strategy=RuntimeStrategy(
            execution_runtime="in_memory",
            max_active_layers=12,
            quantization="fp32",
        ),
        execution_telemetry=ExecutionTelemetry(peak_ram_mb=450.0),
    )

    report_repro = ReproductionToleranceEngine.compare_runs(run1, run2)
    assert report_repro.overall_reproduced is True
    assert report_repro.strategy_matched is True
    assert report_repro.reproduction_category == "STRICT_IDENTICAL_RUNTIME"
    assert "IDENTICAL RUNTIME" in report_repro.numerical_tolerance_verdict

    # Run 3: Cross-Strategy Replication (Out-of-Core INT8)
    run3 = ImmutableExperimentRun(
        run_id="run_003",
        parent_run_id="run_001",
        experiment_type="REPLICATION",
        title="Cross-Strategy Replication Out-of-Core INT8",
        timestamp_utc="2026-08-15T22:10:00Z",
        model=base_model,
        environment=base_env,
        specification=base_spec,
        provenance_chain=prov,
        measurements=meas,
        verdict="VERIFIED",
        runtime_strategy=RuntimeStrategy(
            execution_runtime="out_of_core",
            max_active_layers=1,
            quantization="int8",
        ),
        execution_telemetry=ExecutionTelemetry(peak_ram_mb=120.0),
    )

    report_repl = ReproductionToleranceEngine.compare_runs(run1, run3)
    assert report_repl.overall_reproduced is True
    assert report_repl.strategy_matched is False
    assert report_repl.reproduction_category == "CROSS_STRATEGY_REPLICATION"
    assert "CROSS-STRATEGY" in report_repl.numerical_tolerance_verdict
    assert report_repl.strategy_divergence_summary is not None
    assert report_repl.strategy_divergence_summary["original_runtime"] == "in_memory"
    assert report_repl.strategy_divergence_summary["reproduction_runtime"] == "out_of_core"

