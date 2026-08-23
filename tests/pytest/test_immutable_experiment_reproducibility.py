"""Unit tests for Phase 11: Immutable Experiment Notebook & Reproducibility Architecture."""

import tempfile
from pathlib import Path
import pytest

from backend.science.reproducibility.immutable_store import ImmutableExperimentStore
from backend.science.reproducibility.reproducibility_service import ReproducibilityService
from backend.science.reproducibility.tolerance_engine import ReproductionToleranceEngine
from backend.science.reproducibility.archive_builder import ArchiveBuilder
from backend.science.reproducibility.types import (
    ExecutionEnvironment,
    ExperimentSpecification,
    ImmutableExperimentRun,
    ModelIdentity,
    ProvenanceChain,
)


@pytest.fixture
def temp_store(tmp_path):
    db_path = tmp_path / "test_experiments.db"
    return ImmutableExperimentStore(db_path=db_path)


def _sample_run(run_id="exp_test1", parent_id=None, exp_type="ORIGINAL"):
    model = ModelIdentity(
        model_id="gpt2",
        architecture="GPT2LMHeadModel",
        parameter_count=124439808,
        weights_hash="e1f98d45b7803b9e4a3b7c2e1f4a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a",
        revision_or_commit="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        tokenizer_hash="f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3",
        config_hash="config_hash_sample",
    )
    env = ExecutionEnvironment(
        python_version="3.11.0",
        pytorch_version="2.3.0",
        transformers_version="4.40.0",
        cuda_version=None,
        gpu_name=None,
        gpu_compute_capability=None,
        os_platform="Windows",
        os_release="10",
        mech_version="2.0.0",
        git_commit_sha="c6f9b28a",
        dependency_lock_hash="lock_abc123",
        deterministic_mode=True,
    )
    spec = ExperimentSpecification(
        clean_prompt="The capital of France is",
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
        "clean_logit": 14.82,
        "intervened_logit": 13.75,
        "delta_logit": 1.07,
        "clean_probability": 0.84,
        "intervened_probability": 0.32,
        "delta_probability": 0.52,
        "clean_rank": 0,
        "intervened_rank": 2,
    }
    return ImmutableExperimentRun(
        run_id=run_id,
        parent_run_id=parent_id,
        experiment_type=exp_type,
        title="Sample Experiment",
        timestamp_utc="2026-08-15T19:00:00Z",
        model=model,
        environment=env,
        specification=spec,
        provenance_chain=prov,
        measurements=meas,
        verdict="Verified causal mediation",
    )


def test_immutable_store_save_and_retrieve(temp_store):
    run = _sample_run("exp_101")
    saved = temp_store.save_run(run)

    assert saved.run_id == "exp_101"
    assert len(saved.manifest_sha256) == 64

    retrieved = temp_store.get_run("exp_101")
    assert retrieved is not None
    assert retrieved.title == "Sample Experiment"
    assert retrieved.model.model_id == "gpt2"
    assert retrieved.measurements["clean_logit"] == 14.82

    # Verify cryptographic integrity
    is_valid, msg = temp_store.verify_integrity("exp_101")
    assert is_valid is True
    assert "Integrity verified" in msg


def test_immutable_store_prevents_mutation(temp_store):
    run = _sample_run("exp_immutable")
    temp_store.save_run(run)

    # Attempting to re-save with same run_id must raise ValueError
    with pytest.raises(ValueError, match="already exists and is immutable"):
        temp_store.save_run(run)


def test_lineage_tracking(temp_store):
    run1 = _sample_run("exp_104", parent_id=None, exp_type="ORIGINAL")
    temp_store.save_run(run1)

    run2 = _sample_run("exp_105", parent_id="exp_104", exp_type="REPRODUCTION")
    temp_store.save_run(run2)

    run3 = _sample_run("exp_106", parent_id="exp_105", exp_type="REPLICATION")
    temp_store.save_run(run3)

    lineage = temp_store.get_lineage("exp_105")
    run_ids = [r.run_id for r in lineage]
    assert "exp_104" in run_ids
    assert "exp_105" in run_ids
    assert "exp_106" in run_ids


def test_tolerance_engine_exact_match():
    orig = _sample_run("exp_orig")
    repro = _sample_run("exp_repro", parent_id="exp_orig", exp_type="REPRODUCTION")

    report = ReproductionToleranceEngine.compare_runs(orig, repro, duration_ms=12.5)
    assert report.overall_reproduced is True
    assert "REPRODUCED WITHIN NUMERICAL TOLERANCE" in report.numerical_tolerance_verdict
    assert report.model_matched is True


def test_tolerance_engine_detects_mismatch():
    orig = _sample_run("exp_orig")
    repro = _sample_run("exp_repro", parent_id="exp_orig", exp_type="REPRODUCTION")
    # Simulate a major deviation in delta logit
    repro_meas = dict(repro.measurements)
    repro_meas["delta_logit"] = 0.05  # Was 1.07
    object.__setattr__(repro, "measurements", repro_meas)

    report = ReproductionToleranceEngine.compare_runs(orig, repro)
    assert report.overall_reproduced is False
    assert "REPRODUCTION DIVERGENCE" in report.numerical_tolerance_verdict


def test_archive_builder_manifest():
    run = _sample_run("exp_arch")
    archive_files = ArchiveBuilder.build_archive_dict(run)

    assert "manifest.json" in archive_files
    assert "reproduce.py" in archive_files
    assert "experiment.json" in archive_files
    assert "README.md" in archive_files

    # Verify standalone reproduce.py has essential components
    script = archive_files["reproduce.py"]
    assert "AutoModelForCausalLM" in script
    assert "clean_target_logit" in script
    assert "register_forward_hook" in script

    # Verify ZIP generation
    zip_bytes = ArchiveBuilder.build_zip_bytes(run)
    assert len(zip_bytes) > 0
    assert zip_bytes.startswith(b"PK")


def test_reproducibility_service_lifecycle(temp_store):
    service = ReproducibilityService(store=temp_store)

    run = service.record_experiment(
        specification=ExperimentSpecification(clean_prompt="The capital of France is", target_token=" Paris"),
        provenance=ProvenanceChain(
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
        ),
        measurements={"clean_logit": 14.82, "intervened_logit": 13.75, "delta_logit": 1.07},
        title="Service Test Run",
        verdict="Verified",
    )

    assert run.run_id.startswith("exp_")
    assert run.manifest_sha256 != ""

    # Replicate on new prompt
    repl_run = service.replicate_experiment(
        original_run_id=run.run_id,
        new_prompt="The Eiffel Tower is in",
        new_target_token=" Paris",
    )
    assert repl_run.experiment_type == "REPLICATION"
    assert repl_run.parent_run_id == run.run_id


def test_canonical_spec_hash_content_addressing(temp_store):
    run1 = _sample_run("exp_spec_1")
    saved1 = temp_store.save_run(run1)

    spec_hash = saved1.canonical_spec_hash
    assert len(spec_hash) == 64

    # Query by canonical spec hash
    matches = temp_store.find_by_spec_hash(spec_hash)
    assert len(matches) == 1
    assert matches[0].run_id == "exp_spec_1"


def test_intervention_implementation_identity():
    run = _sample_run("exp_hook_id")
    impl = run.specification.intervention_impl

    assert impl.type == "zero_ablation"
    assert "mlp" in impl.target_module
    assert impl.hook_impl_version == "v1.0.0_torch_native"
    assert len(impl.source_code_hash) == 64
