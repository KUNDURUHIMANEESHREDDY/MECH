"""Phase 13 — Benchmark Execution Tests.

Tests that verify the benchmark runners produce valid, quantitative comparison tables
with all required fields and dataset provenance hashes bound to manifests.

Four tests:
    13.5  Induction benchmark comparison table is complete (MECH score, TL score, diff, CI, dataset hash)
    13.6  IOI benchmark produces a circuit report with logit diffs, CI, and reference circuit
    13.7  SAE benchmark feature interpretation pipeline is complete across all 4 stages
    13.8  Benchmark dataset hash is bound to each ExperimentManifest
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from backend.science.benchmarks.induction_benchmark import run_induction_benchmark
from backend.science.benchmarks.ioi_benchmark import run_ioi_benchmark
from backend.science.benchmarks.sae_benchmark import run_sae_benchmark
from backend.interpretability.sae.sae_lens_adapter import HAS_SAELENS
from backend.science.provenance import manifest_store


_BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestBenchmarkExecution:

    def test_13_5_induction_benchmark_comparison_table_is_complete(self):
        """Induction benchmark produces: MECH score, TL reference score, abs diff, rel diff, CI, dataset hash."""
        report = run_induction_benchmark(layer=5, head=5, n_bootstrap=200)

        # Required top-level fields
        assert report["benchmark"] == "induction_head"
        assert "manifest_id" in report and report["manifest_id"]
        assert "dataset_sha256" in report and len(report["dataset_sha256"]) == 64
        assert report["n_templates"] >= 8

        # Quantitative fields
        assert not math.isnan(report["mech_mean_score"])
        assert not math.isnan(report["tl_mean_score"])
        assert not math.isnan(report["mean_abs_diff"])
        assert report["bootstrap_ci_lower"] <= report["bootstrap_ci_upper"]
        assert 0.0 <= report["control_95th_percentile"] <= 1.0

        # Per-template rows are complete
        for row in report["template_results"]:
            assert "template_id" in row
            assert "mech_score" in row
            assert "tl_score" in row
            assert "abs_diff" in row
            assert "rel_diff_pct" in row
            assert "ablation_ie" in row
            assert "evidence_level" in row

    def test_13_6_ioi_benchmark_produces_circuit_report(self):
        """IOI benchmark produces: logit diffs, CI, cross-template variance, reference circuit."""
        report = run_ioi_benchmark()

        assert report["benchmark"] == "ioi_circuit"
        assert "manifest_id" in report and report["manifest_id"]
        assert "dataset_sha256" in report and len(report["dataset_sha256"]) == 64
        assert report["n_templates"] >= 6

        # Logit diff measurements
        assert not math.isnan(report["mean_logit_diff_clean_vs_corrupted"])
        assert not math.isnan(report["cross_template_variance"])
        assert report["bootstrap_ci_lower"] <= report["bootstrap_ci_upper"]

        # Reference circuit structure present
        assert "reference_circuit" in report
        assert "name_mover_heads" in report["reference_circuit"]
        assert isinstance(report["reference_circuit_applicable"], bool)

        # Per-template results
        for row in report["template_results"]:
            assert "template_id" in row
            assert "logit_diff_clean_vs_corrupted" in row
            assert not math.isnan(row["logit_diff_clean_vs_corrupted"])

    @pytest.mark.skipif(not HAS_SAELENS, reason="sae_lens not installed")
    def test_13_7_sae_benchmark_feature_interpretation_pipeline(self):
        """SAE benchmark runs all 4 stages and reports raw measurements."""
        report = run_sae_benchmark(feature_idx=42, alpha=2.0, n_bootstrap=200)

        assert report["benchmark"] == "sae_feature"
        assert "manifest_id" in report and report["manifest_id"]
        assert "prompt_dataset_hash" in report and len(report["prompt_dataset_hash"]) > 0

        # Stage 1 — positive concept selectivity
        assert len(report["stage_1_positive_selectivities"]) >= 1
        for s in report["stage_1_positive_selectivities"]:
            assert not math.isnan(s)

        # Stage 2 — orthogonal controls
        assert len(report["stage_2_orthogonal_selectivities"]) >= 1

        # Stage 3 — OOD replication
        assert len(report["stage_3_ood_selectivities"]) >= 1

        # Stage 4 — paraphrase replication
        assert len(report["stage_4_paraphrase_selectivities"]) >= 1

        # Zero-alpha control baseline
        assert abs(report["zero_alpha_baseline"]) < 1.0

        # CI is present and valid
        if report["bootstrap_ci"] is not None:
            ci_lower, ci_upper = report["bootstrap_ci"]
            assert ci_lower <= ci_upper

    def test_13_8_benchmark_dataset_hash_bound_to_manifest(self):
        """The dataset_sha256 field in each benchmark report matches the frozen file hash and is stored in its manifest."""
        # Induction
        induction_report = run_induction_benchmark(layer=5, head=5, n_bootstrap=200)
        clean_sha = _file_sha256(_BENCHMARKS_DIR / "induction" / "clean.jsonl")
        ctrl_sha = _file_sha256(_BENCHMARKS_DIR / "induction" / "scrambled_controls.jsonl")
        expected_dataset_hash = hashlib.sha256((clean_sha + ctrl_sha).encode()).hexdigest()
        assert induction_report["dataset_sha256"] == expected_dataset_hash

        # Verify the hash is inside the manifest on disk
        manifest = manifest_store.retrieve(induction_report["manifest_id"])
        assert manifest["prompt_dataset_hash"] == expected_dataset_hash

        # IOI
        ioi_report = run_ioi_benchmark()
        tmpl_sha = _file_sha256(_BENCHMARKS_DIR / "ioi" / "templates.jsonl")
        corr_sha = _file_sha256(_BENCHMARKS_DIR / "ioi" / "corruptions.jsonl")
        expected_ioi_hash = hashlib.sha256((tmpl_sha + corr_sha).encode()).hexdigest()
        assert ioi_report["dataset_sha256"] == expected_ioi_hash

        ioi_manifest = manifest_store.retrieve(ioi_report["manifest_id"])
        assert ioi_manifest["prompt_dataset_hash"] == expected_ioi_hash
