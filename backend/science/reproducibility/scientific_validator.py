"""Scientific Validator Engine — Research-Grade Reproducibility & Governance.

Implements the Four-Tier Comparison:
1. Published Literature (Ground Truth)
2. Canonical Registry (Internal Standard)
3. Reference Baseline (Local Stable High-Water Mark)
4. Current Run (New Experiment)

Includes Statistical Rigor (Cohen's d, Power), Drift Detection, and Reproducibility Scoring.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from backend.science.reproducibility.benchmark_reference_registry import CanonicalBenchmarkRegistry, BenchmarkReference
from backend.science.reproducibility.canonical_circuit_registry import CanonicalCircuitRegistry
from backend.science.models.adapter_base import ModelAdapter

# New Governance Imports
from backend.science.reproducibility.research_snapshot import ResearchSnapshotEngine
from backend.science.reproducibility.research_manifest import ResearchManifestEngine
from backend.science.reproducibility.audit_logger import ScientificAuditLogger
from backend.science.reproducibility.benchmark_certificate import CertificateEngine
from backend.science.reproducibility.model_fingerprint import ModelFingerprintEngine


@dataclass
class ValidationStats:
    observed_mean: float
    observed_median: float
    observed_std: float
    observed_iqr: float
    n: int
    bootstrap_ci: Tuple[float, float]
    difference_pct: float
    cohens_d: float
    verdict: str
    criteria_met: Dict[str, bool]


class ScientificValidator:
    """Validator for high-fidelity mechanistic interpretability research."""

    def __init__(self, registry: Optional[CanonicalBenchmarkRegistry] = None) -> None:
        self.registry = registry or CanonicalBenchmarkRegistry()
        self.audit = ScientificAuditLogger()
        self.snapshot_engine = ResearchSnapshotEngine()
        self.manifest_engine = ResearchManifestEngine()
        self.cert_engine = CertificateEngine()
        self.model_engine = ModelFingerprintEngine()

    # ── Statistical Rigor ──────────────────────────────────────────────────

    def compute_robust_stats(self, data: List[float]) -> Tuple[float, float]:
        if not data: return (0.0, 0.0)
        sorted_data = sorted(data)
        n = len(sorted_data)
        median = sorted_data[n // 2] if n % 2 == 1 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0
        q1, q3 = sorted_data[int(n * 0.25)], sorted_data[int(n * 0.75)]
        return (median, q3 - q1)

    def compute_bootstrap_ci(self, data: List[float], n_bootstrap: int = 2000, ci: float = 0.95) -> Tuple[float, float]:
        if not data: return (0.0, 0.0)
        means = [sum(random.choices(data, k=len(data))) / len(data) for _ in range(n_bootstrap)]
        means.sort()
        return (means[int((1 - ci) / 2 * n_bootstrap)], means[int((1 + ci) / 2 * n_bootstrap)])

    def compute_cohens_d(self, observed: List[float], reference_val: float) -> float:
        if not observed or len(observed) < 2: return 0.0
        mean_obs = sum(observed) / len(observed)
        std_dev = math.sqrt(sum((x - mean_obs) ** 2 for x in observed) / (len(observed) - 1))
        return (mean_obs - reference_val) / std_dev if std_dev > 0 else 0.0

    def calculate_reproducibility_score(self, stats: ValidationStats, parity: float, env_match: bool, ds_integrity: float = 1.0) -> float:
        """Computes 0-100 score based on Data, Env, and Statistical rigor."""
        # Components (Weighted)
        # 40% Data Integrity
        # 30% Environment Match
        # 30% Statistical Rigor

        data_score = ds_integrity * 40.0
        env_score = 30.0 if env_match else 15.0
        stat_score = 30.0 if stats.verdict == "PASS" else 10.0

        # Penalties
        penalty = 0.0
        if parity < 0.999: penalty += (1.0 - parity) * 500.0
        if stats.n < 40: penalty += 10.0

        final_score = data_score + env_score + stat_score - penalty
        return max(0.0, min(100.0, final_score))

    # ── Validation Core ────────────────────────────────────────────────────

    def validate_benchmark(
        self,
        metric_id: str,
        observed_data: List[float],
        patch_success_rate: float = 100.0,
        min_n: int = 40,
        tolerance_pct: float = 2.0,
        reference_baseline: Optional[float] = None
    ) -> ValidationStats:
        """Multi-tier validation against Published and Reference baselines."""
        self.audit.log("Validator", "validate_benchmark", "INFO", f"Starting validation for {metric_id}")

        ref = self.registry.get_primary_reference(metric_id)
        if not ref:
            self.audit.log("Validator", "validate_benchmark", "FAIL", f"No reference for {metric_id}")
            raise ValueError(f"Reference for {metric_id} not found.")

        n = len(observed_data)
        mean_obs = sum(observed_data) / n if n > 0 else 0.0
        std_obs = math.sqrt(sum((x - mean_obs) ** 2 for x in observed_data) / (n - 1)) if n > 1 else 0.0
        median_obs, iqr_obs = self.compute_robust_stats(observed_data)
        ci = self.compute_bootstrap_ci(observed_data)

        # Compare against Published (Wang et al.)
        diff_pct = abs(mean_obs - ref.expected_mean) / ref.expected_mean * 100
        d = self.compute_cohens_d(observed_data, ref.expected_mean)

        criteria = {
            "published_in_ci": ci[0] <= ref.expected_mean <= ci[1],
            "diff_within_tolerance": diff_pct <= tolerance_pct,
            "rigorous_sample": n >= min_n,
            "patch_success_met": patch_success_rate >= 90.0
        }

        # Regression detection vs Reference Baseline (Local Stable)
        if reference_baseline is not None:
            reg_diff = (reference_baseline - mean_obs) / reference_baseline * 100
            if reg_diff > tolerance_pct:
                self.audit.log("Regression", "detect", "WARN", f"Regression detected: {reg_diff:.2f}% drop from baseline")
                criteria["no_regression"] = False
            else:
                criteria["no_regression"] = True

        verdict = "PASS" if all(criteria.values()) else "REVISION_REQUIRED"
        if diff_pct > 15.0: verdict = "FAIL"

        self.audit.log("Validator", "validate_benchmark", verdict, f"Completed with mean {mean_obs:.4f}")

        return ValidationStats(
            observed_mean=round(mean_obs, 4),
            observed_median=round(median_obs, 4),
            observed_std=round(std_obs, 4),
            observed_iqr=round(iqr_obs, 4),
            n=n,
            bootstrap_ci=(round(ci[0], 4), round(ci[1], 4)),
            difference_pct=round(diff_pct, 2),
            cohens_d=round(d, 3),
            verdict=verdict,
            criteria_met=criteria
        )

    # ── Artifact Generation ────────────────────────────────────────────────

    def generate_validation_artifacts(
        self,
        benchmark_results: List[Dict[str, Any]],
        adapter: ModelAdapter,
        dataset_id: str,
        output_dir: str = "benchmark_report"
    ) -> Dict[str, str]:
        """Generates the full Research Manifest, Certificate, and Reports."""
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        self.audit.log("Artifacts", "generate", "INFO", "Generating research artifacts and manifest")

        # 1. Capture Environment Snapshot
        snapshot = self.snapshot_engine.capture()

        # 2. Capture Model Fingerprint
        fingerprint = self.model_engine.capture(adapter)

        # Fail closed: a certificate binds these hashes to a published claim,
        # so an unattested fingerprint (mock_mode adapter, or weights that
        # could not be hashed) must never reach the certificate or manifest.
        if not fingerprint.attested:
            reason = fingerprint.attestation_reason or "model identity unattested"
            self.audit.log("Artifacts", "fingerprint", "FAIL", reason)
            raise ValueError(
                "Cannot generate validation artifacts: the model fingerprint is "
                f"not attested, so the evidence chain cannot be bound to real "
                f"weights. Reason: {reason}"
            )

        # 3. Assemble Results for Certificate
        primary_res = benchmark_results[0] if benchmark_results else {}
        stats = primary_res.get("stats")

        # 4. Compute Reproducibility Score
        repro_score = self.calculate_reproducibility_score(
            stats if isinstance(stats, ValidationStats) else ValidationStats(0,0,0,0,0,(0,0),0,0,"FAIL",{}),
            primary_res.get("parity", 1.0),
            snapshot.git_dirty == False
        )

        # 5. Generate Certificate
        cert = self.cert_engine.generate(
            benchmark_id=primary_res.get("id", "unknown"),
            results={
                "published": 0.88, # Mocked lookup
                "baseline": primary_res.get("baseline"),
                "current": stats.observed_mean if hasattr(stats, 'observed_mean') else 0.0
            },
            hashes={
                "dataset": dataset_id,
                "model": fingerprint.weights_sha256,
                "env": snapshot.snapshot_id
            },
            repro_score=repro_score,
            verdict=stats.verdict == "PASS" if hasattr(stats, 'verdict') else False
        )

        # 6. Assemble Research Manifest (Merkle Root)
        manifest = self.manifest_engine.generate(
            experiment_id=f"EXP-{cert.certificate_id}",
            snapshot=snapshot,
            certificate=asdict(cert),
            audit_log=self.audit.get_entries(),
            reproducibility_score=repro_score
        )

        # 6b. Digital Signature
        manifest.signature = self.manifest_engine.sign_manifest(manifest)

        # Save Files
        paths = {
            "manifest": os.path.join(output_dir, "research_manifest.json"),
            "snapshot": os.path.join(output_dir, "research_snapshot.json"),
            "certificate": os.path.join(output_dir, "benchmark_certificate.json"),
            "audit_log": os.path.join(output_dir, "audit_log.json"),
            "md": os.path.join(output_dir, "scientific_validation.md"),
            "bundle": os.path.join(output_dir, "research_bundle.zip")
        }

        with open(paths["manifest"], "w") as f: json.dump(asdict(manifest), f, indent=2)
        with open(paths["snapshot"], "w") as f: json.dump(asdict(snapshot), f, indent=2)
        with open(paths["certificate"], "w") as f: json.dump(asdict(cert), f, indent=2)
        with open(paths["audit_log"], "w") as f: json.dump(self.audit.get_entries(), f, indent=2)

        # 7. Generate Markdown Report (4-Tier)
        self._write_markdown_report(paths["md"], benchmark_results, repro_score, manifest)

        # 8. Create Portable Bundle (ZIP)
        self.create_research_bundle(output_dir, paths)

        return paths

    def create_research_bundle(self, output_dir: str, paths: Dict[str, str]):
        """Bundles all research artifacts into a single portable ZIP file."""
        bundle_path = paths["bundle"]
        with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for key, file_path in paths.items():
                if key != "bundle" and os.path.exists(file_path):
                    zipf.write(file_path, os.path.basename(file_path))

        self.audit.log("Bundle", "create", "PASS", f"Research bundle created at {bundle_path}")

    def _write_markdown_report(self, path: str, results: List[Dict[str, Any]], score: float, manifest: Any):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Scientific Validation Report (v1.5)\n\n")
            f.write(f"## Reproducibility Score: **{score:.1f}/100**\n\n")

            f.write("## 1. Provenance & Integrity\n")
            f.write(f"- **Experiment ID**: `{manifest.experiment_id}`\n")
            f.write(f"- **Manifest Merkle Root**: `{manifest.root_sha256}`\n")
            f.write(f"- **Status**: **{manifest.status}**\n\n")

            f.write("## 2. Four-Tier Comparison\n")
            f.write("| Metric | Published | Registry | Reference | Current | Delta |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for r in results:
                s = r.get("stats")
                pub = r.get("published", 0.0)
                reg = r.get("registry", pub)
                base = r.get("baseline", "N/A")
                curr = s.observed_mean if hasattr(s, 'observed_mean') else 0.0
                delta = curr - pub
                f.write(f"| {r.id} | {pub} | {reg} | {base} | **{curr}** | {delta:+.4f} |\n")

            f.write("\n---\n*Report generated by ScientificValidator Engine Phase 39.8.*")
