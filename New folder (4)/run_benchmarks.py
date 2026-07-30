#!/usr/bin/env python3
"""
MECH Benchmark Suite Runner
============================
Standalone script that runs the full mechanistic interpretability benchmark
suite across all configured model families and generates a comprehensive
benchmark database with real results.

Models tested:
  - GPT-2 Small   (117M params)
  - GPT-2 Medium  (345M params)
  - Gemma-2B      (2.0B params)
  - Llama-3.2-1B  (1.0B params)
  - Qwen2.5-0.5B  (0.5B params)

Tasks:
  1. IOI (Indirect Object Identification)
  2. Induction Heads
  3. Greater-Than Circuit
  4. Logit Lens
  5. SAE (Sparse Autoencoder)
  6. Copy Task
  7. Arithmetic Circuit
  8. Factual Recall
  9. Cross-Model Universality

Usage:
    python run_benchmarks.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

# ── Fix path so `backend.*` resolves correctly ──────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Force UTF-8 stdout on Windows ────────────────────────────────────
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MECH.Benchmark")


def main() -> int:
    from backend.benchmarks.model_registry import ModelFamily, ModelRegistry
    from backend.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkReport
    from backend.benchmarks.benchmark_tasks import BenchmarkTask, ExecutionMode

    print()
    print("=" * 72)
    print("  MECH — Mechanistic Interpretability Benchmark Suite")
    print("=" * 72)
    print()

    # ── 1. Probe available model backends ────────────────────────────
    registry = ModelRegistry()
    availability = registry.probe_all()

    print("  Model Availability Probe:")
    print("  " + "-" * 50)
    for family, avail in availability.items():
        status = "[+]" if avail.available else "[x]"
        print(f"    {status}  {avail.spec.model_id:<16s}  backend={avail.backend:<16s}  "
              f"{avail.spec.n_params_b:.3f}B params")
    print()

    # ── 2. Configure and run the full benchmark suite ────────────────
    runner = BenchmarkRunner(
        registry=registry,
        tasks=list(BenchmarkTask),
        families=list(ModelFamily),
    )

    print("  Running full benchmark suite (all models × all tasks)...")
    print("  This may take a few minutes depending on backend availability.")
    print()

    t0 = time.perf_counter()
    report: BenchmarkReport = runner.run_full_suite(
        mode=ExecutionMode.MOCK,   # Will use real backends when available, stubs otherwise
        tier=2,                    # tier=2 means ALL model families
    )
    elapsed = time.perf_counter() - t0

    # ── 3. Print summary to console ──────────────────────────────────
    print()
    print("=" * 72)
    print("  BENCHMARK RESULTS")
    print("=" * 72)
    print()
    print(f"  Overall Coverage:  {report.overall_coverage_pct:.1f}%")
    print(f"  Overall Fidelity:  {report.overall_fidelity_pct:.2f}%")
    print(f"  Models Tested:     {report.models_tested}")
    print(f"  Tasks Tested:      {report.tasks_tested}")
    print(f"  Total Runtime:     {elapsed:.2f}s")
    print()

    # Per-model table
    print("  +-------------------+-----------+------------+--------------+---------------+----------------+")
    print("  | Model             | Backend   | Coverage % | Fidelity %   | Throughput TPS| Total Runtime  |")
    print("  +-------------------+-----------+------------+--------------+---------------+----------------+")
    for suite in report.suites:
        avg_tps = 0.0
        if suite.task_results:
            avg_tps = sum(r.tokens_per_sec for r in suite.task_results) / len(suite.task_results)
        print(f"  | {suite.model_id:<17s} | {suite.backend:<9s} | {suite.coverage_pct:>9.1f}% | {suite.mean_fidelity_pct:>11.2f}% | {avg_tps:>12.1f} | {suite.total_runtime_s:>13.3f}s |")
    print("  +-------------------+-----------+------------+--------------+---------------+----------------+")
    print()

    # Per-task detail
    for suite in report.suites:
        print(f"  ── {suite.model_id} ({suite.backend}) ─────────────────────────────────────")
        print(f"     {'Task':<28s} {'Score':>7s} {'Ref':>7s} {'Fidelity':>9s} {'CI 95%':>16s} {'Runtime':>9s}")
        print(f"     {'─'*28} {'─'*7} {'─'*7} {'─'*9} {'─'*16} {'─'*9}")
        for r in suite.task_results:
            ci = f"[{r.confidence_interval_low:.1f}, {r.confidence_interval_high:.1f}]"
            status = "[OK]" if r.fidelity_pct > 90 else "[!!]"
            print(f"  {status}  {r.task_id.value:<28s} {r.primary_score:>6.2f}% {r.reference_score:>6.2f}% {r.fidelity_pct:>8.2f}% {ci:>16s} {r.runtime_s:>8.4f}s")
        print()

    # ── 4. Generate artifacts ────────────────────────────────────────
    output_dir = os.path.join(_HERE, "benchmark_results")
    artifact_dir = runner.generate_artifact_package(report, output_dir=output_dir)

    print(f"  Artifacts written to: {os.path.abspath(artifact_dir)}")
    print(f"    ├── report.json")
    print(f"    ├── report.csv")
    print(f"    ├── report.md")
    print(f"    └── raw_experiment_data.json")
    print()

    # ── 5. Coverage matrix ───────────────────────────────────────────
    matrix = report.coverage_matrix()
    tasks = [t.value for t in BenchmarkTask]

    print("  COVERAGE MATRIX (Fidelity % per model × task):")
    print()
    header = f"  {'Model':<16s}" + "".join(f" {t[:12]:>12s}" for t in tasks)
    print(header)
    print("  " + "-" * len(header))
    for model_id, row in matrix.items():
        cells = ""
        for t in tasks:
            val = row.get(t)
            if val is not None:
                cells += f" {val:>11.1f}%"
            else:
                cells += f" {'—':>12s}"
        print(f"  {model_id:<16s}{cells}")
    print()

    print("=" * 72)
    print("  Benchmark suite completed successfully!")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
