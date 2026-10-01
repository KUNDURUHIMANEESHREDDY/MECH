"""One-Click Reproduction Suite — Reproduce everything.

Landmark Benchmarks:
- IOI
- Induction Heads
- Greater Than
- Logit Lens
- SAE
- Copy Task
- Arithmetic
- Factual Recall
- Universality

Usage:
  python scripts/reproduce_everything.py --mode production
"""

import sys
import os
import argparse
import time

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from backend.benchmarks.benchmark_runner import BenchmarkRunner, ExecutionMode
from backend.research_datasets.dataset_manager import DatasetManager

def main():
    parser = argparse.ArgumentParser(description="One-Click Reproduction Suite")
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "reference", "production"])
    parser.add_argument("--tier", type=int, default=1)
    parser.add_argument("--output", type=str, default="reproduction_report")
    args = parser.parse_args()

    print("="*60)
    print("MECH ONE-CLICK REPRODUCTION SUITE")
    print("="*60)

    # 1. Dataset Verification
    print("\n[1/4] Verifying Golden Datasets...")
    manager = DatasetManager(data_dir="backend/research_datasets")
    try:
        datasets = manager.list_datasets()
        print(f"  - Found {len(datasets)} validated datasets.")
    except Exception as e:
        print(f"  - Dataset Warning: {e}")

    # 2. Suite Execution
    print(f"\n[2/4] Executing Canonical Benchmarks (Mode: {args.mode})...")
    runner = BenchmarkRunner()

    t0 = time.time()
    report = runner.run_full_suite(
        mode=ExecutionMode(args.mode),
        tier=args.tier
    )
    duration = time.time() - t0

    # 3. Artifact Generation
    print("\n[3/4] Generating Research Artifacts...")
    package_dir = runner.generate_artifact_package(report, output_dir=args.output)
    print(f"  - Master Report: {os.path.join(package_dir, 'report.md')}")
    print(f"  - Raw Data: {os.path.join(package_dir, 'raw_experiment_data.json')}")

    # 4. Final Summary
    print("\n[4/4] Summary:")
    print(f"  - Tasks Tested: {report.tasks_tested}")
    print(f"  - Overall Coverage: {report.overall_coverage_pct:.1f}%")
    print(f"  - Overall Fidelity: {report.overall_fidelity_pct:.2f}%")
    print(f"  - Total Runtime: {duration:.2f}s")

    print("\n" + "="*60)
    print("REPRODUCTION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
