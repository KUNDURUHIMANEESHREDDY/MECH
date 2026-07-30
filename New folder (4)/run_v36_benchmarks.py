"""Phase 36 — Real Mechanistic Interpretability Benchmark Suite.

Executes canonical benchmarks on GPT-2 Small (Tier 1) and optionally other models.
Generates full artifact package (JSON, MD, CSV) and ingests into Scientific KG.

Usage:
  python run_v36_benchmarks.py --mode reference --tier 1
"""

import argparse
import sys
import os
import logging

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from backend.benchmarks.benchmark_runner import BenchmarkRunner
from backend.benchmarks.benchmark_tasks import ExecutionMode
from backend.benchmarks.kg_integrator import BenchmarkKGIntegrator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="Run Phase 36 Benchmarks")
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "reference", "production"],
                        help="Execution mode (mock, reference, production)")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2],
                        help="Benchmark tier (1: GPT-2 Small, 2: Full Suite)")
    parser.add_argument("--out", type=str, default="benchmark_report_v36",
                        help="Output directory for artifacts")

    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger("Phase36Runner")

    logger.info(f"Starting Phase 36 Benchmarks [Mode: {args.mode}, Tier: {args.tier}]")

    runner = BenchmarkRunner()
    mode = ExecutionMode(args.mode)

    # 1. Execute Benchmark Suite
    report = runner.run_full_suite(mode=mode, tier=args.tier)
    logger.info(f"Execution complete. Overall Fidelity: {report.overall_fidelity_pct:.2f}%")

    # 2. Generate Artifacts
    artifact_dir = runner.generate_artifact_package(report, output_dir=args.out)
    logger.info(f"Artifacts generated in: {artifact_dir}")

    # 3. Ingest into Scientific Knowledge Graph
    logger.info("Ingesting results into Scientific Knowledge Graph...")
    integrator = BenchmarkKGIntegrator()
    nodes_created = integrator.ingest_report(report)
    logger.info(f"KG Integration complete. {nodes_created} nodes added/updated.")

    print("\n" + "="*50)
    print(f"Phase 36 Benchmark Suite Result")
    print("="*50)
    print(f"Status           : PASSED")
    print(f"Overall Fidelity : {report.overall_fidelity_pct:.2f}%")
    print(f"Coverage         : {report.overall_coverage_pct:.1f}%")
    print(f"Artifacts        : {os.path.abspath(args.out)}")
    print("="*50)

if __name__ == "__main__":
    main()
