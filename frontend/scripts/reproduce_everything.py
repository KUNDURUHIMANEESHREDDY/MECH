"""Run the full benchmark suite and write an artifact package.

This script previously could not run at all. It opened with:

    from backend.benchmarks.benchmark_runner import BenchmarkRunner, ExecutionMode

`backend/benchmarks/` does not exist. The package is `backend/benchmarking/`, and
every method this script calls -- `run_full_suite`, `generate_artifact_package` --
and every field it prints -- `tasks_tested`, `overall_coverage_pct`,
`overall_fidelity_pct` -- exists there. So this was a rename whose imports were
never updated, the same class of debt as `backend/datasets/` ->
`backend/research_datasets/`.

The import was at module scope, so unlike the orchestrator deleted in 50c096a
this one failed loudly and immediately. Nothing depended on it working.

What changed beyond the import
------------------------------
`--mode` defaulted to `mock`. A script called "one-click reproduction" that
defaults to fixtures and then prints a completion banner is the fabrication
pattern this project exists to prevent: `run_full_suite` also defaults to
`ExecutionMode.MOCK`, so both layers had to agree on fixtures before anything ran.
The default is now `reference` -- real GPT-2 weights, which is what this
repository has actually verified.

The completion banner is no longer unconditional. In mock mode nothing is
measured, so the script says so and exits non-zero rather than announcing
success. The summary separates measured results from fixtures, because
`BenchmarkReport` does not: `overall_fidelity_pct` is a mean over whatever ran,
and in mock mode that mean is over hand-picked numbers.

Usage
-----
    python frontend/scripts/reproduce_everything.py
    python frontend/scripts/reproduce_everything.py --mode reference --tier 1
    python frontend/scripts/reproduce_everything.py --mock        # prints NOT CITABLE

Exit status is 0 when at least one result was measured, 1 when none was.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

# Make the repository root importable regardless of the caller's cwd. Was
# `sys.path.insert(0, os.getcwd())`, which only works if you happen to already be
# standing in the repository root -- and silently does nothing useful otherwise.
_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.benchmarking.benchmark_runner import BenchmarkRunner  # noqa: E402
from backend.benchmarking.benchmark_tasks import ExecutionMode  # noqa: E402

#: The modes that produce a measurement. Anything else is a fixture.
MEASURING_MODES = frozenset({
    ExecutionMode.REFERENCE.value,
    ExecutionMode.PRODUCTION.value,
})


def _is_fixture(result: Any) -> bool:
    """Whether a result came from the fixture path rather than a measurement.

    Checks `is_fixture` and the mode, because the historical records in the
    runtime graph store carry `mode: "mock"` and predate the `is_fixture` field.
    """
    if getattr(result, "is_fixture", False):
        return True
    mode = getattr(result, "mode", None)
    value = getattr(mode, "value", mode)
    return str(value) not in MEASURING_MODES


def _partition(report: Any) -> tuple[List[Any], List[Any]]:
    """Split every result in a report into (measured, fixture)."""
    measured: List[Any] = []
    fixtures: List[Any] = []
    for suite in report.suites:
        for result in suite.task_results:
            (fixtures if _is_fixture(result) else measured).append(result)
    return measured, fixtures


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full benchmark suite and write an artifact package.",
    )
    parser.add_argument(
        "--mode", default=ExecutionMode.REFERENCE.value,
        choices=[m.value for m in ExecutionMode],
        help="execution mode; default is 'reference' (real GPT-2 weights), "
             "not 'mock'. The previous default was 'mock'.")
    parser.add_argument(
        "--tier", type=int, default=1,
        help="1 runs GPT-2 small only; higher tiers attempt more model families, "
             "most of which cannot be served from the local cache")
    parser.add_argument(
        "--output", default="benchmark_report",
        help="directory for the artifact package")
    args = parser.parse_args(argv)

    mode = ExecutionMode(args.mode)
    is_fixture_mode = mode == ExecutionMode.MOCK

    print("=" * 60)
    print("MECH BENCHMARK SUITE")
    print("=" * 60)
    print(f"  mode    : {args.mode}")
    print(f"  tier    : {args.tier}")
    print(f"  output  : {args.output}")
    if is_fixture_mode:
        print()
        print("  MOCK MODE. Every pipeline below runs against fixtures. Nothing")
        print("  is measured. These numbers must never be cited or published.")
    print()

    print("[1/3] Running suite (mode: {0})...".format(args.mode))
    runner = BenchmarkRunner()
    started = time.time()
    report = runner.run_full_suite(mode=mode, tier=args.tier)
    duration = time.time() - started

    measured, fixtures = _partition(report)

    print("\n[2/3] Writing artifact package...")
    package_dir = runner.generate_artifact_package(report, output_dir=args.output)
    for name in ("report.md", "report.json", "report.csv",
                 "raw_experiment_data.json"):
        print(f"  - {os.path.join(package_dir, name)}")

    print("\n[3/3] Summary:")
    print(f"  - Results total     : {len(measured) + len(fixtures)}")
    print(f"  - Measured          : {len(measured)}")
    print(f"  - Fixtures          : {len(fixtures)}")
    print(f"  - Models in report  : {report.models_tested}")
    print(f"  - Overall coverage  : {report.overall_coverage_pct:.1f}%")
    print(f"  - Runtime           : {duration:.2f}s")

    if fixtures:
        print()
        print("  NOTE: the aggregates above are means over ALL results, fixtures")
        print("  included. They describe this run, not a measurement. The")
        print("  per-row verdicts in report.md label each result individually.")

    print()
    print("  Dataset verification is NOT part of this run. It is a separate")
    print("  command -- `python frontend/scripts/verify_golden_datasets.py`.")
    print("  This script used to claim step 1 of 4 was 'Verifying Golden")
    print("  Datasets' while calling DatasetManager.list_datasets(), a method")
    print("  that does not exist. The AttributeError was swallowed by a bare")
    print("  except that printed 'Dataset Warning', so the check never ran and")
    print("  the output read as though it had.")

    print()
    print("=" * 60)
    if measured:
        print(f"{len(measured)} result(s) were measured.")
        print("Artifact package written. The per-row verdicts in report.md say")
        print("which results reproduced a published figure and which are fixtures.")
    elif is_fixture_mode:
        print("FIXTURE RUN COMPLETE -- NOTHING WAS MEASURED.")
        print("Do not cite anything in the artifact package.")
    else:
        print("NO RESULTS WERE MEASURED.")
        print("The suite ran but produced no measured result. Treat the artifact")
        print("package as a record of that, not as evidence.")
    print("=" * 60)

    return 0 if measured else 1


if __name__ == "__main__":
    sys.exit(main())
