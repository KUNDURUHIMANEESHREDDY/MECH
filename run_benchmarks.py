"""Run the interpretability benchmark suite against real weights.

This replaces a campaign runner that could not run. The previous
`run_benchmarks.py` called `backend.science.benchmarking_orchestrator`, which
imported four classes from `backend.reproductions` -- a package that does not
exist, and none of whose classes exist anywhere in the repository. The failure
was deferred to call time, so `import run_benchmarks` succeeded and only running
it raised `ModuleNotFoundError`.

That orchestrator was deleted rather than rewired. Its `_record_result` appended
rows into `backend/science/benchmark_database.json` -- a file whose every
existing record is stamped `"measured": false, "provenance": "reference",
"publication_eligible": false` with a notice saying the numbers are
hand-typed literals. The rows it appended carried none of those markers, so a
crashed or empty run would have left entries in a reference file that nothing
could distinguish from fixtures. It also iterated
`["gpt2-small", "gpt2-medium", "gemma-2b", "llama-3-8b", "qwen-7b"]`, five of
which cannot be served here.

What this script does
---------------------
Runs the six real pipelines in `backend.science.reproducibility` via
`BenchmarkRunner`, which distinguishes `NOT_RUN` ("no implementation, or the
model does not perform this task") from `ERROR` ("the measurement broke"), and
carries per-sample traces forward so a statistical validator has something to
work with.

What this script does not do
----------------------------
It does not write `benchmark_database.json`. That file is a labelled fixture and
this script leaves it alone. It does not iterate a list of models hoping some
work; it runs exactly the one model you name, and refuses if those weights are
not present rather than quietly serving a different one.

Usage
-----
    python run_benchmarks.py                     # gpt2, the verified stack
    python run_benchmarks.py --model gpt2 --seed 7
    python run_benchmarks.py --mock              # fixtures; prints NOT CITABLE
    python run_benchmarks.py --json out.json     # machine-readable summary

Exit status is 0 when at least one pipeline measured something, 1 when none did.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

#: The combination `requirements.txt` documents as verified.
DEFAULT_MODEL = "gpt2"
DEFAULT_SEED = 42

BANNER = "=" * 72
WARNING = (
    "MOCK MODE. Every pipeline below is running against fixtures. Nothing is\n"
    "  measured. These numbers must never be cited, published, or written into\n"
    "  any record that claims provenance."
)


def _weights_available_locally(model_id: str) -> tuple[bool, str]:
    """Whether `model_id`'s weights are already in the local HF cache.

    Checked rather than assumed, and checked *before* the run. The failure this
    prevents is concrete: `gpt2_engine` loads exactly one checkpoint and echoes
    the caller's `model_name` into its response, so a request for an absent model
    used to produce a live-looking result attributed to a model that never ran.
    A refusal here is the same discipline one layer up.

    Uses `AutoConfig.from_pretrained(..., local_files_only=True)` because a
    config is cheap to read and its presence means the snapshot is cached. No
    download is attempted: silently fetching several gigabytes because a name was
    mistyped is not what this flag should do.
    """
    try:
        from transformers import AutoConfig
    except Exception as exc:  # pragma: no cover - transformers is a hard dep
        return False, f"transformers is not importable: {exc}"

    try:
        AutoConfig.from_pretrained(model_id, local_files_only=True)
    except Exception as exc:
        return False, (
            f"no local weights for {model_id!r} ({type(exc).__name__}). "
            f"Fetch them into the cache first -- `huggingface-cli download "
            f"{model_id}` -- or pass a model you already have "
            f"(the verified stack is gpt2)."
        )
    return True, ""


def _summarise(payload: Dict[str, Any]) -> tuple[Dict[str, int], List[str]]:
    """Count statuses and list the benchmarks that measured nothing.

    `NOT_RUN` and `ERROR` are counted apart on purpose. `NOT_RUN` means the
    benchmark has no implementation or the model cannot perform the task, which
    is a statement about scope. `ERROR` means something that was attempted
    broke, which is a statement about the code. Collapsing them -- as this
    project's report renderer used to, by rendering everything above 90%
    fidelity as "PASS" -- loses the distinction a reader needs.
    """
    counts: Dict[str, int] = {}
    unmeasured: List[str] = []
    for paper_id, entry in (payload.get("reports") or {}).items():
        status = str((entry or {}).get("status") or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
        if status not in {"PASS", "MEASURED"}:
            unmeasured.append(f"{paper_id} ({status})")
    return counts, unmeasured


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the interpretability benchmark suite against real weights.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace id of the model to run (default: {DEFAULT_MODEL}, "
             f"the only one this repository has verified)")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"seed passed to each pipeline (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--mock", action="store_true",
        help="run pipelines in fixture mode; nothing is measured")
    parser.add_argument(
        "--json", metavar="PATH", default=None,
        help="also write the runner's summary dict to PATH")
    args = parser.parse_args(argv)

    print(BANNER)
    print("MECH benchmark suite")
    print(BANNER)
    print(f"  model      : {args.model}")
    print(f"  seed       : {args.seed}")
    print(f"  mock_mode  : {args.mock}")
    print()

    if args.mock:
        print(WARNING)
        print()
    else:
        available, why = _weights_available_locally(args.model)
        if not available:
            # Refuse rather than substitute. Returning 1 here means a mistyped
            # --model cannot end up producing a report attributed to weights that
            # were never loaded.
            print(f"REFUSING TO RUN: {why}")
            print(BANNER)
            return 1
        print(f"  weights    : found in local cache for {args.model!r}")
        print()

    from backend.science.reproducibility.benchmark_runner import BenchmarkRunner

    # mock_mode defaults to False in BenchmarkRunner. It is passed explicitly
    # here so the value the banner printed and the value the pipelines were built
    # with cannot drift apart.
    runner = BenchmarkRunner(mock_mode=args.mock)
    payload = runner.run_all(model_id=args.model, seed=args.seed)

    counts, unmeasured = _summarise(payload)

    print()
    print(BANNER)
    print("Summary")
    print(BANNER)
    for status in sorted(counts):
        print(f"  {status:<10} {counts[status]}")
    print()
    for item in unmeasured:
        print(f"  not measured: {item}")
    if unmeasured:
        print()
        print("  These did not produce a measurement. They are reported as")
        print("  NOT_RUN or ERROR rather than omitted, so an absent number is")
        print("  visibly absent instead of quietly absent.")

    if args.mock:
        print()
        print("  REMINDER: mock mode. Nothing above was measured.")

    measured = counts.get("PASS", 0) + counts.get("MEASURED", 0)
    print()
    if measured:
        print(f"  {measured} pipeline(s) produced a measurement.")
        print("  This script does not write benchmark_database.json; that file is a")
        print("  labelled fixture and is left untouched. For signed, citable")
        print("  artifacts use backend/science/reproducibility/run_reproducibility_demo.py")
        print("  or BenchmarkRunner directly.")
    else:
        print("  No pipeline produced a measurement. Nothing was written.")
    print(BANNER)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        print(f"  summary written to {args.json}")

    return 0 if measured else 1


if __name__ == "__main__":
    sys.exit(main())
