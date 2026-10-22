"""Run the MECH suite file by file and record the result of each.

One `pytest tests/pytest` session on this host is killed part-way through
by the environment's process watchdog, and a partial run proves nothing:
the source-stability guard attests a session, not a fraction of one. So the
suite is run as a matrix -- one pytest session per test file -- and each
session is a complete session with its own attestation.

The matrix is not a weaker check than one big run. A single run collects
everything in one process, where an import or a global-state change in one
file can affect another; the matrix gives each file its own process, which
is closer to how CI shards. What it does give up is one shared session
scope, so tests that depend on another file having run first would pass
here and fail there. This script reports that class of problem rather than
hiding it: a file that fails only in isolation is still a failure.

Usage:
    python scripts/run_matrix.py --root <repo> --out <results.json>

Exit code 0 when every file passed, skipped or was otherwise accounted for;
1 when any file failed or the run was cut short.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-file-timeout", type=int, default=900)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    suite = root / "tests" / "pytest"
    files = sorted(p for p in suite.glob("test_*.py"))

    env = dict(os.environ)
    env["MECH_REQUIRE_JS_LINT"] = "1"
    env.pop("MECH_SKIP_SOURCE_STABILITY_CHECK", None)

    results = []
    started = time.time()
    for path in files:
        rel = path.relative_to(root).as_posix()
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(path), "-q",
                 "--no-header", "-p", "no:cacheprovider"],
                cwd=str(root), capture_output=True, text=True,
                env=env, timeout=args.per_file_timeout,
            )
            out_text = (proc.stdout or "") + (proc.stderr or "")
            code = proc.returncode
        except subprocess.TimeoutExpired:
            out_text = "TIMEOUT after %ds" % args.per_file_timeout
            code = 124
        elapsed = round(time.time() - t0, 1)
        tail = out_text.strip().splitlines()[-1] if out_text.strip() else ""
        # The source-stability guard prints its verdict as the last line
        # when it reports a run as untrustworthy, and pytest's own summary
        # line otherwise. Either way the last line is the verdict.
        record = {
            "file": rel,
            "returncode": code,
            "seconds": elapsed,
            "summary": tail.strip(),
        }
        results.append(record)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
        print("%-72s rc=%-3s %6.1fs  %s" % (
            rel, code, elapsed, tail.strip()[:110]), flush=True)

    failed = [r for r in results if r["returncode"] not in (0, 5)]
    print("\n%d file(s), %d failed, %.1f minutes total" % (
        len(results), len(failed), (time.time() - started) / 60.0), flush=True)
    for r in failed:
        print("  FAILED %s rc=%s: %s" % (r["file"], r["returncode"],
                                         r["summary"]), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
