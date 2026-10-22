"""Run the MECH suite in chunks and record the result of each chunk.

One `pytest tests/pytest` session on this host is killed part-way through by
the environment's process watchdog, and a partial run proves nothing: the
source-stability guard attests a session, not a fraction of one. So the suite
is run as a sequence of complete pytest sessions, each over a slice of the
test files, and each slice is a whole session with its own attestation.

A chunk is a weaker unit than a single whole-suite session in one respect:
tests that depend on another file having run first would pass here and fail
in one big run. This script reports that class of problem rather than hiding
it -- a file that fails in a chunk is still a failure.

Usage:
    python scripts/run_chunks.py --root <repo> --out <results.json> [--chunk 30]
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
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--only", default="",
                    help="comma-separated file names to run instead of the whole suite")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    suite = root / "tests" / "pytest"

    if args.only:
        wanted = {w.strip() for w in args.only.split(",") if w.strip()}
        files = sorted(p for p in suite.glob("test_*.py")
                       if p.name in wanted)
    else:
        files = sorted(suite.glob("test_*.py"))

    env = dict(os.environ)
    env["MECH_REQUIRE_JS_LINT"] = "1"
    env.pop("MECH_SKIP_SOURCE_STABILITY_CHECK", None)

    results = []
    started = time.time()
    chunks = [files[i:i + args.chunk]
              for i in range(0, len(files), args.chunk)]
    for index, group in enumerate(chunks, 1):
        names = [p.name for p in group]
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--no-header",
                 "-p", "no:cacheprovider"] + names,
                cwd=str(root), capture_output=True, text=True,
                env=env, timeout=1500,
            )
            out_text = (proc.stdout or "") + (proc.stderr or "")
            code = proc.returncode
        except subprocess.TimeoutExpired:
            out_text = "TIMEOUT after 1500s"
            code = 124
        elapsed = round(time.time() - t0, 1)
        lines = [ln for ln in out_text.strip().splitlines() if ln.strip()]
        tail = lines[-1] if lines else ""
        record = {
            "chunk": index,
            "files": names,
            "returncode": code,
            "seconds": elapsed,
            "summary": tail.strip(),
            "not_trustworthy": "RUN NOT TRUSTWORTHY" in out_text,
        }
        results.append(record)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
        print("chunk %d/%d  %d file(s)  rc=%s  %.1fs  %s" % (
            index, len(chunks), len(names), code, elapsed, tail.strip()[:120]),
            flush=True)

    failed = [r for r in results if r["returncode"] not in (0, 5)]
    print("\n%d chunk(s), %d failed, %.1f minutes" % (
        len(results), len(failed), (time.time() - started) / 60.0), flush=True)
    for r in failed:
        print("  FAILED chunk %d rc=%s: %s" % (r["chunk"], r["returncode"],
                                               r["summary"]), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
