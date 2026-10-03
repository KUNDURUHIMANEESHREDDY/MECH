"""Assert the remote actually has the local commit.

Written after a real failure: a ``git push -q`` was reported as successful
because the author's own "pushed" echo stood in for the suppressed output. The
push had not landed -- a stale local branch ref was being pushed, and a
shallow clone's narrowed refspec meant ``git fetch`` never refreshed that ref
either, so the staleness was invisible.

This answers "did it land?" from ``git ls-remote``, which asks the remote
directly rather than inferring from local bookkeeping.

    python verify_remote_sync.py                 # check the tracking branch
    python verify_remote_sync.py --branch other  # check another branch
    python verify_remote_sync.py --ref master    # check a raw ref

Exit 0 when the remote has the local commit, 1 otherwise.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def remote_sha(remote: str, ref: str) -> str | None:
    """Ask the remote for a ref's SHA. Authoritative; ignores local state."""
    out = git("ls-remote", remote, ref, check=False)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    return None


def local_sha(rev: str = "HEAD") -> str | None:
    out = git("rev-parse", "--verify", rev, check=False)
    return out if out and out != rev else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default=None,
                        help="branch to check (default: current tracking branch)")
    parser.add_argument("--ref", default=None,
                        help="full ref to check, e.g. refs/heads/master")
    parser.add_argument("--expect", default=None,
                        help="assert the remote equals this SHA instead of HEAD")
    args = parser.parse_args()

    if args.ref:
        ref = args.ref
        expected = args.expect or local_sha()
        if expected is None:
            print("no local commit to compare against")
            return 1
    else:
        branch = args.branch or git("rev-parse", "--abbrev-ref", "HEAD")
        ref = f"refs/heads/{branch}"
        expected = args.expect or local_sha()
        if expected is None:
            print("no local commit to compare against")
            return 1

    actual = remote_sha(args.remote, ref)
    short = expected[:7]
    # refs/heads/x -> x, so the label reads "origin/x" not "origin/refs/heads/x".
    label = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref

    print("=" * 60)
    print("remote sync check")
    print("=" * 60)
    print(f"  ref      : {args.remote}/{label}")
    print(f"  local    : {short}")
    print(f"  remote   : {(actual or 'MISSING')[:7]}")

    if actual is None:
        print(f"\nFAIL: {args.remote}/{label} does not exist. Not pushed.")
        return 1

    if actual == expected:
        print(f"\nOK: {ref} on {args.remote} is at {short}.")
        return 0

    # Diagnose the mismatch rather than just reporting it.
    print(f"\nFAIL: {args.remote}/{label} is at {actual[:7]}, local is {short}.")
    try:
        behind = git("rev-list", "--count",
                     f"{expected}..{actual}", check=False)
        ahead = git("rev-list", "--count",
                    f"{actual}..{expected}", check=False)
        print(f"  remote is ahead by {behind or '?'} commit(s), "
              f"local ahead by {ahead or '?'}")
        if behind and behind != "0" and ahead and ahead != "0":
            print("  diverged: fetch and reconcile before pushing again")
        elif ahead and ahead != "0":
            print("  local commits are NOT on the remote. The push did not land.")
            print(f"  push with: git push {args.remote} HEAD:{ref}")
        elif behind and behind != "0":
            print("  remote has commits you do not: git pull --rebase")
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not compute divergence: {exc})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
