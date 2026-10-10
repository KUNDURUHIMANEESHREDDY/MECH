"""Capture screenshots of the running MECH UI -- thin wrapper.

The implementation is ``scripts/capture_screenshots.cjs``. This module exists
only so ``python scripts/capture_screenshots.py`` keeps working, and it holds no
logic of its own.

Why the logic moved
-------------------
There were three capture scripts -- this one, ``capture_screenshots.cjs`` and
``capture_screenshots_interactive.cjs`` -- and they had already diverged. Only
the interactive one loaded the model and ran a prompt; only the interactive one
visited the Research Society view. So "the screenshots are up to date" depended
on which script a person remembered to run, and a reviewer could not tell from
the PNGs which experiment produced them.

They were also all documentation rather than tests: ``goto``, a fixed sleep, a
screenshot, and a success message. An API failure left the UI in its error or
loading state, the sleep expired, and the script reported success over a picture
of a failure. Two copies of a screenshot runner is two places for the
assertions to be added to one and not the other, so there is now one.

Usage:
    python scripts/capture_screenshots.py [--only steering,explorer] [--] [args...]

Everything after ``--`` is passed straight through to the Node script:

    python scripts/capture_screenshots.py -- --only=steering --no-load-model
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "capture_screenshots.cjs"


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    # Playwright lives in frontend/node_modules rather than as a Python package,
    # so the Node script is the one that can actually resolve it here.
    node = shutil.which("node")
    if node is None:
        print("[screenshots] node is not on PATH; the capture script needs it.",
              file=sys.stderr)
        return 2
    if not SCRIPT.exists():
        print(f"[screenshots] missing {SCRIPT}", file=sys.stderr)
        return 2

    # `--` is optional: anything that is not one of our own flags is forwarded.
    forwarded = [a for a in args if a != "--"]
    result = subprocess.run([node, str(SCRIPT), *forwarded], cwd=str(ROOT))
    # The exit code is the script's, not this wrapper's. Swallowing it here is
    # how a failed capture ends up looking like a successful one.
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
