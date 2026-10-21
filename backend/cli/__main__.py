"""`python -m backend.cli ...` -- the console script's module twin.

The installed entry point is `mech` (pyproject `[project.scripts]`); this makes
the same CLI reachable from a checkout where nothing was installed, which is how
the tests and the packaging-free dev flow invoke it.
"""

from __future__ import annotations

import sys

from backend.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
