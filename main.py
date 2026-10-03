"""Thin entry-point shim for `python main.py`.

The FastAPI application is defined once, in `backend/main.py`. This module
re-exports that single `app` object so both entry points launch identical
routers, middleware, and CORS policy. It previously built a second, divergent
FastAPI instance -- a different version string, a different CORS allowlist, and
its own duplicated route registrations -- which meant the active security
posture depended on which file was launched.

Run either of these; both now serve the same application:

    python main.py
    uvicorn backend.main:app
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.main import app  # noqa: E402  (requires the sys.path entry above)

logger = logging.getLogger("MECH")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # Loopback only. This API has no authentication and exposes plugin
    # install/enable, so binding 0.0.0.0 put it on the LAN for anyone who could
    # reach the port. MECH_BIND_HOST makes the wider bind a deliberate choice
    # (containers, shared dev machines) rather than an accident.
    #
    # Adopted from security/plugin-and-path-hardening, which added the env
    # override; the loopback default and the shim above are from this branch.
    host = os.environ.get("MECH_BIND_HOST", "127.0.0.1")
    logger.info("Starting MECH Platform backend on http://%s:8000 ...", host)
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=8000,
        reload=False,
        timeout_keep_alive=600,
    )
