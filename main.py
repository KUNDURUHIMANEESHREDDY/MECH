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
    logger.info("Starting MECH Platform backend on http://127.0.0.1:8000 ...")
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        timeout_keep_alive=600,
    )
