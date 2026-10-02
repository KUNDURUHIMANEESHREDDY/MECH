"""MECH Platform - Unified Root Server Entry Point.

This root entry point aligns with and re-exports the canonical backend server
defined in `backend.main`. It ensures consistent routing (/api and /api/v1),
CORS configuration, and lifecycle management regardless of whether the
platform is started via `python main.py`, `uvicorn main:app`, or `backend/main.py`.
"""
import logging
import sys
from pathlib import Path

# Ensure repository root and backend directory are in sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Re-export canonical FastAPI application instance for uvicorn main:app and test imports
from backend.main import app  # noqa: F401

logger = logging.getLogger("MECH")

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Starting MECH Platform from root entry point on http://127.0.0.1:8000 ...")
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        timeout_keep_alive=600,
    )
