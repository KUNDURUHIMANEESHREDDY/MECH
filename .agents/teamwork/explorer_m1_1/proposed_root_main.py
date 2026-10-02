"""MECH Platform - Root Server Entry Point Forwarder (Proposed).

Forwards to the canonical backend application entry point in `backend/main.py`.
Maintains full backward compatibility for invocations targeting root main.py.
"""
import logging
import uvicorn

from backend.main import app

logger = logging.getLogger("MECH")

if __name__ == "__main__":
    logger.info("Starting MECH Platform via canonical backend entry on http://127.0.0.1:8000 ...")
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        timeout_keep_alive=600,
    )
