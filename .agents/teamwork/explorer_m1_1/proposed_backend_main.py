"""MECH Platform - FastAPI Server Entry Point (Proposed Modernization).

Canonical backend entry used by Electron in development/packaged build and standalone servers.
"""
import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("MECH")


async def _preload_gpt2() -> None:
    """Pre-load GPT-2 model in background thread to avoid first-request latency."""
    try:
        from backend.services import gpt2_engine
        if hasattr(gpt2_engine, "is_available") and gpt2_engine.is_available():
            logger.info("Pre-loading GPT-2 engine (torch/transformers)...")
            result = await asyncio.to_thread(gpt2_engine.load)
            status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
            logger.info("GPT-2 engine pre-load complete: status=%s", status)
        else:
            logger.info("GPT-2 engine not available — using seeded fallbacks.")
    except Exception as e:
        logger.warning("GPT-2 pre-loading failed or skipped: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager replacing deprecated @app.on_event hooks.

    Manages clean startup (non-blocking model preloading when requested) and
    graceful shutdown (flushing and truncating SQLite WAL file).
    """
    logger.info("MECH Platform backend starting up...")

    preload_task: Optional[asyncio.Task] = None
    if os.environ.get("MECH_PRELOAD_MODELS", "0").lower() in ("1", "true", "yes"):
        logger.info("MECH_PRELOAD_MODELS enabled: scheduling background GPT-2 preloading...")
        preload_task = asyncio.create_task(_preload_gpt2())
    else:
        logger.info("Backend ready. ML model modules will load on demand.")

    yield

    logger.info("MECH Platform backend shutting down...")

    # Cancel background model pre-loading if still in progress
    if preload_task and not preload_task.done():
        preload_task.cancel()
        try:
            await preload_task
        except (asyncio.CancelledError, Exception):
            pass

    # Clean SQLite database resources and WAL checkpoint
    try:
        from backend.storage.database import checkpoint_wal
        busy, log_pages, checkpointed = checkpoint_wal()
        logger.info(
            "SQLite WAL checkpoint complete (busy=%d, log=%d, checkpointed=%d).",
            busy,
            log_pages,
            checkpointed,
        )
    except Exception as e:
        logger.warning("Error during SQLite WAL checkpoint on shutdown: %s", e)


app = FastAPI(
    title="MECH Research Platform",
    version="2.0.0",
    description="Mechanistic Interpretability Research Platform",
    lifespan=lifespan,
)

# Electron loads from file:// in production and Vite in development.
_extra_origins = [
    o.strip()
    for o in os.environ.get("MECH_CORS_ORIGINS", "").split(",")
    if o.strip()
]
_cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "null",
    "file://",
] + _extra_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "MECH Platform", "version": "2.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# Main frontend services call /api/*. Keep legacy /api/v1/* too for older views.
try:
    from backend.api.dispatcher import router as api_router

    app.include_router(api_router, prefix="/api")
    app.include_router(api_router, prefix="/api/v1")
    logger.info("API dispatcher loaded at /api and /api/v1.")
except Exception as e:
    logger.warning("API dispatcher not loaded: %s", e)

try:
    from backend.api.runtime_api import router as runtime_router

    app.include_router(runtime_router, prefix="/api/v2")
    logger.info("Runtime v2 API loaded at /api/v2.")
except Exception:
    pass


if __name__ == "__main__":
    logger.info("Starting MECH Platform backend on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, timeout_keep_alive=600)
