"""MECH Platform - FastAPI Server Entry Point.

This is the backend entry used by the Electron app in both development and the
packaged .exe. Keep /api routes available without needing a separate server.
"""
import asyncio
import importlib.util
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# When this file is executed directly (`python backend/main.py`) Python puts
# `backend/` on sys.path, not the repository root, so the string import in
# `uvicorn.run("backend.main:app")` below would fail with
# `ModuleNotFoundError: No module named 'backend'`. Put the repo root first.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("MECH")

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application startup/shutdown.

    Replaces the deprecated `@app.on_event` hooks. Teardown checkpoints the
    SQLite WAL so a clean shutdown does not leave a large -wal file behind.
    A checkpoint failure is logged and swallowed: it is not a reason to fail
    shutdown, and the WAL remains durable either way.
    """
    logger.info("MECH starting up (lifespan).")
    try:
        from backend.plugins.service import get_service
        restored = get_service().restore_enabled()
        if restored:
            logger.info("Restored enabled plugins: %s", ", ".join(restored))
    except Exception as exc:
        logger.warning("Plugin restore skipped: %s", exc)

    await _preload_gpt2_engine()

    try:
        yield
    finally:
        logger.info("MECH shutting down (lifespan).")
        try:
            from backend.storage.database import checkpoint_wal
            busy, log_pages, checkpointed = checkpoint_wal()
            if busy:
                logger.warning(
                    "WAL checkpoint busy at shutdown (busy=%s log=%s "
                    "checkpointed=%s); the WAL stays durable but is not "
                    "truncated.", busy, log_pages, checkpointed)
            else:
                logger.info("SQLite WAL checkpoint complete at shutdown "
                            "(%s pages).", log_pages)
        except Exception as exc:
            logger.warning("WAL checkpoint failed at shutdown: %s", exc)


app = FastAPI(
    title="MECH Research Platform",
    version="2.0.0",
    description="Mechanistic Interpretability Research Platform",
    lifespan=lifespan,
)


def _cors_origins() -> list[str]:
    """Exact origins permitted to call the API.

    Electron loads from file://, which browsers report as the opaque origin
    "null"; it stays allowed because the packaged desktop app depends on it.
    Loopback origins on arbitrary ports are handled by
    ``_LOOPBACK_ORIGIN_REGEX`` instead, since Vite binds a random port when
    5173 is taken. MECH_CORS_ORIGINS may add exact origins; entries are
    normalized so a trailing slash cannot silently fail to match.
    """
    origins = ["null"]
    for extra in os.environ.get("MECH_CORS_ORIGINS", "").split(","):
        extra = extra.strip().rstrip("/")
        if extra and extra not in origins:
            origins.append(extra)
    return origins


# Loopback only, on any port, http or https. The trailing group is anchored
# so that "http://localhost.evil.com" and "http://127.0.0.1.attacker.org" do
# NOT match -- a subdomain must not inherit loopback trust.
_LOOPBACK_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=_LOOPBACK_ORIGIN_REGEX,
    allow_credentials=True,
    # Only the verbs and headers this API actually uses.
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)


async def _preload_gpt2_engine() -> None:
    """Pre-load GPT-2 so the first interactive request does not time out.

    Failure here is not fatal: the platform stays up and every dependent view
    degrades to seeded/unavailable responses with an explicit reason.
    """
    logger.info("Backend ready. ML model modules will load on demand.")
    try:
        from backend.services import gpt2_engine
        if gpt2_engine.is_available():
            logger.info("Pre-loading GPT-2 engine (torch/transformers)...")
            result = await asyncio.to_thread(gpt2_engine.load)
            logger.info("GPT-2 engine pre-loaded: status=%s",
                        result.get("status", "unknown"))
        else:
            logger.info("GPT-2 engine not available - using seeded fallbacks.")
    except Exception as exc:
        logger.warning("GPT-2 pre-loading failed: %s", exc)


@app.middleware("http")
async def _request_log_middleware(request: Request, call_next):
    """Record request metadata for the Logging view (no bodies stored)."""
    started = time.perf_counter()
    response = await call_next(request)
    try:
        if request.url.path != "/health":
            from backend.core.request_log import record
            record(
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - started) * 1000.0,
            )
    except Exception:
        pass
    return response


@app.get("/")
def root():
    return {"name": "MECH Platform", "version": "2.0.0", "status": "running"}


@app.get("/health")
def health():
    """Liveness only. Deliberately does not touch ML or storage subsystems.

    A liveness probe that loads torch or writes to disk turns a slow dependency
    into an orchestrator restart loop. For what the deployment can actually do,
    see /health/subsystems.
    """
    return {"status": "healthy"}


@app.get("/health/subsystems")
def health_subsystems(subsystem: Optional[str] = None):
    """Prove each subsystem rather than assuming it.

    Reports process, dependency, model, executor, storage, and evidence health.
    The status vocabulary has no 'healthy' member: nothing here proves health,
    only that a subsystem was probed and what it reported.
    """
    from backend.core.health import health_snapshot, run_probes, overall

    if subsystem:
        wanted = [s.strip() for s in subsystem.split(",") if s.strip()]
        results = run_probes(wanted)
        return {"status": overall(results), "subsystems": results}

    return health_snapshot()


# Main frontend services call /api/*. Keep legacy /api/v1/* too for older views.
# The dispatcher is the core router: if it fails to load, every frontend feature
# is missing and the platform is useless. Fail loudly at startup rather than
# serving a half-working app that merely *looks* healthy.
try:
    from backend.api.dispatcher import router as api_router

    app.include_router(api_router, prefix="/api")
    app.include_router(api_router, prefix="/api/v1")
    logger.info("API dispatcher loaded at /api and /api/v1.")
except Exception:
    logger.exception("API dispatcher failed to load; the platform cannot serve "
                     "/api or /api/v1. Refusing to start.")
    raise

# Optional v2 runtime API. `backend.api.runtime_api` does not currently exist,
# so its absence is an expected condition and gets a warning. A *present but
# broken* module is an infrastructure failure and is logged with a traceback.
# Either way /api/v2 is never mounted silently.
try:
    if importlib.util.find_spec("backend.api.runtime_api") is None:
        logger.warning(
            "/api/v2 not mounted: backend.api.runtime_api does not exist. "
            "Any client calling /api/v2 will get a 404 -- capability absent, "
            "not degraded."
        )
    else:
        from backend.api.runtime_api import router as runtime_router

        app.include_router(runtime_router, prefix="/api/v2")
        logger.info("Runtime v2 API loaded at /api/v2.")
except Exception:
    logger.exception("Runtime v2 API module exists but failed to load; "
                     "/api/v2 is absent.")


if __name__ == "__main__":
    logger.info("Starting MECH Platform backend on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, timeout_keep_alive=600)
