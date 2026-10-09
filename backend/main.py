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
from fastapi.responses import JSONResponse
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
        from backend.core import auth as auth_mod
        auth_mod.get_bearer_token()
        logger.info("Control-plane bearer auth active (token file: %s).",
                    auth_mod._token_file())
    except Exception as exc:
        logger.warning("Bearer token init failed: %s", exc)
    try:
        from backend.core import evidence_graph as eg_mod
        report = eg_mod.quarantine_legacy_records()
        if report.get("moved"):
            logger.info("Quarantined %s legacy evidence record(s) to %s.",
                        report["moved"], report["destination"])
    except Exception as exc:
        logger.warning("Evidence quarantine skipped: %s", exc)
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


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    """Mandatory bearer gate for the control plane.

    Loopback reachability and Origin are never authorization. ``/`` and
    ``/health`` stay public, CORS preflight (OPTIONS) passes through to the
    CORS middleware, and the built frontend's static files stay public so the
    app can load. Everything else under ``/api``, ``/mcp``, ``/health/``,
    ``/openapi.json``, ``/docs`` and ``/redoc`` requires
    ``Authorization: Bearer <MECH_API_TOKEN>``.
    """
    from backend.core import auth as auth_mod

    if auth_mod.is_public_path(request.url.path, request.method):
        return await call_next(request)
    token = auth_mod.extract_bearer(request.headers.get("authorization"))
    if token is None:
        return JSONResponse(
            status_code=401,
            content={"detail": "missing credentials"},
        )
    if not auth_mod.verify_bearer(token):
        return JSONResponse(
            status_code=401,
            content={"detail": "invalid credentials"},
        )
    return await call_next(request)


def _body_cap() -> int:
    """Global request-body ceiling in bytes (MECH_MAX_BODY_BYTES override).

    Prompts have their own 2048-char bound and SAE corpora 200k chars, but
    the rest of the control plane accepts untyped `Dict[str, Any]` bodies.
    Without a ceiling an authenticated caller can push arbitrary megabytes at
    the dispatcher, so bound at the edge and let endpoint schemas be tighter.
    """
    raw = os.environ.get("MECH_MAX_BODY_BYTES", "").strip()
    try:
        value = int(raw) if raw else 2 * 1024 * 1024
    except ValueError:
        value = 2 * 1024 * 1024
    return max(1024, min(value, 64 * 1024 * 1024))


@app.middleware("http")
async def _body_size_middleware(request: Request, call_next):
    """Reject oversized bodies before they are read into memory.

    Checks the declared Content-Length first (cheap, catches the honest
    client) and counts actual streamed bytes otherwise, so a lying or
    chunked header cannot slip a large payload past. 413 is a client error;
    nothing downstream ever sees the oversized body.
    """
    cap = _body_cap()
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > cap:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"request body exceeds {cap} bytes"},
                )
        except ValueError:
            return JSONResponse(
                status_code=400, content={"detail": "malformed content-length"},
            )
    if "content-length" not in request.headers:
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > cap:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"request body exceeds {cap} bytes"},
                )
    return await call_next(request)


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


# MCP control plane: same Core API the UI calls, exposed as tools so external
# agents drive MECH without UI automation. Optional: a failed mount must not
# take the REST API or the UI down with it.
try:
    from backend.mcp_server.server import mcp as _mech_mcp

    app.mount("/mcp", _mech_mcp.streamable_http_app())
    logger.info("MCP control plane mounted at /mcp (tools: mech_*).")
except Exception:
    logger.exception("MCP control plane failed to mount; /mcp is absent.")


def _find_frontend_dist() -> Optional[str]:
    """Locate the built Vue frontend (frontend/dist) for standalone mode.

    Standalone = one process serves both UI and API, so the browser can just
    open http://127.0.0.1:8000 with no Electron and no second server.
    Returns the dist directory path, or None when not built / not bundled.
    """
    candidates = [
        os.path.join(_REPO_ROOT, "frontend", "dist"),  # repo checkout
        os.path.join(os.getcwd(), "frontend", "dist"),  # launched from root
        os.path.join(os.getcwd(), "dist"),  # portable MECH-standalone/dist
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dist"),
        os.path.join(getattr(sys, "_MECH_RESOURCES", ""), "app", "dist"),
    ]
    for cand in candidates:
        if not cand:
            continue
        norm = os.path.normpath(cand)
        index_html = os.path.join(norm, "index.html")
        if os.path.isdir(norm) and os.path.isfile(index_html):
            return norm
    return None


try:
    _DIST_DIR = _find_frontend_dist()
    if _DIST_DIR is not None:
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=_DIST_DIR, html=True), name="frontend")
        logger.info("Serving standalone frontend from %s at /", _DIST_DIR)
    else:
        logger.info(
            "No frontend dist found; running API-only. "
            "Build it with: cd frontend && npm run build:renderer"
        )
except Exception:
    logger.exception("Frontend static mount failed; running API-only.")


if __name__ == "__main__":
    logger.info("Starting MECH Platform backend on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, timeout_keep_alive=600)
