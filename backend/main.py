"""MECH Platform - FastAPI Server Entry Point.

This is the backend entry used by the Electron app in both development and the
packaged .exe. Keep /api routes available without needing a separate server.
"""
import asyncio
import logging
import os
import secrets
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=os.environ.get("MECH_LOG_LEVEL", "WARNING"), format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("MECH")

app = FastAPI(
    title="MECH Research Platform",
    version="2.0.0",
    description="Mechanistic Interpretability Research Platform",
)

_API_KEY = os.environ.get("MECH_API_KEY", secrets.token_urlsafe(32))
_PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}
_rate_limit_window = 60
_rate_limit_max_requests = 300
_rate_limit_store: Dict[str, list] = defaultdict(list)
_rate_limit_lock = asyncio.Lock()


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    path = request.url.path
    if path in _PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != _API_KEY:
        # HTTPException in ASGI middleware surfaces as 500; return JSONResponse for a real 401.
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

    return await call_next(request)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    key = f"{client_ip}:{request.url.path}"
    now = time.time()
    cutoff = now - _rate_limit_window

    async with _rate_limit_lock:
        window = _rate_limit_store[key]
        while window and window[0] < cutoff:
            window.pop(0)
        window.append(now)

        if len(window) > _rate_limit_max_requests:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

        # Periodically prune stale keys to prevent unbounded dictionary memory growth
        if len(_rate_limit_store) > 500:
            stale_keys = [k for k, v in _rate_limit_store.items() if not v or v[-1] < cutoff]
            for k in stale_keys:
                del _rate_limit_store[k]

    return await call_next(request)

# Electron loads from file:// in production and Vite in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "file://",
        # Packaged renderer pages load from file://, which Chromium sends as
        # Origin: null on cross-origin fetches.
        "null",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)


@app.on_event("startup")
async def _startup_probe():
    """Start fast; load ML modules/models only when the frontend asks."""
    user_data = os.environ.get("NEURAL_DEBUGGER_USER_DATA") or os.environ.get("MECH_USER_DATA")
    if user_data:
        key_path = Path(user_data) / "storage" / "api_key.txt"
    else:
        key_path = Path(__file__).parent.parent / "storage" / "api_key.txt"
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(_API_KEY, encoding="utf-8")
        if os.name != "nt":
            try:
                key_path.chmod(0o600)
            except Exception:
                pass
    except Exception as exc:
        logger.warning("Could not write API key file: %s", exc)


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
    from backend.interpretability.api import v1_router, legacy_router
    from backend.jobs.routes import router as jobs_router
    from backend.api.dispatcher import dispatch_router

    # Interpretability API (v1 + legacy)
    app.include_router(v1_router)
    app.include_router(legacy_router)

    # Job automation API
    app.include_router(jobs_router)

    # Legacy JSON-RPC style dispatch
    app.include_router(dispatch_router, prefix="/api")

    logger.info("Interpretability, jobs, and dispatch routers loaded.")
except Exception as e:
    logger.warning("Additional routers not loaded: %s", e)

# NOTE: runtime_api.py does not exist; removed silent failing import.
# from backend.api.runtime_api import router as runtime_router
# app.include_router(runtime_router, prefix="/api/v2")


if __name__ == "__main__":
    logger.info("Starting MECH Platform backend on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, timeout_keep_alive=600)
