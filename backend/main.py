"""MECH Platform - FastAPI Server Entry Point.

This is the backend entry used by the Electron app in both development and the
packaged .exe. Keep /api routes available without needing a separate server.

Rate-limiting uses Redis when REDIS_URL is set (e.g., docker-compose), and
falls back to an in-memory dict store otherwise (local dev / Electron sidecar).
"""
import asyncio
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.core.config import (
    add_cors_middleware,
    create_rate_limit_store,
    get_auth_config,
    get_rate_limit_config,
    get_settings,
)

# Initialize settings early for logging
settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("MECH")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
)

# Initialize config objects
auth_config = get_auth_config()
rate_limit_config = get_rate_limit_config()
rate_limit_store = create_rate_limit_store()

RATE_LIMIT_WINDOW = rate_limit_config["window_seconds"]
RATE_LIMIT_MAX_REQUESTS = rate_limit_config["max_requests"]


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    path = request.url.path

    if auth_config.is_public_path(path):
        return await call_next(request)

    if auth_config.is_auth_disabled():
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    if auth_config.validate_api_key(api_key):
        return await call_next(request)

    return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    key = f"mech:rl:{client_ip}:{request.url.path}"

    allowed = await rate_limit_store.check_limit(key, RATE_LIMIT_WINDOW, RATE_LIMIT_MAX_REQUESTS)

    if not allowed:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    return await call_next(request)


# Add CORS middleware
add_cors_middleware(app)


@app.on_event("startup")
async def _startup_probe():
    """Start fast; load ML modules/models only when the frontend asks."""
    key_path = settings.api_key_file_path
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(settings.api_key, encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write API key file: %s", exc)
        return
    try:
        from backend.utils.secure_file import secure_file_permissions
        secure_file_permissions(key_path)
    except Exception as exc:
        logger.debug("Could not secure API key file permissions: %s", exc)

    # Initialize dynamic session probes at app open time
    try:
        from backend.runtime.dynamic_prompt_sampler import init_session

        probes, session_file = init_session(n_per_category=settings.dynamic_probe_samples_per_category)
        logger.info(
            "Dynamic probe session initialized (%d probes saved to %s).",
            len(probes),
            session_file,
        )
    except Exception as exc:
        logger.warning("Dynamic probe session initialization skipped: %s", exc)


@app.on_event("shutdown")
async def _shutdown():
    """Cleanup on shutdown."""
    await rate_limit_store.close()


@app.get("/")
def root():
    return {"name": settings.app_name, "version": settings.app_version, "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


try:
    from backend.api.experiments_router import router as experiments_router

    app.include_router(experiments_router)
    app.include_router(experiments_router, prefix="/api")
    app.include_router(experiments_router, prefix="/api/v1")
    logger.info("Experiments router loaded.")
except Exception as e:
    logger.warning("Experiments router not loaded: %s", e)

# Main frontend services call /api/*. Keep legacy /api/v1/* too for older views.
try:
    from backend.api.dispatcher import router as api_router

    app.include_router(api_router, prefix="/api")
    app.include_router(api_router, prefix="/api/v1")
    logger.info("API dispatcher loaded at /api and /api/v1.")
except Exception as e:
    logger.warning("API dispatcher not loaded: %s", e)

try:
    from backend.api.scientific_router import router as scientific_router
    from backend.api.telemetry_dag_router import router as telemetry_dag_router
    from backend.api.research_router import router as research_router
    from backend.api.reasoning_router import router as reasoning_router

    app.include_router(scientific_router)
    app.include_router(telemetry_dag_router)
    app.include_router(research_router)
    app.include_router(reasoning_router)
    logger.info("Scientific, telemetry DAG, research, and reasoning routers loaded.")
except Exception as e:
    logger.warning("Scientific, telemetry DAG, research, or reasoning router not loaded: %s", e)

try:
    from backend.api.dispatcher import dispatch_router
    from backend.interpretability.api import legacy_router, v1_router

    # Interpretability API (v1 + legacy)
    app.include_router(v1_router)
    app.include_router(legacy_router)

    # Legacy JSON-RPC style dispatch
    app.include_router(dispatch_router, prefix="/api")

    logger.info("Interpretability and dispatch routers loaded.")
except Exception as e:
    logger.warning("Additional routers not loaded: %s", e)


if __name__ == "__main__":
    logger.info("Starting MECH Platform backend on http://%s:%d ...", settings.host, settings.port)
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        timeout_keep_alive=settings.timeout_keep_alive,
    )