"""MECH Platform - FastAPI Server Entry Point.

This is the backend entry used by the Electron app in both development and the
packaged .exe. Keep /api routes available without needing a separate server.
"""
import asyncio
import logging
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("MECH")

app = FastAPI(
    title="MECH Research Platform",
    version="2.0.0",
    description="Mechanistic Interpretability Research Platform",
)

# Electron loads from file:// in production and Vite in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "null",
        "file://",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup_probe():
    """Start fast; load ML modules/models only when the frontend asks."""
    logger.info("Backend ready. ML model modules will load on demand.")


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
