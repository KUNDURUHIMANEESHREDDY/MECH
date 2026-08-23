"""MECH Platform - Unified Server Entry Point.

Loads the core FastAPI application from backend.main and serves the frontend
single-page application (SPA) when built dist files are present.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Import the configured FastAPI app from backend.main
from backend.main import app

logger = logging.getLogger("MECH")

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

# Mount frontend assets and provide SPA fallback if dist exists
if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").is_file():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path == "api" or full_path.startswith("api/") or full_path in {"health", "docs", "redoc", "openapi.json"}:
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIST / "index.html"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting MECH Platform on http://{host}:{port} ...")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        timeout_keep_alive=600,
    )