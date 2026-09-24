from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from pathlib import Path
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("MECH")

app = FastAPI(
    title="MECH Research Platform",
    version="2.0",
    description="Mechanistic Interpretability Research Platform"
)

# Allow frontend (Vite). Extra dev origins via MECH_CORS_ORIGINS (comma-separated).
import os as _os

_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173", "null", "file://"]
_CORS_ORIGINS += [o.strip() for o in _os.environ.get("MECH_CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent

@app.on_event("startup")
async def _preload_gpt2_engine():
    """Pre-load GPT-2 model at startup to avoid first-request timeout."""
    try:
        from backend.services import gpt2_engine
        if gpt2_engine.is_available():
            logger.info("Pre-loading GPT-2 engine (torch/transformers)...")
            result = await asyncio.to_thread(gpt2_engine.load)
            status = result.get("status", "unknown")
            logger.info(f"GPT-2 engine pre-loaded: status={status}")
        else:
            logger.info("GPT-2 engine not available — using seeded fallbacks.")
    except Exception as e:
        logger.warning(f"GPT-2 pre-loading failed: {e}")

@app.get("/")
def home():
    return {
        "name": "MECH Research Platform",
        "status": "running",
        "version": "2.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# --------------------------
# Register API routers
# --------------------------
try:
    from backend.api.dispatcher import router as api_router
    app.include_router(api_router, prefix="/api")
    logger.info("API Dispatcher loaded.")
except Exception as e:
    logger.warning(f"Dispatcher not loaded: {e}")

# Try loading the v2 runtime API if available
try:
    from backend.api.runtime_api import router as runtime_router
    app.include_router(runtime_router, prefix="/api/v2")
    logger.info("Runtime v2 API loaded.")
except Exception:
    pass

if __name__ == "__main__":
    logger.info("Starting MECH Platform...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_keep_alive=600,
    )
