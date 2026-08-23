"""CORS Configuration for MECH Platform."""

from backend.core.config.settings import get_settings


def get_cors_config() -> dict:
    """Get CORS middleware configuration."""
    settings = get_settings()
    return {
        "allow_origins": settings.cors_allow_origins,
        "allow_credentials": settings.cors_allow_credentials,
        "allow_methods": settings.cors_allow_methods,
        "allow_headers": settings.cors_allow_headers,
        "expose_headers": settings.cors_expose_headers,
        "max_age": settings.cors_max_age,
    }


def add_cors_middleware(app) -> None:
    """Add CORS middleware to FastAPI app."""
    from fastapi.middleware.cors import CORSMiddleware

    config = get_cors_config()
    app.add_middleware(CORSMiddleware, **config)