"""Configuration Management Module for MECH Platform."""

from backend.core.config.auth_config import AuthConfig, get_auth_config
from backend.core.config.cors_config import add_cors_middleware, get_cors_config
from backend.core.config.model_config import (
    ModelConfig,
    get_builtin_model,
    get_model_config,
    list_builtin_models,
)
from backend.core.config.rate_limit_config import (
    InMemoryRateLimitStore,
    RateLimitStore,
    RedisRateLimitStore,
    create_rate_limit_store,
    get_rate_limit_config,
)
from backend.core.config.settings import Settings, get_settings, reset_settings_cache

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    "reset_settings_cache",
    # CORS
    "get_cors_config",
    "add_cors_middleware",
    # Rate Limiting
    "RateLimitStore",
    "InMemoryRateLimitStore",
    "RedisRateLimitStore",
    "create_rate_limit_store",
    "get_rate_limit_config",
    # Auth
    "AuthConfig",
    "get_auth_config",
    # Model
    "ModelConfig",
    "get_model_config",
    "get_builtin_model",
    "list_builtin_models",
]