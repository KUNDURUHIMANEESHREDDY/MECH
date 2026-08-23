"""Centralized Pydantic Settings for MECH Platform."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App metadata
    app_name: str = "MECH Research Platform"
    app_version: str = "2.0.0"
    app_description: str = "Mechanistic Interpretability Research Platform"
    log_level: str = Field(default="WARNING", alias="MECH_LOG_LEVEL")
    environment: str = Field(default="development", alias="MECH_ENV")

    # API Configuration - NO DEFAULTS for security-sensitive values
    api_key: str = Field(alias="MECH_API_KEY")
    dev_keys: list[str] = Field(default=[], alias="MECH_DEV_KEYS")
    disable_auth: bool = Field(default=False, alias="MECH_DISABLE_AUTH")
    public_paths: list[str] = Field(default=["/", "/health", "/docs", "/redoc", "/openapi.json"])

    # CORS Configuration
    cors_allow_origins: list[str] = Field(
        default=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "file://",
            "null",
        ],
        alias="MECH_CORS_ORIGINS",
    )
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = Field(default=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    cors_allow_headers: list[str] = Field(default=["*"])
    cors_expose_headers: list[str] = Field(default=["*"])
    cors_max_age: int = 600

    # Rate Limiting Configuration
    rate_limit_window_seconds: int = Field(default=60, alias="MECH_RATE_LIMIT_WINDOW")
    rate_limit_max_requests: int = Field(default=300, alias="MECH_RATE_LIMIT_MAX_REQUESTS")
    rate_limit_redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    rate_limit_redis_connect_timeout: int = 2
    rate_limit_redis_socket_timeout: int = 2
    rate_limit_in_memory_max_keys: int = 500
    rate_limit_fail_closed: bool = Field(default=True, alias="MECH_RATE_LIMIT_FAIL_CLOSED")

    # Redis Configuration
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    redis_encoding: str = "utf-8"
    redis_decode_responses: bool = True

    # Storage Configuration
    user_data_dir: Optional[str] = Field(default=None, alias="MECH_USER_DATA")
    neural_debugger_user_data: Optional[str] = Field(default=None, alias="NEURAL_DEBUGGER_USER_DATA")
    storage_api_key_filename: str = "api_key.txt"

    # Model Configuration
    default_model: str = Field(default="gpt2", alias="MECH_DEFAULT_MODEL")
    model_zoo_path: Optional[str] = Field(default=None, alias="MECH_MODEL_ZOO_PATH")
    model_cache_dir: Optional[str] = Field(default=None, alias="MECH_MODEL_CACHE_DIR")

    # Runtime Configuration
    dynamic_probe_samples_per_category: int = Field(default=2, alias="MECH_DYNAMIC_PROBE_SAMPLES")

    # Server Configuration
    host: str = Field(default="127.0.0.1", alias="MECH_HOST")
    port: int = Field(default=8000, alias="MECH_PORT")
    reload: bool = Field(default=False, alias="MECH_RELOAD")
    timeout_keep_alive: int = 600

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Ensure API key is not a known default in production."""
        if v in {"mech_dev_key_default", "dev", "test", "changeme", "secret"}:
            import os
            if os.environ.get("MECH_ENV") == "production":
                raise ValueError("Default API key not allowed in production. Set MECH_API_KEY to a secure random value.")
        return v

    @field_validator("disable_auth")
    @classmethod
    def validate_disable_auth(cls, v: bool) -> bool:
        """Warn when auth is disabled."""
        if v:
            import warnings
            warnings.warn(
                "Authentication is DISABLED (MECH_DISABLE_AUTH=1). "
                "This is a security risk and should only be used in isolated test environments.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @property
    def effective_user_data_dir(self) -> Path:
        """Get the effective user data directory."""
        base = self.user_data_dir or self.neural_debugger_user_data
        if base:
            return Path(base)
        return Path(__file__).parent.parent.parent / "storage"

    @property
    def api_key_file_path(self) -> Path:
        """Get the API key file path."""
        return self.effective_user_data_dir / "storage" / self.storage_api_key_filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Reset the settings cache (useful for testing)."""
    get_settings.cache_clear()