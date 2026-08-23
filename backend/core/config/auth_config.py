"""Authentication Configuration for MECH Platform."""

import secrets
import threading
import time
from typing import Optional

from backend.core.config.settings import get_settings


class AuthConfig:
    """Authentication configuration and validation with key rotation support."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.api_key
        self._dev_keys = set(settings.dev_keys)
        self._public_paths = set(settings.public_paths)
        self._disable_auth = settings.disable_auth
        self._environment = settings.environment
        self._key_history: list[tuple[str, float]] = [(self._api_key, time.time())]
        self._lock = threading.RLock()

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def dev_keys(self) -> set[str]:
        return self._dev_keys

    @property
    def public_paths(self) -> set[str]:
        return self._public_paths

    @property
    def disable_auth(self) -> bool:
        return self._disable_auth

    @property
    def environment(self) -> str:
        return self._environment

    def is_public_path(self, path: str) -> bool:
        """Check if path is public (no auth required)."""
        if path in self._public_paths:
            return True
        if path.startswith("/docs") or path.startswith("/redoc"):
            return True
        return False

    def is_auth_disabled(self) -> bool:
        """Check if authentication is disabled (test env or explicit flag)."""
        return self._disable_auth or self._environment == "test"

    def validate_api_key(self, api_key: Optional[str]) -> bool:
        """Validate API key against current key, dev keys, and recent history (for rotation grace period)."""
        if not api_key:
            return False
        if self.is_auth_disabled():
            return True
        with self._lock:
            if secrets.compare_digest(api_key, self._api_key):
                return True
            for dev_key in self._dev_keys:
                if secrets.compare_digest(api_key, dev_key):
                    return True
            # Allow keys from last 24 hours for rotation grace period
            now = time.time()
            for key, ts in self._key_history:
                if secrets.compare_digest(api_key, key) and now - ts < 86400:
                    return True
        return False

    def rotate_key(self, new_key: Optional[str] = None) -> str:
        """Rotate the API key. Returns the new key."""
        with self._lock:
            new_key = new_key or secrets.token_urlsafe(32)
            old_key = self._api_key
            self._api_key = new_key
            self._key_history.append((new_key, time.time()))
            # Keep only last 48 hours of keys
            cutoff = time.time() - 172800
            self._key_history = [(k, ts) for k, ts in self._key_history if ts > cutoff]

            # Write new key to file for renderer sync
            settings = get_settings()
            key_path = settings.api_key_file_path
            try:
                key_path.parent.mkdir(parents=True, exist_ok=True)
                key_path.write_text(new_key, encoding="utf-8")
            except Exception as exc:
                import logging
                logging.getLogger("MECH").warning("Could not write rotated API key file: %s", exc)
                return
            try:
                from backend.utils.secure_file import secure_file_permissions
                secure_file_permissions(key_path)
            except Exception as exc:
                import logging
                logging.getLogger("MECH").debug("Could not secure rotated API key file permissions: %s", exc)

            import logging
            logging.getLogger("MECH").info("API key rotated (old key valid for 24h grace period)")
            return new_key

    def get_key_history(self) -> list[tuple[str, float]]:
        """Get key rotation history (for debugging)."""
        with self._lock:
            return list(self._key_history)


_auth_config_instance: Optional[AuthConfig] = None


def get_auth_config() -> AuthConfig:
    """Get authentication configuration singleton."""
    global _auth_config_instance
    if _auth_config_instance is None:
        _auth_config_instance = AuthConfig()
    return _auth_config_instance


def reset_auth_config() -> None:
    """Reset auth config singleton (for testing)."""
    global _auth_config_instance
    _auth_config_instance = None