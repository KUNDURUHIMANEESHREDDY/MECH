"""Epic 9: Error Framework — typed runtime exceptions."""


class RuntimeError(Exception):
    """Base runtime error."""


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded."""


class ModelNotFoundError(RuntimeError):
    """Raised when a requested model is not in the registry."""


class InferenceError(RuntimeError):
    """Raised when inference fails."""


class HookError(RuntimeError):
    """Raised when hook registration or execution fails."""


class SessionError(RuntimeError):
    """Raised when a session operation fails."""


class SessionNotFoundError(SessionError):
    """Raised when a session ID does not exist."""


class CacheError(RuntimeError):
    """Raised when a cache operation fails."""


class MemoryError(RuntimeError):
    """Raised when the runtime runs out of memory."""


class APIError(RuntimeError):
    """Raised when an API-level error occurs."""


class HealthCheckError(RuntimeError):
    """Raised when a health check component fails."""
