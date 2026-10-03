"""Runtime-specific exceptions and error mapping helpers."""

from __future__ import annotations

import torch


class RuntimeExecutionError(Exception):
    """Base exception for model runtime failures."""


class MissingModelError(RuntimeExecutionError):
    """Raised when a requested model cannot be resolved or loaded."""


class ModelOutOfMemoryError(RuntimeExecutionError):
    """Raised when model loading or execution runs out of memory."""


class InvalidPromptError(RuntimeExecutionError):
    """Raised when a prompt cannot be tokenized or executed."""


# The exceptions below are raised by six runtime modules that imported them from
# here, which this module did not define. Every one of those modules therefore
# failed at import time:
#
#   activation_cache, health          -> CacheError
#   hook_framework, interpreter       -> HookError
#   model_manager                     -> ModelLoadError, ModelNotFoundError
#   session_manager                   -> SessionNotFoundError
#
# They are defined here rather than inlined at the raise sites because `errors`
# is the intended home for them and six modules already agree on that. The
# hierarchy is chosen so existing broad handlers keep working:
# `MissingModelError` is already documented as "a requested model cannot be
# resolved or loaded", which is exactly `ModelLoadError`, so it is the correct
# base rather than a parallel one.


class CacheError(RuntimeExecutionError):
    """Raised when the activation cache cannot be read, written or evicted."""


class HookError(RuntimeExecutionError):
    """Raised when a hook is unknown, malformed, or registered incorrectly."""


class ModelLoadError(MissingModelError):
    """Raised when loading a model fails, or when no model is loaded.

    Subclasses MissingModelError, which already covers "cannot be resolved or
    loaded", so `except MissingModelError` catches this too.
    """


class ModelNotFoundError(MissingModelError):
    """Raised when a named model is not present in the registry.

    Subclasses MissingModelError for the same reason as ModelLoadError.
    """


class SessionNotFoundError(RuntimeExecutionError):
    """Raised when a session id does not resolve to a stored session."""


def is_out_of_memory_error(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True

    message = str(error).lower()
    return (
        "out of memory" in message
        or "cuda oom" in message
        or "cublas_status_alloc_failed" in message
        or "defaultcpuallocator" in message and "can't allocate memory" in message
    )
