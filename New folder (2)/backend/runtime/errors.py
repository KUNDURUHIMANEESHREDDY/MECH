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
