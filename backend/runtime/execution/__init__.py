"""Model execution API."""

# Deferred to first attribute access: execution.runner imports the torch
# model stack (~8s), which should not load during app/API startup.

_LAZY_EXPORTS = {
    "ExecutionResult": ("backend.runtime.execution.runner", "ExecutionResult"),
    "ModelExecutor": ("backend.runtime.execution.runner", "ModelExecutor"),
    "TokenizedPrompt": ("backend.runtime.execution.types", "TokenizedPrompt"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    entry = _LAZY_EXPORTS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(entry[0])
    value = getattr(module, entry[1])
    globals()[name] = value
    return value
