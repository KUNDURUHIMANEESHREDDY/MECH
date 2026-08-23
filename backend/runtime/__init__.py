"""Runtime APIs for loading and executing language models."""

# Eager imports here pull torch/transformers (~11s) into app/API startup
# (execution.runner and models.gpt2 both import the ML stack). Names are
# resolved on first attribute access instead.

_LAZY_EXPORTS = {
    "InvalidPromptError": ("backend.runtime.errors", "InvalidPromptError"),
    "MissingModelError": ("backend.runtime.errors", "MissingModelError"),
    "ModelOutOfMemoryError": ("backend.runtime.errors", "ModelOutOfMemoryError"),
    "RuntimeExecutionError": ("backend.runtime.errors", "RuntimeExecutionError"),
    "ExecutionResult": ("backend.runtime.execution.runner", "ExecutionResult"),
    "ModelExecutor": ("backend.runtime.execution.runner", "ModelExecutor"),
    "GPT2ModelLoader": ("backend.runtime.models.gpt2", "GPT2ModelLoader"),
    "LoadedModel": ("backend.runtime.models.gpt2", "LoadedModel"),
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
