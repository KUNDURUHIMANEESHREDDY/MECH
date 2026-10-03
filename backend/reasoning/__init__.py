"""Reasoning journey tracing and neuron debugging for model interpretability.

The two concrete implementations (`journey_tracer` and `neuron_debugger`) do
not exist in this repository. The previous `__init__.py` imported them at
package level, so `import backend.reasoning` raised ModuleNotFoundError for
anyone touching the package at all -- even though no code anywhere references
these names.

`backend.reasoning` now imports cleanly, and the names resolve lazily: they
fail with a clear reason the moment something actually tries to use one, which
is the same fail-closed rule the rest of the platform follows. If the tracers
are implemented later, move them from the lazy block to ordinary imports.
"""

from typing import Any

__all__ = [
    "JourneyStage",
    "NeuronDebugger",
    "NeuronInspection",
    "ReasoningJourney",
    "ReasoningJourneyTracer",
]


def __getattr__(name: str) -> Any:
    """Resolve the missing tracer classes only when something asks for one.

    Fails closed with a named reason rather than letting an import-time
    ModuleNotFoundError hide which symbol is unimplemented.
    """
    if name in __all__:
        raise ImportError(
            f"backend.reasoning.{name} is not implemented: the "
            f"journey_tracer / neuron_debugger modules are not present in "
            f"this repository. This is a capability gap, not an import error "
            f"to swallow."
        )
    raise AttributeError(
        f"module 'backend.reasoning' has no attribute {name!r}"
    )
