"""Algorithm Registry.

Enables dynamic registration and plugin extension of interpretability algorithms.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class AlgorithmRegistry:
    """Registry for mechanistic interpretability algorithms."""

    def __init__(self) -> None:
        self._algorithms: Dict[str, Any] = {}

    def register(self, name: str, algorithm_instance: Any) -> None:
        self._algorithms[name] = algorithm_instance

    def get(self, name: str) -> Optional[Any]:
        return self._algorithms.get(name)

    # Was `List[str]`, with no `List` in scope. It did not raise at import
    # because this module has `from __future__ import annotations`, which
    # leaves annotations unevaluated strings -- so the NameError was latent
    # rather than immediate, surfacing only if something called
    # typing.get_type_hints() on this method. The sibling registry
    # (discovery/algorithms/registry.py) already uses PEP 585 lowercase, so this
    # file had simply been missed by that migration.
    def list_algorithms(self) -> list[str]:
        return list(self._algorithms.keys())


_algorithm_registry_instance: AlgorithmRegistry | None = None


def get_algorithm_registry() -> AlgorithmRegistry:
    global _algorithm_registry_instance
    if _algorithm_registry_instance is None:
        _algorithm_registry_instance = AlgorithmRegistry()
    return _algorithm_registry_instance
