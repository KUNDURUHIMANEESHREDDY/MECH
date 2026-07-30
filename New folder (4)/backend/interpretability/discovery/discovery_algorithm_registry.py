"""Decoupled Discovery Algorithm Registry."""

from __future__ import annotations

from typing import Any, Callable, Dict, List


class DiscoveryAlgorithmRegistry:
    """Plugin-style registry for discovery algorithms decoupled from DiscoveryEngine."""

    def __init__(self) -> None:
        self.algorithms: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    def register_algorithm(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self.algorithms[name] = fn

    def list_algorithms(self) -> List[str]:
        return list(self.algorithms.keys())

    def execute_algorithm(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.algorithms:
            raise KeyError(f"Algorithm '{name}' not found in DiscoveryAlgorithmRegistry")
        return self.algorithms[name](payload)
