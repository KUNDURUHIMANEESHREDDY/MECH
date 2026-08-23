"""Base Factory Classes for MECH Platform."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

logger = logging.getLogger("MECH.factories")

T = TypeVar("T")


class Factory(ABC):
    """Abstract base factory."""

    def __init__(self) -> None:
        self._instances: Dict[str, Any] = {}
        self._registered_types: Dict[str, Type] = {}

    @abstractmethod
    def get_type_name(self) -> str:
        """Return the factory type name (e.g., 'engine', 'runtime')."""
        pass

    def register(self, name: str, cls: Type[T]) -> None:
        """Register a type with the factory."""
        if name in self._registered_types:
            logger.warning("Overwriting registered %s: %s", self.get_type_name(), name)
        self._registered_types[name] = cls
        logger.debug("Registered %s: %s", self.get_type_name(), name)

    def unregister(self, name: str) -> bool:
        """Unregister a type."""
        if name in self._registered_types:
            del self._registered_types[name]
            return True
        return False

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Create an instance of a registered type."""
        if name not in self._registered_types:
            raise ValueError(
                f"Unknown {self.get_type_name()}: {name}. "
                f"Available: {list(self._registered_types.keys())}"
            )
        cls = self._registered_types[name]
        return cls(*args, **kwargs)

    def get_singleton(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Get or create a singleton instance."""
        key = f"{name}:{args}:{frozenset(kwargs.items())}"
        if key not in self._instances:
            self._instances[key] = self.create(name, *args, **kwargs)
        return self._instances[key]

    def list_types(self) -> Dict[str, Type]:
        """List all registered types."""
        return self._registered_types.copy()

    def clear_instances(self) -> None:
        """Clear cached instances."""
        self._instances.clear()


class ConfigurableFactory(Factory):
    """Factory that supports configuration-based instantiation."""

    def create_from_config(self, config: Dict[str, Any]) -> Any:
        """Create instance from configuration dict."""
        name = config.pop("type", None)
        if not name:
            raise ValueError("Config must contain 'type' field")
        return self.create(name, **config)