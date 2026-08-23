"""Runtime Factory for MECH Platform."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from backend.core.factories.base import ConfigurableFactory
from backend.core.config import get_settings

logger = logging.getLogger("MECH.factories.runtime")


class RuntimeBackend(ABC):
    """Base class for runtime backends."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    @abstractmethod
    def execute(self, dag: Any, inputs: Dict[str, Any]) -> Any:
        """Execute a computation DAG."""
        pass

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        pass


class RuntimeFactory(ConfigurableFactory):
    """Factory for runtime backends."""

    def __init__(self) -> None:
        super().__init__()
        self._register_builtins()

    def get_type_name(self) -> str:
        return "runtime"

    def _register_builtins(self) -> None:
        """Register built-in runtime types."""
        try:
            from backend.runtime.in_memory_runtime import InMemoryRuntime
            self.register("in_memory", InMemoryRuntime)
        except ImportError as e:
            logger.debug("InMemoryRuntime not available: %s", e)

        try:
            from backend.runtime.out_of_core_runtime import OutOfCoreRuntime
            self.register("out_of_core", OutOfCoreRuntime)
        except ImportError as e:
            logger.debug("OutOfCoreRuntime not available: %s", e)

        try:
            from backend.runtime.adaptive import AdaptiveRuntime
            self.register("adaptive", AdaptiveRuntime)
        except ImportError as e:
            logger.debug("AdaptiveRuntime not available: %s", e)

    def create_runtime(
        self,
        runtime_type: str = "auto",
        memory_budget_gb: Optional[float] = None,
        model_id: str = "gpt2",
        **kwargs: Any,
    ) -> RuntimeBackend:
        """Create a runtime backend."""
        settings = get_settings()
        if runtime_type == "auto":
            # Auto-select based on available memory
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            if memory_budget_gb and available_gb > memory_budget_gb * 2:
                runtime_type = "in_memory"
            else:
                runtime_type = "out_of_core"
            logger.info(
                "Auto-selected runtime: %s (available: %.1f GB, budget: %s)",
                runtime_type,
                available_gb,
                memory_budget_gb,
            )

        # Pass model_id and other kwargs to the runtime constructor
        create_kwargs = {"model_id": model_id}
        create_kwargs.update(kwargs)
        
        if runtime_type == "out_of_core" and memory_budget_gb:
            create_kwargs["ram_budget_mb"] = int(memory_budget_gb * 1024)
        
        return self.create(runtime_type, **create_kwargs)

    def list_runtimes(self) -> List[str]:
        """List available runtime types."""
        return list(self._registered_types.keys())


# Global instance
_runtime_factory: Optional[RuntimeFactory] = None


def get_runtime_factory() -> RuntimeFactory:
    """Get the global runtime factory."""
    global _runtime_factory
    if _runtime_factory is None:
        _runtime_factory = RuntimeFactory()
    return _runtime_factory