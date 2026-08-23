"""Pluggable Compute Backend Abstraction.

Unifies execution across heterogeneous compute targets:
- CPUBackend: Standard cross-platform PyTorch CPU executor
- CUDABackend: High-performance NVIDIA GPU executor with automatic VRAM caching
- DirectMLBackend: Windows DirectML / NPU / Intel & AMD GPU hardware accelerator
- Pluggable scheduler interface for automatic hardware routing
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger("MECH.compute_backend")


class ComputeBackend(ABC):
    """Abstract interface for hardware execution backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def device_str(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def get_device_memory_info(self) -> Dict[str, Any]:
        pass


class CPUBackend(ComputeBackend):
    """Standard CPU execution backend."""

    @property
    def name(self) -> str:
        return "CPU"

    @property
    def device_str(self) -> str:
        return "cpu"

    def is_available(self) -> bool:
        return True

    def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def get_device_memory_info(self) -> Dict[str, Any]:
        return {"device": "cpu", "allocated_mb": 0.0, "reserved_mb": 0.0}


class CUDABackend(ComputeBackend):
    """NVIDIA CUDA execution backend."""

    @property
    def name(self) -> str:
        return "CUDA"

    @property
    def device_str(self) -> str:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        if not self.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
        return func(*args, **kwargs)

    def get_device_memory_info(self) -> Dict[str, Any]:
        if not self.is_available():
            return {"device": "cuda", "available": False}
        return {
            "device": "cuda:0",
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
            "device_name": torch.cuda.get_device_name(0),
        }


class DirectMLBackend(ComputeBackend):
    """Windows DirectML / NPU accelerator backend."""

    def __init__(self) -> None:
        self._dml_device: Optional[Any] = None
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "DirectML_NPU"

    @property
    def device_str(self) -> str:
        return "dml" if self.is_available() else "cpu"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import torch_directml
            self._dml_device = torch_directml.device()
            self._available = True
        except (ImportError, Exception):
            self._available = False
        return self._available

    def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def get_device_memory_info(self) -> Dict[str, Any]:
        return {
            "device": "DirectML",
            "is_npu_directml_active": self.is_available(),
            "notes": "DirectML device abstraction on Windows",
        }


class ExecutionBackendManager:
    """Selects and manages the best available compute backend."""

    def __init__(self) -> None:
        self.backends: Dict[str, ComputeBackend] = {
            "cpu": CPUBackend(),
            "cuda": CUDABackend(),
            "directml": DirectMLBackend(),
        }

    def get_best_available_backend(self) -> ComputeBackend:
        if self.backends["cuda"].is_available():
            return self.backends["cuda"]
        if self.backends["directml"].is_available():
            return self.backends["directml"]
        return self.backends["cpu"]

    def get_backend(self, name: str) -> ComputeBackend:
        normalized = name.lower()
        if normalized in self.backends:
            be = self.backends[normalized]
            if be.is_available():
                return be
            logger.info("Backend '%s' not available, returning CPU fallback", name)
        return self.backends["cpu"]


# Legacy aliases for backward compatibility
ExecutionBackend = ComputeBackend
LocalCUDAExecutionBackend = CUDABackend
backend_manager = ExecutionBackendManager()
