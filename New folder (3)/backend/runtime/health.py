"""Epic 10: Health Monitor — system health and resource monitoring.

Exposes active sessions, loaded hooks, cache memory, events/sec
in addition to standard CPU/GPU/RAM metrics.
"""

from __future__ import annotations

import os
import platform
import sys

import torch
import psutil

from .activation_cache import cache
from .model_manager import _loaded_name
from .event_bus import bus
from .hook_framework import HOOK_TYPES


def get_health() -> dict:
    mem = psutil.virtual_memory()
    process = psutil.Process()

    health = {
        "status": "ok",
        "timestamp": __import__("time").time(),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
        "cpu": {
            "cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "percent": psutil.cpu_percent(interval=None),
            "process_percent": process.cpu_percent(interval=None),
        },
        "ram": {
            "total_mb": round(mem.total / (1024 * 1024), 1),
            "available_mb": round(mem.available / (1024 * 1024), 1),
            "used_mb": round(mem.used / (1024 * 1024), 1),
            "percent": mem.percent,
            "process_mb": round(process.memory_info().rss / (1024 * 1024), 1),
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "runtime": {
            "model_loaded": _loaded_name is not None,
            "model_name": _loaded_name or "",
            "cache_entries": cache.count(),
            "cache_memory_estimate_mb": _estimate_cache_memory(),
            "active_sessions": 0,
            "loaded_hooks": len(HOOK_TYPES),
            "events_logged": bus.log.count,
            "events_per_sec": round(bus.log.events_per_sec, 1),
        },
    }

    # Add active sessions count
    try:
        from .session_manager import session_manager
        health["runtime"]["active_sessions"] = session_manager.count()
    except Exception:
        pass

    if torch.cuda.is_available():
        health["gpu"] = {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1),
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 1),
            "cached_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 1),
            "utilization_percent": _estimate_gpu_util(),
        }
        health["pytorch"]["cuda_version"] = torch.version.cuda
    else:
        health["gpu"] = {"available": False}

    return health


def _estimate_cache_memory() -> float:
    """Rough estimate of cache memory based on entry count."""
    return round(cache.count() * 0.5, 1)  # ~0.5 MB per cached activation


def _estimate_gpu_util() -> float:
    """Return a rough GPU utilization estimate."""
    try:
        return round(torch.cuda.utilization(), 1)
    except Exception:
        return 0.0
