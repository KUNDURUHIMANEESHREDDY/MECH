"""Epic 6: Performance Profiler — collect timing, memory, GPU metrics.

Generates a structured performance timeline with per-phase breakdowns:

    Load → Inference → Hooks → Cache → Serialization → API
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import psutil


PHASES = ["load", "inference", "hooks", "cache", "serialization", "api"]


@dataclass
class MetricSample:
    name: str
    duration: float
    cpu_percent: float
    memory_mb: float
    gpu_available: bool
    gpu_memory_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Profiler:
    """Collects metrics across named operations with phase tracking."""

    def __init__(self):
        self._samples: list[MetricSample] = []

    @contextmanager
    def measure(self, name: str, **metadata):
        start = time.perf_counter()
        start_cpu = psutil.cpu_percent(interval=None)
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        start_gpu_mem = 0.0
        if torch.cuda.is_available():
            start_gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)

        try:
            yield
        finally:
            duration = time.perf_counter() - start
            end_cpu = psutil.cpu_percent(interval=None)
            end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            end_gpu_mem = 0.0
            if torch.cuda.is_available():
                end_gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)

            sample = MetricSample(
                name=name,
                duration=duration,
                cpu_percent=(start_cpu + end_cpu) / 2,
                memory_mb=(start_mem + end_mem) / 2,
                gpu_available=torch.cuda.is_available(),
                gpu_memory_mb=(start_gpu_mem + end_gpu_mem) / 2,
                metadata=metadata,
            )
            self._samples.append(sample)

    def report(self) -> dict:
        if not self._samples:
            return {"samples": [], "total_duration_ms": 0.0, "phases": {}, "timeline": []}

        total = sum(s.duration for s in self._samples)
        breakdown: dict[str, float] = {}
        timeline: list[dict] = []

        for s in self._samples:
            phase = self._classify_phase(s.name)
            breakdown.setdefault(phase, 0.0)
            breakdown[phase] += s.duration
            timeline.append({
                "phase": phase,
                "name": s.name,
                "duration_ms": round(s.duration * 1000, 2),
                "cpu_percent": s.cpu_percent,
                "memory_mb": round(s.memory_mb, 1),
                "gpu_memory_mb": round(s.gpu_memory_mb, 1),
                "metadata": s.metadata,
            })

        return {
            "samples": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration * 1000, 2),
                    "cpu_percent": s.cpu_percent,
                    "memory_mb": round(s.memory_mb, 1),
                    "gpu_memory_mb": round(s.gpu_memory_mb, 1),
                    "metadata": s.metadata,
                }
                for s in self._samples
            ],
            "total_duration_ms": round(total * 1000, 2),
            "phases": {k: round(v * 1000, 2) for k, v in sorted(breakdown.items())},
            "timeline": timeline,
        }

    def _classify_phase(self, name: str) -> str:
        for ph in PHASES:
            if name.startswith(ph):
                return ph
        return "other"

    def clear(self) -> None:
        self._samples.clear()
