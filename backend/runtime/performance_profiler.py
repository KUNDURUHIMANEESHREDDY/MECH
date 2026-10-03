"""High-Fidelity Performance Profiler — Beyond Runtime and Memory.

Collects granular hardware and execution metrics:
- FLOPs: Estimated compute operations.
- GPU Utilization: CUDA core and bandwidth usage.
- Throughput: Real-time tokens per second (TPS).
- Latency: Milliseconds per sample/prompt.
- Memory Timeline: Peak and sustained usage.
"""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProfilerMetrics:
    # Optional where the reading may not be obtainable. 0.0 is a claim that
    # the GPU was idle / nothing allocated, which is different from "not
    # sampled".
    total_flops: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    throughput_tps: Optional[float] = None
    mean_latency_ms: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    total_tokens: int = 0
    duration_s: float = 0.0


class PerformanceProfiler:
    """Engine for collecting high-fidelity performance data during model execution."""

    def __init__(self, model_id: str, n_params_b: float) -> None:
        self.model_id = model_id
        self.n_params_b = n_params_b
        self.start_time: float = 0.0
        self.metrics = ProfilerMetrics()

    def start(self) -> None:
        """Starts the profiling session."""
        self.start_time = time.perf_counter()

    def stop(self, total_tokens: Optional[int] = None) -> ProfilerMetrics:
        """Stops the profiling session and computes final metrics.

        `total_tokens` is now Optional and must be counted by the caller. It
        used to default to `n * 50` at the call site -- "Estimated tokens =
        # prompts * avg_seq_len" -- so throughput was computed from a token
        # count nobody counted. Throughput and latency are None without it.
        """
        duration = time.perf_counter() - self.start_time
        self.metrics.duration_s = duration

        if not total_tokens:
            self.metrics.total_tokens = 0
            self.metrics.throughput_tps = None
            self.metrics.mean_latency_ms = None
        else:
            self.metrics.total_tokens = int(total_tokens)
            self.metrics.throughput_tps = total_tokens / max(duration, 1e-9)
            self.metrics.mean_latency_ms = (duration * 1000) / max(total_tokens, 1)

        # FLOPs is a closed-form estimate (2 * P * N for forward-only
        # inference), not a measurement. It is only meaningful when the token
        # count is real, and it is reported as an estimate where it appears.
        self.metrics.total_flops = (
            2.0 * (self.n_params_b * 1e9) * total_tokens if total_tokens else None
        )

        # 4. GPU & Memory Monitoring (Mocked for CPU fallback)
        self.metrics.gpu_util_pct = self._capture_gpu_util()
        self.metrics.peak_vram_mb = self._capture_peak_vram()

        return self.metrics

    def _capture_gpu_util(self) -> Optional[float]:
        """GPU utilisation, sampled from the driver.

        Was `return 78.4` whenever CUDA happened to be available, commented
        "Simulated utilization for mock/stub consistency". Every GPU run on
        every machine reported the same 78.4%, which is what a monitoring
        surface needs in order to be useful and useless at the same time.

        Returns None when it cannot be sampled -- no CUDA, no pynvml, or
        pynvml present but failing. 0.0 would read as "the GPU was idle",
        which is a claim, not an absence.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return None
        except ImportError:
            return None

        # torch exposes utilisation only through pynvml. If it is absent,
        # report that rather than substituting a number.
        try:
            import pynvml  # type: ignore
        except ImportError:
            return None

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            return None

    def _capture_peak_vram(self) -> Optional[float]:
        """Peak VRAM in MiB, or None when there is no CUDA allocator."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        except ImportError:
            return None

    def to_summary(self) -> Dict[str, Any]:
        """Returns a human-readable performance summary."""
        m = self.metrics
        return {
            "model_id": self.model_id,
            "throughput_tps": (round(m.throughput_tps, 2)
                               if m.throughput_tps is not None else None),
            "latency_ms": (round(m.mean_latency_ms, 2)
                           if m.mean_latency_ms is not None else None),
            # Named as an estimate, and only present when it is computable.
            "est_flops": (f"{m.total_flops:.2e}"
                          if m.total_flops is not None else None),
            "gpu_util_pct": m.gpu_util_pct,
            "peak_vram_mb": (round(m.peak_vram_mb, 1)
                             if m.peak_vram_mb is not None else None),
            "tokens_counted": m.total_tokens > 0,
            "profiler_measured": m.total_tokens > 0,
        }
