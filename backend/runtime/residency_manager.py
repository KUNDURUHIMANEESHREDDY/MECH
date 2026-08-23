"""3-Tier Tensor Residency and Memory Eviction Manager.

Coordinates disk storage (mmap/safetensors), host CPU RAM, and GPU VRAM tiers
with explicit tensor pinning, prefetching, and LRU eviction policies.
"""

from __future__ import annotations

import collections
import concurrent.futures
import gc
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import psutil
import torch
import torch.nn as nn

from .activation_artifact import ActivationArtifact


class DeviceTier(str, Enum):
    DISK = "disk"
    CPU_RAM = "cpu_ram"
    GPU_VRAM = "gpu_vram"


class EnforcementMode(str, Enum):
    STRICT = "strict"
    BEST_EFFORT = "best_effort"


class ViolationAction(str, Enum):
    EVICT = "evict"
    SPILL = "spill"
    ABORT = "abort"


class MemoryBudgetExceededError(RuntimeError):
    """Raised when strict memory budget enforcement fails to keep memory within budget."""
    pass


@dataclass
class MemoryBudgetPolicy:
    """Explicit hardware-level memory budget and proactive enforcement policy."""
    ram_limit_bytes: int = 16 * 1024 * 1024 * 1024  # 16 GB
    vram_limit_bytes: int = 4 * 1024 * 1024 * 1024   # 4 GB
    activation_limit_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GB
    weight_limit_bytes: int = 8 * 1024 * 1024 * 1024      # 8 GB
    safety_margin_bytes: int = 64 * 1024 * 1024          # 64 MB
    enforcement_mode: EnforcementMode = EnforcementMode.STRICT
    violation_action: ViolationAction = ViolationAction.EVICT


@dataclass
class LayerResidencyState:
    """Tracks physical location of layer weights."""
    layer_index: int
    current_tier: DeviceTier
    module_ref: Optional[nn.Module]
    is_pinned: bool = False
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0


class ResidencyManager:
    """Manages multi-tier model parameter and activation tensor residency with asynchronous prefetching and strict budget enforcement."""

    def __init__(
        self,
        vram_budget_mb: float = 4096.0,
        ram_budget_mb: float = 16384.0,
        max_active_layers: int = 3,
        cache_dir: Optional[Path | str] = None,
        default_compute_device: str = "cpu",
        budget_policy: Optional[MemoryBudgetPolicy] = None,
    ) -> None:
        self.max_active_layers = max(1, max_active_layers)
        self.cache_dir = Path(cache_dir or Path(__file__).resolve().parent.parent.parent / "storage" / "activations_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.default_compute_device = default_compute_device
        self.peak_active_layers_observed: int = 0

        # Memory Budget Policy
        if budget_policy is not None:
            self.policy = budget_policy
        else:
            self.policy = MemoryBudgetPolicy(
                ram_limit_bytes=int(ram_budget_mb * 1024 * 1024),
                vram_limit_bytes=int(vram_budget_mb * 1024 * 1024),
                enforcement_mode=EnforcementMode.STRICT,
                violation_action=ViolationAction.EVICT,
            )

        self.vram_budget_bytes = self.policy.vram_limit_bytes
        self.ram_budget_bytes = self.policy.ram_limit_bytes

        # Physical telemetry trackers
        self._process = psutil.Process()
        self.peak_rss_bytes: int = self._process.memory_info().rss
        self.peak_vram_bytes: int = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        self.total_load_time_sec: float = 0.0
        self.total_eviction_time_sec: float = 0.0
        self.total_compute_time_sec: float = 0.0
        self.prefetch_requests: int = 0
        self.prefetch_hits: int = 0
        self.prefetch_misses: int = 0

        # Async prefetch worker
        self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._prefetch_futures: Dict[int, concurrent.futures.Future] = {}

        # Layer residency table: layer_idx -> LayerResidencyState
        self._layers: Dict[int, LayerResidencyState] = {}
        # Activation cache: artifact_id -> ActivationArtifact
        self._activations: Dict[str, ActivationArtifact] = collections.OrderedDict()
        
        # Loader callbacks for layer modules: layer_idx -> Callable[[], nn.Module]
        self._layer_loaders: Dict[int, Callable[[], nn.Module]] = {}



    def register_layer_loader(self, layer_idx: int, loader_fn: Callable[[], nn.Module], size_bytes: int = 0) -> None:
        """Registers a demand-page loader for a specific transformer layer."""
        self._layer_loaders[layer_idx] = loader_fn
        self._layers[layer_idx] = LayerResidencyState(
            layer_index=layer_idx,
            current_tier=DeviceTier.DISK,
            module_ref=None,
            is_pinned=False,
            size_bytes=size_bytes,
        )

    def sample_memory(self) -> None:
        """Samples physical process RSS and CUDA VRAM to track peak memory."""
        rss = self._process.memory_info().rss
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated()
            if vram > self.peak_vram_bytes:
                self.peak_vram_bytes = vram

    def check_and_enforce_budget(self) -> None:
        """Actively checks physical memory pressure against policy and executes violation actions."""
        self.sample_memory()
        
        # Check RAM pressure
        ram_headroom = self.policy.ram_limit_bytes - self.peak_rss_bytes
        if ram_headroom < self.policy.safety_margin_bytes:
            if self.policy.violation_action in (ViolationAction.SPILL, ViolationAction.EVICT):
                # 1. Spill in-memory activation artifacts to disk
                for art in list(self._activations.values()):
                    if art.current_location != "disk":
                        art.spill_to_disk(self.cache_dir)
                # 2. Evict unpinned layers
                for s in list(self._layers.values()):
                    if s.module_ref is not None and not s.is_pinned:
                        self.release_layer(s.layer_index)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.sample_memory()

            if self.policy.enforcement_mode == EnforcementMode.STRICT and self.policy.violation_action == ViolationAction.ABORT:
                if self.peak_rss_bytes > self.policy.ram_limit_bytes:
                    raise MemoryBudgetExceededError(
                        f"Strict RAM budget exceeded: peak RSS {self.peak_rss_bytes / (1024*1024):.1f} MB > "
                        f"limit {self.policy.ram_limit_bytes / (1024*1024):.1f} MB."
                    )

        # Check VRAM pressure if CUDA is present
        if torch.cuda.is_available():
            vram_headroom = self.policy.vram_limit_bytes - self.peak_vram_bytes
            if vram_headroom < self.policy.safety_margin_bytes:
                if self.policy.violation_action in (ViolationAction.SPILL, ViolationAction.EVICT):
                    for s in list(self._layers.values()):
                        if s.module_ref is not None and not s.is_pinned:
                            self.release_layer(s.layer_index)
                    torch.cuda.empty_cache()
                    self.sample_memory()

                if self.policy.enforcement_mode == EnforcementMode.STRICT and self.policy.violation_action == ViolationAction.ABORT:
                    if self.peak_vram_bytes > self.policy.vram_limit_bytes:
                        raise MemoryBudgetExceededError(
                            f"Strict VRAM budget exceeded: peak VRAM {self.peak_vram_bytes / (1024*1024):.1f} MB > "
                            f"limit {self.policy.vram_limit_bytes / (1024*1024):.1f} MB."
                        )

    def prefetch_layer(self, layer_idx: int, target_device: Optional[str] = None) -> None:
        """Asynchronously triggers background loading/staging of next layer from disk."""
        self.prefetch_requests += 1
        if layer_idx not in self._layers or layer_idx in self._prefetch_futures:
            return

        state = self._layers[layer_idx]
        if state.module_ref is not None:
            return  # Already resident

        loader = self._layer_loaders.get(layer_idx)
        if loader is None:
            return

        dev = target_device or self.default_compute_device

        def _async_load():
            t0 = time.perf_counter()
            mod = loader()
            mod = mod.to(dev)
            t_dur = time.perf_counter() - t0
            return mod, t_dur

        self._prefetch_futures[layer_idx] = self._prefetch_executor.submit(_async_load)

    def load_layer_to_device(
        self,
        layer_idx: int,
        target_device: Optional[str] = None,
        pin: bool = False,
    ) -> nn.Module:
        """Demand-pages a layer into target device, utilizing prefetched module if available."""
        self.check_and_enforce_budget()
        t_start = time.perf_counter()
        dev = target_device or self.default_compute_device
        target_tier = DeviceTier.GPU_VRAM if "cuda" in dev else DeviceTier.CPU_RAM

        state = self._layers.get(layer_idx)
        if state is None:
            raise KeyError(f"Layer {layer_idx} not registered with ResidencyManager.")

        # Check if already loaded to requested tier
        if state.module_ref is not None and state.current_tier == target_tier:
            state.last_accessed = time.time()
            if pin:
                state.is_pinned = True
            self.sample_memory()
            return state.module_ref

        # Evict unpinned layers if memory pressure threshold reached
        self._enforce_memory_limits(target_tier)

        # Check if there is an in-flight or completed prefetch future
        if layer_idx in self._prefetch_futures:
            fut = self._prefetch_futures.pop(layer_idx)
            module, load_dur = fut.result()
            self.prefetch_hits += 1
            self.total_load_time_sec += load_dur
        elif state.module_ref is not None:
            module = state.module_ref
            module = module.to(dev)
            self.prefetch_misses += 1
            self.total_load_time_sec += (time.perf_counter() - t_start)
        else:
            loader = self._layer_loaders.get(layer_idx)
            if loader is None:
                raise ValueError(f"No loader registered for layer {layer_idx}.")
            module = loader()
            module = module.to(dev)
            self.prefetch_misses += 1
            self.total_load_time_sec += (time.perf_counter() - t_start)

        state.module_ref = module
        state.current_tier = target_tier
        state.last_accessed = time.time()
        if pin:
            state.is_pinned = True

        current_active = sum(1 for s in self._layers.values() if s.module_ref is not None)
        if current_active > self.peak_active_layers_observed:
            self.peak_active_layers_observed = current_active

        self.sample_memory()
        return module

    def release_layer(self, layer_idx: int, force: bool = False) -> None:
        """Evicts a layer module from active device back to disk/unloaded state."""
        t_start = time.perf_counter()
        state = self._layers.get(layer_idx)
        if state is None or state.module_ref is None:
            return

        if state.is_pinned and not force:
            return

        # Drop reference to allow GC
        del state.module_ref
        state.module_ref = None
        state.current_tier = DeviceTier.DISK
        state.is_pinned = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        self.total_eviction_time_sec += (time.perf_counter() - t_start)
        self.sample_memory()

    def reset_peak_stats(self) -> None:
        """Resets the peak active layers and memory high-water mark."""
        self.peak_active_layers_observed = sum(1 for s in self._layers.values() if s.module_ref is not None)
        self.peak_rss_bytes = self._process.memory_info().rss
        self.peak_vram_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        self.total_load_time_sec = 0.0
        self.total_eviction_time_sec = 0.0
        self.total_compute_time_sec = 0.0
        self.prefetch_requests = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0

    def store_activation(self, artifact: ActivationArtifact, pin_in_ram: bool = False) -> str:
        """Stores an intermediate activation artifact with LRU tracking and proactive budget checking."""
        self.check_and_enforce_budget()
        self._activations[artifact.artifact_id] = artifact
        if not pin_in_ram and len(self._activations) > 32:
            # Spill older activation artifacts to disk
            oldest_id, oldest_art = next(iter(self._activations.items()))
            if oldest_art.current_location != "disk":
                oldest_art.spill_to_disk(self.cache_dir)
        self.sample_memory()
        return artifact.artifact_id

    def retrieve_activation(self, artifact_id: str, target_device: Optional[str] = None) -> torch.Tensor:
        """Retrieves an activation tensor, restoring from disk if spilled."""
        artifact = self._activations.get(artifact_id)
        if artifact is None:
            raise KeyError(f"Activation artifact '{artifact_id}' not found.")
        # Mark recently used in OrderedDict
        self._activations.move_to_end(artifact_id)
        tensor = artifact.get_tensor(target_device or self.default_compute_device)
        self.sample_memory()
        return tensor

    def _enforce_memory_limits(self, target_tier: DeviceTier) -> None:
        """Evicts unpinned layers when active count reaches or exceeds max_active_layers."""
        active_unpinned = [
            s for s in self._layers.values()
            if s.module_ref is not None and not s.is_pinned
        ]
        while len(active_unpinned) >= self.max_active_layers:
            active_unpinned.sort(key=lambda s: s.last_accessed)
            oldest = active_unpinned.pop(0)
            self.release_layer(oldest.layer_index)

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Returns comprehensive statistics on active tensor residency, budget policy, and storage usage."""
        self.sample_memory()
        active_vram_layers = [s.layer_index for s in self._layers.values() if s.current_tier == DeviceTier.GPU_VRAM]
        active_ram_layers = [s.layer_index for s in self._layers.values() if s.current_tier == DeviceTier.CPU_RAM]
        
        total_io_sec = self.total_load_time_sec + self.total_eviction_time_sec
        total_time_sec = total_io_sec + self.total_compute_time_sec
        compute_utilization = (self.total_compute_time_sec / total_time_sec * 100.0) if total_time_sec > 0 else 100.0
        io_stall = (total_io_sec / total_time_sec * 100.0) if total_time_sec > 0 else 0.0
        
        hit_rate = (self.prefetch_hits / max(self.prefetch_requests, 1)) * 100.0

        return {
            "policy": {
                "enforcement_mode": self.policy.enforcement_mode.value,
                "violation_action": self.policy.violation_action.value,
                "ram_limit_mb": round(self.policy.ram_limit_bytes / (1024 * 1024), 2),
                "vram_limit_mb": round(self.policy.vram_limit_bytes / (1024 * 1024), 2),
                "safety_margin_mb": round(self.policy.safety_margin_bytes / (1024 * 1024), 2),
            },
            "vram_budget_mb": self.vram_budget_bytes / (1024 * 1024),
            "ram_budget_mb": self.ram_budget_bytes / (1024 * 1024),
            "max_active_layers": self.max_active_layers,
            "peak_active_layers_observed": self.peak_active_layers_observed,
            "peak_rss_mb": round(self.peak_rss_bytes / (1024 * 1024), 2),
            "peak_vram_mb": round(self.peak_vram_bytes / (1024 * 1024), 2),
            "total_load_time_ms": round(self.total_load_time_sec * 1000, 2),
            "total_eviction_time_ms": round(self.total_eviction_time_sec * 1000, 2),
            "total_compute_time_ms": round(self.total_compute_time_sec * 1000, 2),
            "prefetch_requests": self.prefetch_requests,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "prefetch_hit_rate_pct": round(hit_rate, 2),
            "compute_utilization_pct": round(compute_utilization, 2),
            "io_stall_pct": round(io_stall, 2),
            "active_vram_layers": active_vram_layers,
            "active_ram_layers": active_ram_layers,
            "total_registered_layers": len(self._layers),
            "total_cached_activations": len(self._activations),
            "disk_cache_dir": str(self.cache_dir),
            "memory_budget_compliant": (
                (self.peak_rss_bytes <= self.ram_budget_bytes) and
                (self.peak_vram_bytes <= self.vram_budget_bytes)
            ),
        }



