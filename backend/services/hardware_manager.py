"""Hardware Detection, VRAM Estimation & OOM Recovery for MECH.

Estimates VRAM requirements before expensive tensor operations and catches CUDA OOM
gracefully to prevent crashing background workers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger("MECH.hardware_manager")


class ResourceLimitError(Exception):
    """Raised when an operation exceeds available GPU VRAM or memory limit."""


class HardwareManager:
    """Monitors GPU devices, computes memory estimates, and guards execution."""

    @staticmethod
    def get_hardware_status() -> Dict[str, Any]:
        """Returns live hardware profile, CUDA availability, and memory stats."""
        cuda_available = torch.cuda.is_available()
        device_name = "CPU"
        total_vram_bytes = 0
        allocated_vram_bytes = 0
        free_vram_bytes = 0

        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
            allocated_vram_bytes = torch.cuda.memory_allocated(0)
            free_vram_bytes = total_vram_bytes - allocated_vram_bytes

        return {
            "cuda_available": cuda_available,
            "device_name": device_name,
            "device_type": "cuda" if cuda_available else "cpu",
            "total_vram_mb": round(total_vram_bytes / (1024 * 1024), 1),
            "free_vram_mb": round(free_vram_bytes / (1024 * 1024), 1),
            "allocated_vram_mb": round(allocated_vram_bytes / (1024 * 1024), 1),
        }

    @staticmethod
    def estimate_vram_requirement(
        model_name: str = "gpt2",
        batch_size: int = 1,
        seq_len: int = 128,
        dtype_bytes: int = 4,  # float32 default
    ) -> Dict[str, Any]:
        """Calculates VRAM requirement before loading or executing interventions."""
        model_params = {
            "gpt2": 124_439_808,
            "gpt2-medium": 354_823_168,
            "gpt2-large": 774_030_080,
            "gpt2-xl": 1_557_611_200,
        }.get(model_name.lower(), 124_439_808)

        # Weight footprint
        weights_mb = (model_params * dtype_bytes) / (1024 * 1024)

        # Activation footprint per forward pass (estimated across 12 layers)
        n_layers = 12 if "medium" not in model_name else 24
        hidden_dim = 768 if "medium" not in model_name else 1024
        act_bytes = batch_size * seq_len * hidden_dim * n_layers * dtype_bytes * 4
        activations_mb = act_bytes / (1024 * 1024)

        # Safety overhead (250MB buffer)
        overhead_mb = 250.0
        total_required_mb = weights_mb + activations_mb + overhead_mb

        hw = HardwareManager.get_hardware_status()
        can_fit_gpu = hw["cuda_available"] and (hw["free_vram_mb"] > total_required_mb)

        remediation_options = []
        if not can_fit_gpu and hw["cuda_available"]:
            remediation_options = [
                "Reduce prompt length (seq_len)",
                "Switch to CPU execution mode",
                "Execute with 16-bit precision (bfloat16/fp16)",
                "Use gradient-free evaluation",
            ]

        return {
            "model_name": model_name,
            "estimated_weights_mb": round(weights_mb, 1),
            "estimated_activations_mb": round(activations_mb, 1),
            "total_required_mb": round(total_required_mb, 1),
            "available_vram_mb": hw["free_vram_mb"],
            "can_execute_on_gpu": can_fit_gpu,
            "recommended_device": "cuda" if can_fit_gpu else "cpu",
            "remediation_options": remediation_options,
        }

    @staticmethod
    def execute_with_oom_guard(fn, *args, **kwargs):
        """Executes a tensor function, catching CUDA OOM and freeing memory without crashing."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            err_str = str(exc).lower()
            if "out of memory" in err_str or (hasattr(torch.cuda, "OutOfMemoryError") and isinstance(exc, torch.cuda.OutOfMemoryError)):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.error("Caught CUDA OOM cleanly in execution worker: %s", exc)
                raise ResourceLimitError(f"GPU Out of Memory: {exc}") from exc
            raise


# Global hardware manager singleton
hardware_manager = HardwareManager()
