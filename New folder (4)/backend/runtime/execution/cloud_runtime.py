"""Cloud Runtime & Provider Integration.

Provides unified cloud provider interfaces for AWS, Azure, GCP, RunPod, and Lambda Labs.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class CloudProviderDTO:
    """Standardized Cloud Provider DTO reporting GPU, VRAM, cost, and latency."""

    def __init__(
        self,
        name: str,
        gpu_type: str,
        vram_gb: int,
        hourly_cost: float,
        latency_ms: float,
        available: bool = True,
    ) -> None:
        self.name = name
        self.gpu_type = gpu_type
        self.vram_gb = vram_gb
        self.hourly_cost = hourly_cost
        self.latency_ms = latency_ms
        self.available = available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "gpu_type": self.gpu_type,
            "vram_gb": self.vram_gb,
            "hourly_cost": self.hourly_cost,
            "latency_ms": self.latency_ms,
            "available": self.available,
        }


class CloudRuntimeManager:
    """Manager for multi-cloud runtime providers."""

    def __init__(self) -> None:
        self.providers: Dict[str, CloudProviderDTO] = {
            "AWS": CloudProviderDTO("AWS", "NVIDIA H100", 80, 4.10, 12.0),
            "Azure": CloudProviderDTO("Azure", "NVIDIA A100", 80, 3.85, 14.5),
            "GCP": CloudProviderDTO("GCP", "NVIDIA A100", 40, 2.93, 11.2),
            "RunPod": CloudProviderDTO("RunPod", "NVIDIA RTX 4090", 24, 0.69, 18.0),
            "Lambda": CloudProviderDTO("Lambda", "NVIDIA H100", 80, 2.49, 15.0),
        }

    def list_providers(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.providers.values()]

    def get_provider(self, name: str) -> Dict[str, Any] | None:
        provider = self.providers.get(name)
        return provider.to_dict() if provider else None
