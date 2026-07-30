"""Capability Registry.

Exposes central capability queries (supports_sae, supports_patching, supports_cluster, supports_cloud, supports_llama).
"""

from __future__ import annotations

from typing import Dict, Set


class CapabilityRegistry:
    """Central registry querying system and plugin capabilities."""

    def __init__(self) -> None:
        self._capabilities: Set[str] = {
            "sae",
            "patching",
            "cluster",
            "cloud",
            "llama",
            "gemma",
            "qwen",
            "mistral",
            "deepseek",
            "visualization",
            "validation",
            "reproducibility",
        }

    def supports(self, capability: str) -> bool:
        return capability.lower() in self._capabilities

    def register_capability(self, capability: str) -> None:
        self._capabilities.add(capability.lower())

    def list_capabilities(self) -> Dict[str, bool]:
        return {cap: True for cap in sorted(self._capabilities)}
