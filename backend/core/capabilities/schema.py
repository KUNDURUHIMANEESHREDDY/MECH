"""Capability Schema for MECH Platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from backend.core.plugins.base import ToolManifest


@dataclass
class CapabilitySpec:
    """Specification for a capability."""

    name: str
    description: str
    version: str = "1.0.0"
    category: str = "general"
    dependencies: List[str] = field(default_factory=list)
    provides_tools: List[str] = field(default_factory=list)
    required_models: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "dependencies": self.dependencies,
            "provides_tools": self.provides_tools,
            "required_models": self.required_models,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilitySpec":
        return cls(**data)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CapabilitySpec):
            return False
        return self.name == other.name


# Capability categories
CAPABILITY_CATEGORIES = {
    "localization": "Layer/component localization and attribution",
    "causal": "Causal intervention and mediation analysis",
    "dictionary_learning": "SAE and dictionary learning methods",
    "circuits": "Circuit discovery and analysis",
    "verification": "Scientific validation and falsification",
    "comparative": "Cross-model comparison and alignment",
    "redundancy": "Backup heads and self-repair analysis",
    "general": "General platform capabilities",
}


def get_category_description(category: str) -> str:
    """Get description for a capability category."""
    return CAPABILITY_CATEGORIES.get(category, "Unknown category")


def list_categories() -> Dict[str, str]:
    """List all capability categories with descriptions."""
    return CAPABILITY_CATEGORIES.copy()