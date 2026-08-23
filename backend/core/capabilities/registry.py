"""Capability Registry for MECH Platform."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from backend.core.capabilities.schema import CapabilitySpec, get_category_description, list_categories

logger = logging.getLogger("MECH.capabilities")


class CapabilityRegistry:
    """Registry for system capabilities."""

    def __init__(self) -> None:
        self._capabilities: Dict[str, CapabilitySpec] = {}
        self._capability_tools: Dict[str, Set[str]] = {}  # capability -> tool names
        self._tool_capabilities: Dict[str, Set[str]] = {}  # tool -> capability names

    def register(self, spec: CapabilitySpec) -> None:
        """Register a capability specification."""
        if spec.name in self._capabilities:
            logger.warning("Capability '%s' already registered, overwriting", spec.name)
        self._capabilities[spec.name] = spec
        logger.debug("Registered capability: %s (category: %s)", spec.name, spec.category)

    def unregister(self, name: str) -> bool:
        """Unregister a capability."""
        if name in self._capabilities:
            del self._capabilities[name]
            logger.debug("Unregistered capability: %s", name)
            return True
        return False

    def get(self, name: str) -> Optional[CapabilitySpec]:
        """Get a capability by name."""
        return self._capabilities.get(name)

    def supports(self, capability: str) -> bool:
        """Check if a capability is supported."""
        return capability.lower() in {c.lower() for c in self._capabilities}

    def list_capabilities(self, category: Optional[str] = None) -> List[CapabilitySpec]:
        """List all capabilities, optionally filtered by category."""
        caps = list(self._capabilities.values())
        if category:
            caps = [c for c in caps if c.category == category]
        return sorted(caps, key=lambda c: c.name)

    def list_capability_dicts(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List capabilities as dictionaries."""
        return [c.to_dict() for c in self.list_capabilities(category)]

    def list_categories(self) -> Dict[str, str]:
        """List all capability categories."""
        return list_categories()

    def get_capabilities_for_tool(self, tool_name: str) -> List[str]:
        """Get capabilities provided by a tool."""
        return list(self._tool_capabilities.get(tool_name, set()))

    def get_tools_for_capability(self, capability: str) -> List[str]:
        """Get tools that provide a capability."""
        return list(self._capability_tools.get(capability.lower(), set()))

    def register_tool_capabilities(self, tool_name: str, capabilities: List[str]) -> None:
        """Register which capabilities a tool provides."""
        for cap in capabilities:
            cap_lower = cap.lower()
            self._capability_tools.setdefault(cap_lower, set()).add(tool_name)
            self._tool_capabilities.setdefault(tool_name, set()).add(cap_lower)

    def clear(self) -> None:
        """Clear all capabilities."""
        self._capabilities.clear()
        self._capability_tools.clear()
        self._tool_capabilities.clear()


# Global registry instance
_global_registry: Optional[CapabilityRegistry] = None


def get_capability_registry() -> CapabilityRegistry:
    """Get the global capability registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CapabilityRegistry()
    return _global_registry


def reset_capability_registry() -> None:
    """Reset the global capability registry (for testing)."""
    global _global_registry
    _global_registry = None