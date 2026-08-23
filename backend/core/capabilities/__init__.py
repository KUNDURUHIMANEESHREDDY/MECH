"""Capability Definition Module for MECH Platform."""

from backend.core.capabilities.builtins import (
    BUILTIN_CAPABILITIES,
    get_builtin_capability,
    list_builtin_capabilities,
    register_builtin_capabilities,
)
from backend.core.capabilities.discovery import (
    auto_discover_all,
    discover_capabilities_from_plugins,
    get_capability_summary,
    sync_tool_capabilities,
)
from backend.core.capabilities.registry import CapabilityRegistry, get_capability_registry, reset_capability_registry
from backend.core.capabilities.schema import (
    CAPABILITY_CATEGORIES,
    CapabilitySpec,
    get_category_description,
    list_categories,
)

__all__ = [
    # Schema
    "CapabilitySpec",
    "CAPABILITY_CATEGORIES",
    "get_category_description",
    "list_categories",
    # Registry
    "CapabilityRegistry",
    "get_capability_registry",
    "reset_capability_registry",
    # Built-ins
    "BUILTIN_CAPABILITIES",
    "get_builtin_capability",
    "list_builtin_capabilities",
    "register_builtin_capabilities",
    # Discovery
    "auto_discover_all",
    "discover_capabilities_from_plugins",
    "sync_tool_capabilities",
    "get_capability_summary",
]