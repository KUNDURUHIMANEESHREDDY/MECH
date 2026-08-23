"""Capability Discovery for MECH Platform - Auto-discovers capabilities from plugins."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from backend.core.capabilities.registry import CapabilityRegistry, get_capability_registry, reset_capability_registry
from backend.core.capabilities.schema import CapabilitySpec
from backend.core.plugins import get_registry as get_plugin_registry, load_all_plugins, reset_registry
from backend.core.plugins.base import Plugin, PluginManifest

logger = logging.getLogger("MECH.capabilities.discovery")


# Mapping from plugin names to capability names
PLUGIN_CAPABILITY_MAP = {
    "logit_lens": ["logit_lens", "localization"],
    "activation_patching": ["patching", "causal_tracing"],
    "sae_features": ["sae", "dictionary_learning"],
    "circuit_discovery": ["acdc_circuits", "circuits"],
    "path_patching": ["patching", "causal_tracing"],
    "hallucination_experiment": ["hallucination", "causal_tracing"],
    "semantic_falsification": ["falsification", "verification"],
    "circuit_metrics": ["metrics", "verification"],
    "live_intervention": ["live_intervention", "causal_tracing"],
    "scientific_validation": ["scientific_validation", "verification"],
    "cross_model_universality": ["cross_model", "universality", "comparative"],
    "backup_circuits": ["backup_heads", "redundancy"],
}


def discover_capabilities_from_plugins(
    plugin_registry=None,
    capability_registry: Optional[CapabilityRegistry] = None,
) -> List[str]:
    """Discover capabilities from loaded plugins."""
    if plugin_registry is None:
        plugin_registry = get_plugin_registry()

    if capability_registry is None:
        capability_registry = get_capability_registry()

    discovered = []

    for plugin_name in plugin_registry._plugins:
        capabilities = PLUGIN_CAPABILITY_MAP.get(plugin_name, [])
        for cap_name in capabilities:
            # Register capability if not already present
            if not capability_registry.supports(cap_name):
                # Create a basic capability spec from plugin
                plugin = plugin_registry.get_plugin(plugin_name)
                if plugin:
                    spec = CapabilitySpec(
                        name=cap_name,
                        description=f"Capability provided by {plugin_name} plugin",
                        category=plugin.manifest.capabilities[0] if plugin.manifest.capabilities else "general",
                        provides_tools=[t.name for t in plugin.list_tools()],
                        tags=plugin.manifest.tags,
                    )
                    capability_registry.register(spec)
                    discovered.append(cap_name)

    return discovered


def sync_tool_capabilities(
    plugin_registry=None,
    capability_registry: Optional[CapabilityRegistry] = None,
) -> None:
    """Sync tool-capability relationships from plugins."""
    if plugin_registry is None:
        plugin_registry = get_plugin_registry()

    if capability_registry is None:
        capability_registry = get_capability_registry()

    for plugin_name, plugin in plugin_registry._plugins.items():
        capabilities = PLUGIN_CAPABILITY_MAP.get(plugin_name, [])
        for tool in plugin.list_tools():
            capability_registry.register_tool_capabilities(tool.name, capabilities)


def auto_discover_all(
    plugin_registry=None,
    capability_registry: Optional[CapabilityRegistry] = None,
) -> Dict[str, List[str]]:
    """Auto-discover everything: load plugins, discover capabilities, sync tools."""
    # Reset registries for clean discovery
    reset_registry()
    reset_capability_registry()

    # Always get fresh registry instances after reset
    plugin_registry = get_plugin_registry()
    capability_registry = get_capability_registry()

    # Load plugins
    loaded_plugins = load_all_plugins(plugin_registry)

    # Register built-in capabilities
    from backend.core.capabilities.builtins import register_builtin_capabilities
    register_builtin_capabilities(capability_registry)

    # Discover from plugins
    discovered_caps = discover_capabilities_from_plugins(plugin_registry, capability_registry)

    # Sync tool-capability mappings
    sync_tool_capabilities(plugin_registry, capability_registry)

    logger.info(
        "Auto-discovery complete: %d plugins, %d new capabilities",
        len(loaded_plugins),
        len(discovered_caps),
    )

    return {
        "loaded_plugins": loaded_plugins,
        "discovered_capabilities": discovered_caps,
        "total_capabilities": len(capability_registry.list_capabilities()),
        "total_tools": len(plugin_registry._tools),
    }


def get_capability_summary() -> Dict[str, Any]:
    """Get a summary of all capabilities and their tools."""
    plugin_registry = get_plugin_registry()
    capability_registry = get_capability_registry()

    summary = {
        "categories": capability_registry.list_categories(),
        "capabilities": {},
        "tools_by_category": {},
        "tools_by_capability": {},
    }

    for cap in capability_registry.list_capabilities():
        tools = capability_registry.get_tools_for_capability(cap.name)
        summary["capabilities"][cap.name] = {
            "description": cap.description,
            "category": cap.category,
            "version": cap.version,
            "tools": tools,
            "dependencies": cap.dependencies,
            "tags": cap.tags,
        }

    for category in plugin_registry.list_categories():
        tools = plugin_registry.get_tools_by_category(category)
        summary["tools_by_category"][category] = [t.name for t in tools]

    for cap_name in capability_registry._capabilities:
        tools = capability_registry.get_tools_for_capability(cap_name)
        if tools:
            summary["tools_by_capability"][cap_name] = tools

    return summary