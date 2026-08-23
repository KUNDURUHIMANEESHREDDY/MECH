"""Plugin/Tool Registry for MECH Platform."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from backend.core.plugins.base import BaseTool, Plugin, PluginManifest, ToolManifest

logger = logging.getLogger("MECH.plugins.registry")


class ToolRegistry:
    """Central registry for tools and plugins."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._plugins: Dict[str, Plugin] = {}
        self._capabilities: Set[str] = set()
        self._tool_categories: Dict[str, Set[str]] = {}

    # Tool management
    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            logger.warning("Tool '%s' already registered, overwriting", tool.name)
        self._tools[tool.name] = tool
        self._tool_categories.setdefault(tool.category, set()).add(tool.name)
        logger.debug("Registered tool: %s (category: %s)", tool.name, tool.category)

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            tool = self._tools.pop(name)
            if tool.category in self._tool_categories:
                self._tool_categories[tool.category].discard(name)
            logger.debug("Unregistered tool: %s", name)
            return True
        return False

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all tools, optionally filtered by category."""
        if category:
            tool_names = self._tool_categories.get(category, set())
            tools = [self._tools[name] for name in tool_names if name in self._tools]
        else:
            tools = list(self._tools.values())
        return [tool.to_dict() for tool in tools]

    def execute_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            return {"status": "error", "error": f"Tool '{name}' not found in registry"}
        tool.initialize()
        return tool.execute(**params)

    def get_tool_manifest(self, name: str) -> Optional[ToolManifest]:
        """Get tool manifest by name."""
        tool = self.get_tool(name)
        return tool.manifest if tool else None

    # Plugin management
    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin and its tools."""
        if plugin.name in self._plugins:
            logger.warning("Plugin '%s' already registered, overwriting", plugin.name)
        self._plugins[plugin.name] = plugin
        plugin.register_tools(self)
        self._capabilities.update(plugin.manifest.capabilities)
        logger.info("Registered plugin: %s v%s (%d tools)", plugin.name, plugin.version, len(plugin.list_tools()))

    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin and its tools."""
        if name in self._plugins:
            plugin = self._plugins.pop(name)
            for tool in plugin.list_tools():
                self.unregister_tool(tool.name)
            self._capabilities.difference_update(plugin.manifest.capabilities)
            logger.info("Unregistered plugin: %s", name)
            return True
        return False

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins."""
        return [plugin.manifest.to_dict() for plugin in self._plugins.values()]

    # Capability management
    def supports(self, capability: str) -> bool:
        """Check if a capability is supported."""
        return capability.lower() in self._capabilities

    def register_capability(self, capability: str) -> None:
        """Register a capability."""
        self._capabilities.add(capability.lower())

    def list_capabilities(self) -> Dict[str, bool]:
        """List all capabilities."""
        return {cap: True for cap in sorted(self._capabilities)}

    # Category management
    def list_categories(self) -> List[str]:
        """List all tool categories."""
        return sorted(self._tool_categories.keys())

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get all tools in a category."""
        tool_names = self._tool_categories.get(category, set())
        return [self._tools[name] for name in tool_names if name in self._tools]

    # Utility
    def clear(self) -> None:
        """Clear all tools and plugins."""
        self._tools.clear()
        self._plugins.clear()
        self._capabilities.clear()
        self._tool_categories.clear()


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = None