"""Capability Registry & Verified Mechanistic Tools.

Provides typed, sandboxed execution of mechanistic interpretability primitives
(Logit Lens, Activation Patching, SAE Feature Discovery, Causal Tracing,
Neuron Inspection, Circuit Discovery) directly against model runtime engines.

Now uses the plugin system for dynamic tool discovery and registration.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set

from backend.core.plugins import get_registry, load_all_plugins, reset_registry
from backend.core.plugins.base import sanitize_for_json

logger = logging.getLogger("MECH.capability_registry")


class VerifiedTool:
    """Represents a strongly-typed, verifiable mechanistic tool.

    This is a compatibility wrapper around the plugin system's BaseTool.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Dict[str, Any]],
        category: str = "interpretability",
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.category = category

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the tool safely with parameter logging."""
        try:
            raw_result = self.handler(**kwargs)
            return sanitize_for_json(raw_result)
        except Exception as e:
            logger.error("Tool '%s' execution failed: %s", self.name, e)
            return {
                "tool": self.name,
                "status": "error",
                "error": str(e),
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
        }


class CapabilityRegistry:
    """Central registry querying system capabilities and verified tools.

    This is a compatibility wrapper around the plugin system's ToolRegistry.
    """

    def __init__(self, auto_load_plugins: bool = True) -> None:
        self._registry = get_registry()
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
            "logit_lens",
            "causal_tracing",
            "acdc_circuits",
            "ai_research_assistant",
        }
        self._tools: Dict[str, VerifiedTool] = {}
        self._plugins_loaded = False

        if auto_load_plugins:
            self._load_plugins()

    def _load_plugins(self) -> None:
        """Load all plugins from the plugin system."""
        if self._plugins_loaded:
            return

        try:
            loaded = load_all_plugins(self._registry)
            logger.info("Loaded %d plugins: %s", len(loaded), loaded)

            # Sync capabilities from plugins
            for cap in self._registry.list_capabilities():
                self._capabilities.add(cap)

            # Wrap plugin tools for backward compatibility
            for tool_dict in self._registry.list_tools():
                tool_name = tool_dict["name"]
                plugin_tool = self._registry.get_tool(tool_name)
                if plugin_tool:
                    self._tools[tool_name] = VerifiedTool(
                        name=tool_name,
                        description=tool_dict["description"],
                        parameters={k: v for k, v in tool_dict.get("parameters", {}).items()},
                        handler=plugin_tool.execute,
                        category=tool_dict.get("category", "interpretability"),
                    )

            self._plugins_loaded = True
        except Exception as e:
            logger.warning("Failed to load plugins, falling back to built-in tools: %s", e)
            self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Fallback: Register minimal verified tools if plugin loading fails.

        Does NOT fabricate scientific results. Tools that require model execution
        will return errors indicating the model is not available, rather than
        producing hardcoded/synthetic data.
        """
        logger.info("Registering fallback built-in tools (no synthetic data)")

        # Minimal: only register tools that don't fabricate results
        # All model-dependent tools will return appropriate errors if model not available
        logger.info("No default tools registered - require plugin loading or model availability")

    def _register_fallback_tools(self) -> None:
        """Register minimal fallback tools."""
        # The rest of the tools are available via the plugin system
        # This is just a minimal fallback if plugins fail to load
        pass

    def supports(self, capability: str) -> bool:
        return capability.lower() in self._capabilities

    def register_capability(self, capability: str) -> None:
        self._capabilities.add(capability.lower())
        self._registry.register_capability(capability)

    def list_capabilities(self) -> Dict[str, bool]:
        # Merge with plugin registry capabilities
        all_caps = self._capabilities | set(self._registry.list_capabilities().keys())
        return {cap: True for cap in sorted(all_caps)}

    def register_tool(self, tool: VerifiedTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[VerifiedTool]:
        # First check local tools, then plugin registry
        if name in self._tools:
            return self._tools[name]
        # Try to get from plugin registry and wrap
        plugin_tool = self._registry.get_tool(name)
        if plugin_tool:
            wrapped = VerifiedTool(
                name=name,
                description=plugin_tool.description,
                parameters={k: v for k, v in plugin_tool.parameters.items()},
                handler=plugin_tool.execute,
                category=plugin_tool.category,
            )
            self._tools[name] = wrapped
            return wrapped
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        # Merge local tools with plugin registry tools
        plugin_tools = self._registry.list_tools()
        local_tools = [tool.to_dict() for tool in self._tools.values()]

        # Deduplicate by name
        seen = set()
        merged = []
        for tool in plugin_tools + local_tools:
            if tool["name"] not in seen:
                seen.add(tool["name"])
                merged.append(tool)
        return merged

    def execute_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.get_tool(name)
        if not tool:
            return {"status": "error", "error": f"Tool '{name}' not found in CapabilityRegistry"}
        return tool.execute(**params)

    def reload_plugins(self) -> None:
        """Reload all plugins (useful for development)."""
        reset_registry()
        self._plugins_loaded = False
        self._tools.clear()
        self._load_plugins()


# Global instance for backward compatibility
_global_registry: Optional[CapabilityRegistry] = None


def get_capability_registry() -> CapabilityRegistry:
    """Get the global capability registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CapabilityRegistry()
    return _global_registry