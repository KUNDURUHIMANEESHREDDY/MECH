"""Platform Plugin Registry."""

from __future__ import annotations

from typing import Any, Dict, List


class PluginRegistry:
    """Registry tracking active platform plugins."""

    def __init__(self) -> None:
        self._plugins: Dict[str, Dict[str, Any]] = {}

    def register(self, plugin_id: str, metadata: Dict[str, Any]) -> None:
        self._plugins[plugin_id] = metadata

    def list_plugins(self) -> List[Dict[str, Any]]:
        return list(self._plugins.values())


_plugin_reg_instance: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    global _plugin_reg_instance
    if _plugin_reg_instance is None:
        _plugin_reg_instance = PluginRegistry()
    return _plugin_reg_instance
