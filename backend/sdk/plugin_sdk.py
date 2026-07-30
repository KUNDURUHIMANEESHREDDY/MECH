"""Developer Plugin SDK Entry Point."""

from __future__ import annotations

from typing import Any, Dict
from .interfaces import BasePlugin


class PluginSDK:
    """Entry point for third-party developer plugin registration."""

    def __init__(self) -> None:
        self._registered: Dict[str, BasePlugin] = {}

    def register_plugin(self, plugin: BasePlugin) -> Dict[str, Any]:
        self._registered[plugin.plugin_id] = plugin
        return {
            "status": "registered",
            "plugin_id": plugin.plugin_id,
            "name": plugin.name,
            "version": plugin.version,
        }

    def get_plugin(self, plugin_id: str) -> BasePlugin | None:
        return self._registered.get(plugin_id)
