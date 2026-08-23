"""Plugin Registry — Central store of all loaded MechPlugin instances.

Provides thread-safe registration, lookup, and lifecycle management.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Iterator, List, Optional

from .plugin_base import MechPlugin, PluginManifest

logger = logging.getLogger(__name__)


class PluginAlreadyRegisteredError(Exception):
    pass


class PluginNotFoundError(Exception):
    pass


class PluginRegistry:
    """
    Thread-safe central registry of all active MechPlugin instances.

    Usage
    -----
    >>> registry = PluginRegistry()
    >>> registry.register(my_plugin)
    >>> plugin = registry.get("mylab.ioi_extension")
    >>> registry.unregister("mylab.ioi_extension")
    """

    _instance: Optional["PluginRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._plugins: Dict[str, MechPlugin] = {}
        self._mu: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Singleton access (optional convenience)                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def global_instance(cls) -> "PluginRegistry":
        """Return the process-wide singleton registry."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    def register(self, plugin: MechPlugin, *, allow_override: bool = False) -> None:
        """Register a plugin instance.

        Args:
            plugin: Instantiated MechPlugin subclass.
            allow_override: If True, silently replaces an existing plugin with the same ID.

        Raises:
            PluginAlreadyRegisteredError: If a plugin with the same ID is already registered
                and allow_override is False.
        """
        manifest: PluginManifest = plugin.manifest
        pid = manifest.plugin_id
        with self._mu:
            if pid in self._plugins and not allow_override:
                raise PluginAlreadyRegisteredError(
                    f"Plugin '{pid}' is already registered. "
                    "Use allow_override=True to replace it."
                )
            if pid in self._plugins:
                try:
                    self._plugins[pid].on_unload()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Plugin '%s' on_unload() raised: %s", pid, exc)
            self._plugins[pid] = plugin
            try:
                plugin.on_load()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Plugin '%s' on_load() raised: %s", pid, exc)
        logger.info("Plugin registered: %s v%s by %s", pid, manifest.version, manifest.author)

    def unregister(self, plugin_id: str) -> None:
        """Unload and remove a plugin by its ID.

        Raises:
            PluginNotFoundError: If no plugin with that ID is registered.
        """
        with self._mu:
            if plugin_id not in self._plugins:
                raise PluginNotFoundError(f"Plugin '{plugin_id}' is not registered.")
            plugin = self._plugins.pop(plugin_id)
            try:
                plugin.on_unload()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Plugin '%s' on_unload() raised: %s", plugin_id, exc)
        logger.info("Plugin unregistered: %s", plugin_id)

    def unregister_all(self) -> None:
        """Unload all plugins."""
        with self._mu:
            for pid, plugin in list(self._plugins.items()):
                try:
                    plugin.on_unload()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Plugin '%s' on_unload() raised: %s", pid, exc)
            self._plugins.clear()

    # ------------------------------------------------------------------ #
    # Lookup                                                               #
    # ------------------------------------------------------------------ #

    def get(self, plugin_id: str) -> MechPlugin:
        """Return a registered plugin by ID.

        Raises:
            PluginNotFoundError: If not found.
        """
        with self._mu:
            if plugin_id not in self._plugins:
                raise PluginNotFoundError(f"Plugin '{plugin_id}' is not registered.")
            return self._plugins[plugin_id]

    def get_optional(self, plugin_id: str) -> Optional[MechPlugin]:
        """Return plugin or None."""
        with self._mu:
            return self._plugins.get(plugin_id)

    def list_manifests(self) -> List[PluginManifest]:
        """Return manifests of all registered plugins."""
        with self._mu:
            return [p.manifest for p in self._plugins.values()]

    def list_plugins(self) -> List[MechPlugin]:
        """Return all registered plugin instances."""
        with self._mu:
            return list(self._plugins.values())

    def __len__(self) -> int:
        with self._mu:
            return len(self._plugins)

    def __iter__(self) -> Iterator[MechPlugin]:
        with self._mu:
            return iter(list(self._plugins.values()))

    def __contains__(self, plugin_id: str) -> bool:
        with self._mu:
            return plugin_id in self._plugins
