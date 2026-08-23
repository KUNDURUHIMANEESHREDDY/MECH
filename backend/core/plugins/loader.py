"""Plugin Loader for MECH Platform - Entry-point and filesystem loading."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from backend.core.plugins.base import Plugin, PluginManifest
from backend.core.plugins.registry import ToolRegistry, get_registry

logger = logging.getLogger("MECH.plugins.loader")


class PluginLoadError(Exception):
    """Exception raised when plugin loading fails."""

    pass


class PluginLoader:
    """Loads plugins from entry points and filesystem."""

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry or get_registry()
        self._loaded_modules: Dict[str, Any] = {}

    def load_from_entry_points(self, group: str = "mech.plugins") -> List[str]:
        """Load plugins from setuptools entry points."""
        loaded = []
        try:
            # Python 3.10+
            from importlib.metadata import entry_points

            eps = entry_points(group=group)
        except (ImportError, TypeError):
            try:
                # Python 3.9-
                from importlib_metadata import entry_points

                eps = entry_points().get(group, [])
            except ImportError:
                logger.debug("importlib_metadata not available, skipping entry points")
                return loaded

        for ep in eps:
            try:
                plugin_class = ep.load()
                if not issubclass(plugin_class, Plugin):
                    raise PluginLoadError(f"Entry point {ep.name} does not export a Plugin subclass")
                plugin = plugin_class()
                self.registry.register_plugin(plugin)
                loaded.append(ep.name)
                logger.info("Loaded plugin from entry point: %s", ep.name)
            except Exception as e:
                logger.error("Failed to load plugin from entry point %s: %s", ep.name, e)

        return loaded

    def load_from_file(self, plugin_path: Path) -> Optional[str]:
        """Load a plugin from a Python file."""
        if not plugin_path.exists() or plugin_path.suffix != ".py":
            raise PluginLoadError(f"Invalid plugin file: {plugin_path}")

        module_name = f"mech_plugin_{plugin_path.stem}"
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Cannot load module from {plugin_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        self._loaded_modules[module_name] = module

        # Look for Plugin subclass or get_plugin() function
        plugin_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Plugin) and attr is not Plugin:
                plugin_class = attr
                break

        if plugin_class is None and hasattr(module, "get_plugin"):
            plugin_instance = module.get_plugin()
            if isinstance(plugin_instance, Plugin):
                self.registry.register_plugin(plugin_instance)
                return plugin_instance.name

        if plugin_class is None:
            raise PluginLoadError(f"No Plugin class or get_plugin() found in {plugin_path}")

        plugin = plugin_class()
        self.registry.register_plugin(plugin)
        return plugin.name

    def load_from_directory(self, plugin_dir: Path, recursive: bool = True) -> List[str]:
        """Load all plugins from a directory."""
        loaded = []
        pattern = "**/*.py" if recursive else "*.py"

        for plugin_file in plugin_dir.glob(pattern):
            if plugin_file.name.startswith("__"):
                continue
            try:
                name = self.load_from_file(plugin_file)
                if name:
                    loaded.append(name)
            except PluginLoadError as e:
                logger.error("Failed to load plugin from %s: %s", plugin_file, e)
            except Exception as e:
                logger.error("Unexpected error loading plugin from %s: %s", plugin_file, e)

        return loaded

    def load_builtin_plugins(self) -> List[str]:
        """Load all built-in plugins from backend.core.plugins.builtins."""
        builtins_dir = Path(__file__).parent / "builtins"
        return self.load_from_directory(builtins_dir, recursive=False)

    def create_plugin_instance(self, plugin_class: Type[Plugin], manifest: PluginManifest) -> Plugin:
        """Create a plugin instance with manifest."""
        return plugin_class(manifest)


def load_all_plugins(registry: Optional[ToolRegistry] = None) -> List[str]:
    """Load all plugins (entry points + builtins)."""
    loader = PluginLoader(registry)
    loaded = []
    loaded.extend(loader.load_from_entry_points())
    loaded.extend(loader.load_builtin_plugins())
    return loaded