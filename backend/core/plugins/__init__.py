"""Plugin/Tool Registry Module for MECH Platform."""

from backend.core.plugins.base import (
    BaseTool,
    FunctionTool,
    ParameterSpec,
    Plugin,
    PluginManifest,
    ToolManifest,
)
from backend.core.plugins.loader import PluginLoader, load_all_plugins, PluginLoadError
from backend.core.plugins.manifest import (
    ManifestValidator,
    create_plugin_manifest,
    create_tool_manifest,
    load_manifest_from_file,
    save_manifest_to_file,
)
from backend.core.plugins.registry import ToolRegistry, get_registry, reset_registry

__all__ = [
    # Base classes
    "BaseTool",
    "FunctionTool",
    "ParameterSpec",
    "Plugin",
    "PluginManifest",
    "ToolManifest",
    # Registry
    "ToolRegistry",
    "get_registry",
    "reset_registry",
    # Loader
    "PluginLoader",
    "load_all_plugins",
    "PluginLoadError",
    # Manifest
    "ManifestValidator",
    "create_plugin_manifest",
    "create_tool_manifest",
    "load_manifest_from_file",
    "save_manifest_to_file",
]