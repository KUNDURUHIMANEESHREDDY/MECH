"""Plugin SDK Package — public API surface."""

from .plugin_base import MechPlugin, PluginManifest
from .plugin_loader import PluginLoader, PluginLoadError
from .plugin_registry import PluginRegistry, PluginAlreadyRegisteredError, PluginNotFoundError
from .plugin_hooks import (
    PluginHookBus,
    HookResult,
    HOOK_EXPERIMENT_PLANNED,
    HOOK_CAMPAIGN_STARTED,
    HOOK_CAMPAIGN_COMPLETED,
    HOOK_BELIEF_UPDATED,
    HOOK_EVIDENCE_FUSED,
    HOOK_KNOWLEDGE_GRAPH_UPDATED,
    HOOK_PAPER_GENERATED,
    HOOK_VALIDATION_COMPLETED,
    ALL_HOOKS,
)

__all__ = [
    # Base
    "MechPlugin",
    "PluginManifest",
    # Loader
    "PluginLoader",
    "PluginLoadError",
    # Registry
    "PluginRegistry",
    "PluginAlreadyRegisteredError",
    "PluginNotFoundError",
    # Hook bus
    "PluginHookBus",
    "HookResult",
    "HOOK_EXPERIMENT_PLANNED",
    "HOOK_CAMPAIGN_STARTED",
    "HOOK_CAMPAIGN_COMPLETED",
    "HOOK_BELIEF_UPDATED",
    "HOOK_EVIDENCE_FUSED",
    "HOOK_KNOWLEDGE_GRAPH_UPDATED",
    "HOOK_PAPER_GENERATED",
    "HOOK_VALIDATION_COMPLETED",
    "ALL_HOOKS",
]
