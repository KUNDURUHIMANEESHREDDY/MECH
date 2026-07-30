"""Plugin Hook Bus — Dispatches platform events to subscribed plugins.

Each platform component (Discovery Planner, Campaign Manager, etc.) calls
`HookBus.emit(hook_name, payload)`. The bus fans out to every registered
plugin that implements that hook, collects return values, and returns them.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from .plugin_registry import PluginRegistry

logger = logging.getLogger(__name__)

# Canonical hook names — matches method names on MechPlugin
HOOK_EXPERIMENT_PLANNED      = "on_experiment_planned"
HOOK_CAMPAIGN_STARTED        = "on_campaign_started"
HOOK_CAMPAIGN_COMPLETED      = "on_campaign_completed"
HOOK_BELIEF_UPDATED          = "on_belief_updated"
HOOK_EVIDENCE_FUSED          = "on_evidence_fused"
HOOK_KNOWLEDGE_GRAPH_UPDATED = "on_knowledge_graph_updated"
HOOK_PAPER_GENERATED         = "on_paper_generated"
HOOK_VALIDATION_COMPLETED    = "on_validation_completed"

ALL_HOOKS = [
    HOOK_EXPERIMENT_PLANNED,
    HOOK_CAMPAIGN_STARTED,
    HOOK_CAMPAIGN_COMPLETED,
    HOOK_BELIEF_UPDATED,
    HOOK_EVIDENCE_FUSED,
    HOOK_KNOWLEDGE_GRAPH_UPDATED,
    HOOK_PAPER_GENERATED,
    HOOK_VALIDATION_COMPLETED,
]


class HookResult:
    """Aggregated results from all plugins for a single hook emission."""

    def __init__(self, hook: str, payload: Any) -> None:
        self.hook = hook
        self.original_payload = payload
        self.plugin_results: Dict[str, Any] = {}   # plugin_id → return value
        self.errors: Dict[str, str] = {}            # plugin_id → error message
        self.final_payload: Any = payload           # may be mutated by chain plugins

    def first_non_none(self) -> Optional[Any]:
        """Return the first non-None plugin return value, or None."""
        for v in self.plugin_results.values():
            if v is not None:
                return v
        return None


class PluginHookBus:
    """
    Event bus that dispatches platform hook events to all registered plugins.

    Platform components call ``bus.emit(HOOK_*, payload)`` and receive a
    :class:`HookResult` aggregating every plugin's response.

    Architecture
    ------------
    - Hooks are dispatched **synchronously** in registration order.
    - Plugins that raise exceptions are logged but do NOT break the chain.
    - "Mutating" hooks (``on_experiment_planned``, ``on_paper_generated``)
      pass the return value of each plugin as the payload to the next,
      enabling a filter pipeline.
    """

    MUTATING_HOOKS = {HOOK_EXPERIMENT_PLANNED, HOOK_PAPER_GENERATED}

    def __init__(self, registry: Optional[PluginRegistry] = None) -> None:
        self._registry = registry or PluginRegistry.global_instance()
        self._mu = threading.RLock()
        self._middleware: List[Callable[[str, Any], None]] = []

    # ------------------------------------------------------------------ #
    # Middleware                                                           #
    # ------------------------------------------------------------------ #

    def add_middleware(self, fn: Callable[[str, Any], None]) -> None:
        """Register a callable invoked before each hook dispatch for logging/tracing."""
        with self._mu:
            self._middleware.append(fn)

    # ------------------------------------------------------------------ #
    # Emission                                                             #
    # ------------------------------------------------------------------ #

    def emit(self, hook_name: str, payload: Any = None) -> HookResult:
        """Dispatch a hook event to all registered plugins that implement it.

        Args:
            hook_name: One of the ``HOOK_*`` constants.
            payload: The data to pass to each plugin hook method.

        Returns:
            :class:`HookResult` with per-plugin results and errors.
        """
        result = HookResult(hook=hook_name, payload=payload)
        current_payload = payload

        # Run middleware
        with self._mu:
            mw = list(self._middleware)
        for fn in mw:
            try:
                fn(hook_name, current_payload)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Middleware error on hook '%s': %s", hook_name, exc)

        plugins = self._registry.list_plugins()
        for plugin in plugins:
            pid = plugin.manifest.plugin_id
            # Only call if plugin declares this hook
            if hook_name not in plugin.manifest.hooks:
                continue
            handler = getattr(plugin, hook_name, None)
            if handler is None or not callable(handler):
                continue
            try:
                ret = handler(current_payload)
                result.plugin_results[pid] = ret
                # Mutating hooks: chain the return value as next payload
                if hook_name in self.MUTATING_HOOKS and ret is not None:
                    current_payload = ret
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{type(exc).__name__}: {exc}"
                result.errors[pid] = error_msg
                logger.warning(
                    "Plugin '%s' raised during hook '%s': %s", pid, hook_name, error_msg
                )

        result.final_payload = current_payload
        return result

    # ------------------------------------------------------------------ #
    # Convenience emitters (typed wrappers)                               #
    # ------------------------------------------------------------------ #

    def on_experiment_planned(self, plan: Dict[str, Any]) -> HookResult:
        return self.emit(HOOK_EXPERIMENT_PLANNED, plan)

    def on_campaign_started(self, campaign: Dict[str, Any]) -> HookResult:
        return self.emit(HOOK_CAMPAIGN_STARTED, campaign)

    def on_campaign_completed(self, campaign: Dict[str, Any]) -> HookResult:
        return self.emit(HOOK_CAMPAIGN_COMPLETED, campaign)

    def on_belief_updated(self, belief_state: Dict[str, Any]) -> HookResult:
        return self.emit(HOOK_BELIEF_UPDATED, belief_state)

    def on_evidence_fused(self, evidence_bundle: Dict[str, Any]) -> HookResult:
        return self.emit(HOOK_EVIDENCE_FUSED, evidence_bundle)

    def on_knowledge_graph_updated(self, event: Dict[str, Any]) -> HookResult:
        return self.emit(HOOK_KNOWLEDGE_GRAPH_UPDATED, event)

    def on_paper_generated(self, paper: Dict[str, Any]) -> HookResult:
        return self.emit(HOOK_PAPER_GENERATED, paper)

    def on_validation_completed(self, health_report: Dict[str, Any]) -> HookResult:
        return self.emit(HOOK_VALIDATION_COMPLETED, health_report)
