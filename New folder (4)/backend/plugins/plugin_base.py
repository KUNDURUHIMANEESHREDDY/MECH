"""Plugin SDK — Abstract Base Class for all MECH research plugins.

Every plugin must subclass MechPlugin and implement the required hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PluginManifest:
    """Declarative metadata every plugin must provide."""
    plugin_id: str                       # Globally unique slug, e.g. 'mylab.ioi_extension'
    name: str                            # Human-readable name
    version: str                         # Semantic version string, e.g. '1.0.0'
    author: str
    description: str
    hooks: List[str] = field(default_factory=list)   # Hook names this plugin implements
    ui_view: Optional[str] = None        # Optional: React component filename to inject
    dependencies: List[str] = field(default_factory=list)  # Other plugin_ids required first


class MechPlugin(ABC):
    """
    Abstract base class for all MECH research platform plugins.

    Lifecycle
    ---------
    1. Plugin is discovered and loaded by PluginLoader.
    2. PluginRegistry calls plugin.on_load() and stores it.
    3. Platform emits hook events; PluginHooks dispatches them to subscribers.
    4. Plugin may optionally provide a UI view component.
    5. PluginRegistry calls plugin.on_unload() when removed.

    Every plugin must implement :meth:`manifest` and any hooks it declares.
    """

    # ------------------------------------------------------------------ #
    # Required                                                             #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def manifest(self) -> PluginManifest:
        """Return the plugin's declarative metadata."""

    # ------------------------------------------------------------------ #
    # Lifecycle hooks (optional overrides)                                 #
    # ------------------------------------------------------------------ #

    def on_load(self) -> None:
        """Called once after the plugin is successfully registered."""

    def on_unload(self) -> None:
        """Called when the plugin is removed from the registry."""

    # ------------------------------------------------------------------ #
    # Platform integration hooks (optional overrides)                      #
    # ------------------------------------------------------------------ #

    # --- Discovery Planner ---
    def on_experiment_planned(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Called when the Discovery Planner creates a new experiment plan.
        Return a modified plan dict to override, or None to pass-through."""

    # --- Research Campaign Manager ---
    def on_campaign_started(self, campaign: Dict[str, Any]) -> None:
        """Called when a new Research Campaign is created."""

    def on_campaign_completed(self, campaign: Dict[str, Any]) -> None:
        """Called when a Research Campaign finishes."""

    # --- Bayesian Belief Engine ---
    def on_belief_updated(self, belief_state: Dict[str, Any]) -> None:
        """Called after each Bayesian posterior update."""

    # --- Evidence Fusion ---
    def on_evidence_fused(self, evidence_bundle: Dict[str, Any]) -> None:
        """Called when the Evidence Fusion Engine produces a new bundle."""

    # --- Knowledge Graph ---
    def on_knowledge_graph_updated(self, event: Dict[str, Any]) -> None:
        """Called whenever a node or edge is added to the Knowledge Graph."""

    # --- Publication Engine ---
    def on_paper_generated(self, paper: Dict[str, Any]) -> Optional[str]:
        """Called when a paper is auto-generated.
        Return a modified abstract string to override, or None to pass-through."""

    # --- Validation Framework ---
    def on_validation_completed(self, health_report: Dict[str, Any]) -> None:
        """Called after each Continuous Validation run."""

    # --- UI ---
    def get_ui_component(self) -> Optional[str]:
        """Return the name of a JSX component file to mount in the sidebar UI.
        Must reside in src/components/plugins/<plugin_id>/ directory."""
        return self.manifest.ui_view
