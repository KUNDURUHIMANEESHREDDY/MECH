"""Fixture plugin used by both the isolation tests and the HTTP round-trip.

Implements every hook the hook bus can dispatch, with two behaviours the tests
depend on:

* ``on_experiment_planned`` returns a *mutated copy* of its input and adds
  ``mutated: True``. If payloads were dropped or aliased across the process
  boundary, the assertions would fail.
* ``on_paper_generated`` appends a marker to the abstract string, so a mutating
  string return can be proven to survive the round trip.
"""

from typing import Any, Dict, List, Optional

from backend.plugins.plugin_base import MechPlugin, PluginManifest
from backend.plugins.plugin_hooks import (
    HOOK_BELIEF_UPDATED,
    HOOK_CAMPAIGN_COMPLETED,
    HOOK_CAMPAIGN_STARTED,
    HOOK_EVIDENCE_FUSED,
    HOOK_EXPERIMENT_PLANNED,
    HOOK_KNOWLEDGE_GRAPH_UPDATED,
    HOOK_PAPER_GENERATED,
    HOOK_VALIDATION_COMPLETED,
)

MARKER = "[fixture: worker round-trip confirmed]"


class FixturePlugin(MechPlugin):
    """Echoes payloads back with a marker so the IPC hop is observable."""

    _MANIFEST = PluginManifest(
        plugin_id="mech.test.live_probe",
        name="Live Probe",
        version="1.0.0",
        author="MECH Platform Team",
        description="Round-trip probe for the plugin worker.",
        hooks=[
            HOOK_EXPERIMENT_PLANNED,
            HOOK_CAMPAIGN_STARTED,
            HOOK_CAMPAIGN_COMPLETED,
            HOOK_BELIEF_UPDATED,
            HOOK_EVIDENCE_FUSED,
            HOOK_KNOWLEDGE_GRAPH_UPDATED,
            HOOK_PAPER_GENERATED,
            HOOK_VALIDATION_COMPLETED,
        ],
    )

    def __init__(self) -> None:
        self.seen: List[Any] = []

    @property
    def manifest(self) -> PluginManifest:
        return self._MANIFEST

    def on_experiment_planned(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.seen.append(plan)
        out = dict(plan)
        out["mutated"] = True
        return out

    def on_campaign_started(self, campaign: Dict[str, Any]) -> None:
        self.seen.append(campaign)

    def on_campaign_completed(self, campaign: Dict[str, Any]) -> None:
        self.seen.append(campaign)

    def on_belief_updated(self, belief_state: Dict[str, Any]) -> None:
        self.seen.append(belief_state)

    def on_evidence_fused(self, evidence_bundle: Dict[str, Any]) -> None:
        self.seen.append(evidence_bundle)

    def on_knowledge_graph_updated(self, event: Dict[str, Any]) -> None:
        self.seen.append(event)

    def on_paper_generated(self, paper: Dict[str, Any]) -> Optional[str]:
        self.seen.append(paper)
        return str(paper.get("abstract", "")) + "\n\n" + MARKER

    def on_validation_completed(self, health_report: Dict[str, Any]) -> None:
        self.seen.append(health_report)


def register() -> MechPlugin:
    return FixturePlugin()
