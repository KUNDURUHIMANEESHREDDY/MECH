"""Example plugin — IOI Experiment Logger.

Demonstrates the full MechPlugin API by logging every IOI-related
experiment, campaign, and paper event to a local audit trail.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.plugins.plugin_base import MechPlugin, PluginManifest
from backend.plugins.plugin_hooks import (
    HOOK_CAMPAIGN_COMPLETED,
    HOOK_CAMPAIGN_STARTED,
    HOOK_EXPERIMENT_PLANNED,
    HOOK_PAPER_GENERATED,
    HOOK_VALIDATION_COMPLETED,
)

logger = logging.getLogger(__name__)

_AUDIT_FILE = Path(__file__).parent / "ioi_audit_trail.jsonl"


class IOIExperimentLoggerPlugin(MechPlugin):
    """
    Logs every experiment plan, campaign event, and paper that mentions
    'IOI' or 'indirect object identification' to a local JSONL audit trail.
    Demonstrates all major plugin hooks in one coherent example.
    """

    _MANIFEST = PluginManifest(
        plugin_id="mech.example.ioi_experiment_logger",
        name="IOI Experiment Logger",
        version="1.0.0",
        author="MECH Platform Team",
        description=(
            "Example plugin: logs every IOI-related experiment plan, campaign event, "
            "and paper to a local JSONL audit trail."
        ),
        hooks=[
            HOOK_EXPERIMENT_PLANNED,
            HOOK_CAMPAIGN_STARTED,
            HOOK_CAMPAIGN_COMPLETED,
            HOOK_PAPER_GENERATED,
            HOOK_VALIDATION_COMPLETED,
        ],
        ui_view=None,  # no custom sidebar panel
    )

    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []

    @property
    def manifest(self) -> PluginManifest:
        return self._MANIFEST

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def on_load(self) -> None:
        logger.info("[IOILogger] Plugin loaded — audit trail: %s", _AUDIT_FILE)

    def on_unload(self) -> None:
        logger.info("[IOILogger] Plugin unloaded — %d events logged.", len(self._events))

    # ------------------------------------------------------------------ #
    # Hook implementations                                                 #
    # ------------------------------------------------------------------ #

    def on_experiment_planned(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Log IOI plans; pass all others through unchanged."""
        name = str(plan.get("name", "")).lower()
        if "ioi" in name or "indirect object" in name:
            self._log_event("experiment_planned", plan)
        return None  # do not mutate plan

    def on_campaign_started(self, campaign: Dict[str, Any]) -> None:
        self._log_event("campaign_started", campaign)

    def on_campaign_completed(self, campaign: Dict[str, Any]) -> None:
        self._log_event("campaign_completed", {
            "campaign_id": campaign.get("campaign_id"),
            "goal": campaign.get("goal"),
            "total_experiments": campaign.get("total_experiments", 0),
        })

    def on_paper_generated(self, paper: Dict[str, Any]) -> Optional[str]:
        """Append a citation note to IOI papers."""
        abstract = paper.get("abstract", "")
        if "ioi" in abstract.lower() or "indirect object" in abstract.lower():
            self._log_event("paper_generated", {"title": paper.get("title")})
            # Return enriched abstract — mutating hook
            return abstract + "\n\n[Logged by IOI Experiment Logger Plugin v1.0.0]"
        return None

    def on_validation_completed(self, health_report: Dict[str, Any]) -> None:
        passed = health_report.get("passed_benchmarks", 0)
        total = health_report.get("total_benchmarks_run", 0)
        self._log_event("validation_completed", {
            "passed": passed,
            "total": total,
            "health_score": health_report.get("platform_health_score"),
        })

    # ------------------------------------------------------------------ #
    # Inspection                                                           #
    # ------------------------------------------------------------------ #

    @property
    def audit_events(self) -> List[Dict[str, Any]]:
        """Return all logged events (in-memory copy)."""
        return list(self._events)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        entry = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "event": event_type,
            **data,
        }
        self._events.append(entry)
        try:
            with _AUDIT_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # non-critical — in-memory log is the source of truth


def register() -> MechPlugin:
    """Plugin SDK entry point — called by PluginLoader."""
    return IOIExperimentLoggerPlugin()
