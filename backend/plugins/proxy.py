"""Parent-side proxy for a plugin running in a worker process.

``RemotePluginProxy`` subclasses :class:`~backend.plugins.plugin_base.MechPlugin`
so the existing registry, hook bus, and service code treat a sandboxed worker
exactly like an in-process plugin. Every hook call is forwarded as JSON over
the worker's stdin/stdout; the plugin object itself never exists in this
process.

The manifest is fetched once during the handshake and cached locally, so
``plugin.manifest`` — which the registry and hook bus read on every dispatch —
costs no IPC.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from .plugin_base import MechPlugin, PluginManifest
from .plugin_hooks import ALL_HOOKS
from .protocol import MAX_MESSAGE_BYTES, ProtocolError, decode, encode

logger = logging.getLogger(__name__)


class RemotePluginError(Exception):
    """The worker died, timed out, or violated the protocol."""


def _manifest_from_dict(data: Dict[str, Any]) -> PluginManifest:
    return PluginManifest(
        plugin_id=str(data.get("plugin_id", "")),
        name=str(data.get("name", "")),
        version=str(data.get("version", "")),
        author=str(data.get("author", "")),
        description=str(data.get("description", "")),
        hooks=list(data.get("hooks") or []),
        ui_view=data.get("ui_view"),
        dependencies=list(data.get("dependencies") or []),
    )


class RemotePluginProxy(MechPlugin):
    """A :class:`MechPlugin` whose implementation lives in a worker process."""

    def __init__(
        self,
        stdin,
        stdout,
        manifest: PluginManifest,
        timeout: float = 30.0,
    ) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._manifest = manifest
        self._timeout = timeout
        self._lock = threading.RLock()
        self._next_id = 0
        self._alive = True

    # -- transport -----------------------------------------------------
    def _roundtrip(self, message: Dict[str, Any]) -> Any:
        """Send one request and await its reply. Raises on any failure."""
        if not self._alive:
            raise RemotePluginError("plugin worker is not running")
        with self._lock:
            try:
                self._stdin.write(encode(message))
                self._stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                self._alive = False
                raise RemotePluginError(
                    f"plugin worker stdin closed: {exc}") from exc

            line = self._stdout.readline()
            if not line:
                self._alive = False
                raise RemotePluginError("plugin worker exited unexpectedly")
            if len(line) > MAX_MESSAGE_BYTES:
                self._alive = False
                raise RemotePluginError("plugin worker reply exceeded the size cap")

            try:
                response = decode(line)
            except ProtocolError as exc:
                self._alive = False
                raise RemotePluginError(f"malformed worker reply: {exc}") from exc

        if response.get("event") == "failed":
            self._alive = False
            raise RemotePluginError(
                f"worker reported failure in {response.get('stage', '?')}: "
                f"{response.get('error', '')}")

        if not response.get("ok"):
            self._alive = False
            raise RemotePluginError(str(response.get("error", "worker call failed")))
        return response.get("result")

    def _call(self, hook: str, *args: Any) -> Any:
        with self._lock:
            self._next_id += 1
            msg_id = self._next_id
        return self._roundtrip({"id": msg_id, "op": "call", "hook": hook,
                               "args": list(args)})

    # -- MechPlugin surface --------------------------------------------
    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def on_load(self) -> None:
        """No-op: the worker already ran on_load during the load handshake."""

    def on_unload(self) -> None:
        try:
            self._roundtrip({"op": "call", "hook": "__noop__", "args": []})
        except Exception:  # noqa: BLE001
            pass

    def on_experiment_planned(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._call("on_experiment_planned", plan)

    def on_campaign_started(self, campaign: Dict[str, Any]) -> None:
        self._call("on_campaign_started", campaign)

    def on_campaign_completed(self, campaign: Dict[str, Any]) -> None:
        self._call("on_campaign_completed", campaign)

    def on_belief_updated(self, belief_state: Dict[str, Any]) -> None:
        self._call("on_belief_updated", belief_state)

    def on_evidence_fused(self, evidence_bundle: Dict[str, Any]) -> None:
        self._call("on_evidence_fused", evidence_bundle)

    def on_knowledge_graph_updated(self, event: Dict[str, Any]) -> None:
        self._call("on_knowledge_graph_updated", event)

    def on_paper_generated(self, paper: Dict[str, Any]) -> Optional[str]:
        return self._call("on_paper_generated", paper)

    def on_validation_completed(self, health_report: Dict[str, Any]) -> None:
        self._call("on_validation_completed", health_report)

    def get_ui_component(self) -> Optional[str]:
        return self._manifest.ui_view

    # -- introspection -------------------------------------------------
    @property
    def is_sandboxed(self) -> bool:
        return True

    @property
    def is_alive(self) -> bool:
        return self._alive

    def declared_hooks(self) -> List[str]:
        return list(ALL_HOOKS)
