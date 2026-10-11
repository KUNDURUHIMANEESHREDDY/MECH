"""Capability Registry.

Exposes central capability queries (supports_sae, supports_patching,
supports_cluster, supports_cloud, supports_llama).

`list_capabilities()` previously returned `{cap: True for cap in ...}` -- every
value `True` by construction, so the registry could not express "registered but
not implemented". That is the structural half of EXP-005: the surface which
would have to source a `NOT_EXECUTED` status had nowhere to put it. It now
reports `False` for capabilities in
`execution_status.UNIMPLEMENTED_CAPABILITIES`, and `status_for()` returns the
status itself rather than a bare boolean.
"""

from __future__ import annotations

from typing import Dict, Set

from backend.core.execution_status import (
    IMPLEMENTED,
    NOT_EXECUTED,
    UNIMPLEMENTED_CAPABILITIES,
)


class CapabilityRegistry:
    """Central registry querying system and plugin capabilities."""

    def __init__(self) -> None:
        self._capabilities: Set[str] = {
            "sae",
            "patching",
            "cluster",
            "cloud",
            "llama",
            "gemma",
            "qwen",
            "mistral",
            "deepseek",
            "visualization",
            "validation",
            "reproducibility",
        }

    def supports(self, capability: str) -> bool:
        """True when the capability has a working implementation.

        A registered-but-unimplemented capability returns False. It stays in
        `_capabilities` -- registration and implementation are different facts
        -- but callers asking "can I do this" must not be told yes.
        """
        key = capability.lower()
        return key in self._capabilities and key not in UNIMPLEMENTED_CAPABILITIES

    def status_for(self, capability: str) -> str:
        """`IMPLEMENTED`, or `NOT_EXECUTED` for a registered gap.

        For an unregistered capability this returns `NOT_EXECUTED` rather than
        raising: a caller asking about something absent should learn that it
        is absent, not that it has a bug.
        """
        key = capability.lower()
        if key in UNIMPLEMENTED_CAPABILITIES:
            return NOT_EXECUTED
        return IMPLEMENTED if key in self._capabilities else NOT_EXECUTED

    def register_capability(self, capability: str) -> None:
        self._capabilities.add(capability.lower())

    def list_capabilities(self) -> Dict[str, bool]:
        """Implementation state per capability.

        Not a `{cap: True}` comprehension: a registered capability with no
        working engine behind it reports False, and `status_for()` gives the
        reason.
        """
        return {cap: self.supports(cap) for cap in sorted(self._capabilities)}
