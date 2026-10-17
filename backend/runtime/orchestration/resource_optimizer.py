"""Automatic Resource Optimizer Engine.

STATIC POLICY TABLE, NOT A MEASUREMENT. ``optimize_resources`` maps a strategy
name to a fixed row of hardware choices. It reads no cluster state, checks no
GPU inventory, and inspects no job profile, so ``gpu_target`` is a stated
preference rather than a finding about what exists or what is free.

The class is named "Optimizer" and the method "optimal", which invites the
opposite reading, so every plan now names its own source.
"""

from __future__ import annotations

from typing import Any, Dict


class ResourceOptimizerEngine:
    """Selects a fixed policy row for a strategy. Nothing is measured."""

    # Declared once so the rows and their default cannot drift apart.
    _POLICIES: Dict[str, Dict[str, str]] = {
        "Speed": {
            "gpu_target": "NVIDIA H100",
            "precision": "FP16",
            "compression": "None",
            "streaming": "predictive_prefetch",
        },
        "Memory": {
            "gpu_target": "NVIDIA RTX 4090",
            "precision": "INT8",
            "compression": "INT8Codec",
            "streaming": "lazy",
        },
        "Cost": {
            "gpu_target": "RunPod RTX 4090",
            "precision": "INT8",
            "compression": "FP16Codec",
            "streaming": "lazy",
        },
    }
    _DEFAULT_STRATEGY = "Balanced"
    _BALANCED: Dict[str, str] = {
        "gpu_target": "NVIDIA A100",
        "precision": "FP16",
        "compression": "FP16Codec",
        "streaming": "predictive_prefetch",
    }

    def optimize_resources(self, strategy: str = "Balanced") -> Dict[str, Any]:
        """Return the policy row for ``strategy``, labelled as a policy row.

        An unrecognised strategy falls back to Balanced, and ``strategy``
        reports the row actually used so the fallback is visible rather than
        implied by the absence of the caller's name.
        """
        known = strategy in self._POLICIES
        row = self._POLICIES[strategy] if known else self._BALANCED

        plan: Dict[str, Any] = dict(row)
        plan["strategy"] = strategy if known else self._DEFAULT_STRATEGY
        plan["source"] = "static_policy_table"
        plan["measured"] = False
        plan["validation_eligible"] = False
        plan["publication_eligible"] = False
        plan["reason"] = (
            "Hardware choices come from a fixed policy table. No cluster "
            "state, GPU inventory, or job profile was read, so this plan is a "
            "preference and not a measurement of available capacity."
        )
        return plan
