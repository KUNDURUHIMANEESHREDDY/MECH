"""Mechanism Benchmarking Engine.

Would evaluate mechanism robustness and intervention fidelity by running real
ablations and causal interventions and comparing them against baselines.

Status: NOT IMPLEMENTED
-----------------------
This module previously returned, for every caller and with no mechanism::

    return {
        "mechanism_name": mechanism_name,
        "robustness_score": 0.935,
        "causal_effect_size": 4.12,
        "ablation_drop_pct": 82.5,
        "status": "Validated",
    }

No benchmark, ablation, or intervention was run; 0.935, 4.12, and 82.5 were
hardcoded and identical for every mechanism. Reporting them as validated
would invent mechanism evidence that was never measured.

`benchmark_mechanism()` now raises. Callers already fail closed on
`LiveUnavailable` rather than substituting a value.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.science.models.adapter_base import LiveUnavailable


class MechanismBenchmarkingEngine:
    """Not implemented. Raises rather than reporting fabricated benchmarks."""

    def benchmark_mechanism(self, mechanism_name: str = "IOI Circuit") -> Dict[str, Any]:
        """Not implemented. Raises rather than reporting fabricated metrics."""
        raise LiveUnavailable(
            "MechanismBenchmarkingEngine is not implemented. It previously "
            "returned hardcoded values of 0.935 (robustness_score), 4.12 "
            "(causal_effect_size), and 82.5 (ablation_drop_pct) for every "
            "mechanism. No mechanism benchmark is performed by this module."
        )
