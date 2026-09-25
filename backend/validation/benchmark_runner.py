"""Mechanistic Benchmark Runner.

The desktop currently has no connected live benchmark executor.  This module
must fail closed rather than returning a plausible score: a benchmark result
is scientific evidence only when a real executor returns measured fields.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.agents.evidence_policy import field_map


class MechanisticBenchmarkRunner:
    """Report benchmark execution availability without fabricating results."""

    def run_benchmark(self, benchmark_id: str = "bench_default") -> Dict[str, Any]:
        return {
            "benchmark_id": benchmark_id,
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(
                ("benchmark_id", "status", "reason"), "unavailable"
            ),
            "reason": "No live benchmark executor is connected.",
        }
