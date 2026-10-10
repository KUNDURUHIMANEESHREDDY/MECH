"""Benchmark Registry System.

Status: NOT IMPLEMENTED
-----------------------
This module previously returned, for every benchmark and with no measurement::

    results[b_id] = {"score": 0.94, "passed": True}
    return {"total_benchmarks": len(results), "passed_count": len(results), "results": results}

No benchmark was run, no prompts were evaluated, and no model was connected.
Every benchmark reported the same 0.94 score and the same pass flag, so
run_all_benchmarks() produced a fake registry pass.

run_all_benchmarks() now raises. Listing remains a static catalog of
benchmark metadata, not a result.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.science.models.adapter_base import LiveUnavailable


class BenchmarkRegistry:
    """Manages standard benchmark metadata. Run results are not implemented."""

    def __init__(self) -> None:
        self.benchmarks: Dict[str, Dict[str, Any]] = {
            "IOI": {"name": "Indirect Object Identification", "task_type": "CircuitRecall", "eval_samples": 1000},
            "GreaterThan": {"name": "Greater Than Year Comparison", "task_type": "NumericalReasoning", "eval_samples": 500},
            "InductionHeads": {"name": "Pattern Continuation A B ... A->B", "task_type": "SequenceMatching", "eval_samples": 800},
            "CopyTask": {"name": "Repeated Token Copying", "task_type": "AttentionCopy", "eval_samples": 600},
            "FactualRecall": {"name": "Country-Capital Knowledge Retrieval", "task_type": "FactualAssociation", "eval_samples": 1200},
            "Arithmetic": {"name": "Single-Digit Addition", "task_type": "SymbolicMath", "eval_samples": 500},
        }

    def list_benchmarks(self) -> List[Dict[str, Any]]:
        return [{"id": k, **v} for k, v in self.benchmarks.items()]

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Status: NOT IMPLEMENTED.

        This previously returned '{"score": 0.94, "passed": True}' for every
        benchmark, with 'passed_count' equal to the total number of benchmarks.
        That was wrong because the same pass/fail numbers were asserted without
        running IOI, GreaterThan, InductionHeads, CopyTask, FactualRecall, or
        Arithmetic.
        """
        raise LiveUnavailable(
            "BenchmarkRegistry.run_all_benchmarks is not implemented. It "
            "previously returned {'score': 0.94, 'passed': True} for every "
            "benchmark and marked the whole registry passed. No benchmark was "
            "run and no model was connected."
        )
