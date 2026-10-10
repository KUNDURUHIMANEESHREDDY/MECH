"""IOI Benchmark Suite.

Status: NOT IMPLEMENTED
-----------------------
This module previously returned, for any model name and with no prompts::

    return {
        "benchmark_name": "IOI Benchmark",
        "model_name": model_name,
        "ioi_accuracy": 0.942,
        "average_logit_diff": 3.84,
        "num_prompts": 1000,
        "status": "Passed",
        ...
    }

No IOI prompts were run, no logit diffs were computed, and no circuit fidelity
was measured. 'model_name' was echoed back while 0.942, 3.84, 1000, and
'Passed' stayed constant, so the result was a fixture, not an evaluation.

run_ioi_eval() now raises. Real IOI measurement lives in the live discovery
path rather than this reference stub.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.science.models.adapter_base import LiveUnavailable


class IOIBenchmarkSuite:
    """Not implemented. Raises rather than reporting fabricated IOI metrics."""

    def run_ioi_eval(self, model_name: str = "GPT-2 Small") -> Dict[str, Any]:
        """Status: NOT IMPLEMENTED.

        This previously returned 'ioi_accuracy: 0.942', 'num_prompts: 1000',
        'average_logit_diff: 3.84', 'status: "Passed"', and 'provenance:
        "reference"' for any 'model_name'. Those numbers were constant and not
        computed from model outputs, so a caller could not tell whether any IOI
        evaluation had happened.
        """
        raise LiveUnavailable(
            "IOIBenchmarkSuite.run_ioi_eval is not implemented. It previously "
            "returned ioi_accuracy=0.942, average_logit_diff=3.84, "
            "num_prompts=1000, status='Passed', and provenance='reference' for "
            "any model_name without running prompts or computing logit diffs."
        )
