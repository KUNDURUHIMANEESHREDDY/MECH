"""Copy Task Reproduction Pipeline.

Would reproduce the induction/copy-task results (Elhage et al. 2021; Olsson et
al. 2022) by measuring how much attention heads direct to previous occurrences of
the current token, and comparing that against a shuffled control.

Status: NOT IMPLEMENTED
-----------------------
This module previously returned, for every caller and with no branch::

    # Simulated metrics for the mock/test environment
    return {
        "Copy Score": 0.92,
        "Attention to previous occurrence": 0.88,
    }

The comment describes an intent the code did not implement: there was no mock
flag, so callers with weights loaded received these numbers, and
`benchmark_runner` stored them in a reproducibility report.

0.92 and 0.88 are also suspiciously good. GPT-2 small does not achieve those
figures on an induction-head measurement, and neither number was ever derived
from attention weights.

`run()` now raises. The scheduler and `benchmark_runner` both already treat a
raise as "did not run" and record the reason.

Note for whoever implements this: the induction-head machinery needed for a real
version already exists and is exercised --
`backend/science/reproducibility/induction_heads_pipeline.py` measures
previous-token attention against a uniform baseline and a repeated-block
control, and reports a mechanism attribution of `previous_token_copying`. The
honest path here is to reuse that measurement rather than invent a second,
weaker one, and to report whatever it actually finds.
"""

from __future__ import annotations

from typing import Any, Dict

from ..models.adapter_base import LiveUnavailable


class CopyTaskPipeline:
    """Not implemented. Raises rather than reporting a fabricated copy score."""

    PAPER_ID = "copy_task"

    def __init__(self, model_manager: Any = None) -> None:
        # Retained for call-site compatibility with the other pipelines.
        self.model_manager = model_manager

    def run(self, model_id: str = "gpt2-small") -> Dict[str, float]:
        raise LiveUnavailable(
            "CopyTaskPipeline is not implemented. It previously returned "
            "hardcoded values of 0.92 (Copy Score) and 0.88 (Attention to "
            "previous occurrence) for every caller, including callers with "
            "weights loaded. No copy-task measurement is performed by this "
            "pipeline. The induction-heads pipeline already performs a real "
            "measurement of the same underlying behaviour; see the module "
            "docstring."
        )
