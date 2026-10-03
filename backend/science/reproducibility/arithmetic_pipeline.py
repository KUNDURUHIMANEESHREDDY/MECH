"""Arithmetic Task Reproduction Pipeline.

Intended to reproduce multi-digit arithmetic circuits (Nkenye et al. / the
modular-arithmetic and addition results in the mechanistic-interpretability
literature) by localising the circuit with causal scrubbing on real weights.

Status: NOT IMPLEMENTED
-----------------------
This module previously returned::

    # Simulated metrics for the mock/test environment
    return {
        "Modulo Addition Accuracy": 0.45,
        "Base-10 Addition Accuracy": 0.85,
    }

Those two numbers were unconditional. There was no mock flag and no branch: the
comment described an intent the code did not implement, so `run()` fabricated a
result for every caller regardless of whether weights were loaded. Worse, the
fabrication was reachable through a real code path -- `benchmark_runner` calls
`run(model_id=...)` for this pipeline and stores the result in a report.

The numbers are now gone rather than gated. Two reasons for not adding a mock
mode:

* A conditional stub is still a stub. It only differs from the previous version
  in which caller receives the fabrication.
* Nothing has a use for these two accuracy values that a real measurement would
  not satisfy. A fixture here would be a claim-shaped object with no consumer.

`run()` therefore raises. The scheduler and `benchmark_runner` both already treat
a raise as "did not run" and record the reason, which is the correct outcome for
an unimplemented benchmark.

To implement it for real, the work is:
  1. Pick a concrete, cited setup (model, task family, number of digits) and
     record it, rather than inferring one from a paper's abstract.
  2. Build clean/corrupted prompt pairs whose corrupted form flips the answer
     while preserving token length.
  3. Scrub causally across layers with hooks on the loaded weights and report
     the measured per-layer effect.

Note for whoever does this: the golden-benchmark entry for arithmetic declares
`Llama3-8B`. An 8B model is ~16GB in fp16, so it does not fit the 6GB GPU this
was last developed on. Either pick a model that fits and change the declared
`model_family` to match what was actually measured, or leave the benchmark
NOT_RUN. Comparing a gpt2-small result against a Llama3-8B baseline would be
the same cross-model comparison error the validation scheduler already guards
against with `baseline_comparable`.
"""

from __future__ import annotations

from typing import Any

from ..models.adapter_base import LiveUnavailable


class ArithmeticPipeline:
    """Not implemented. Raises rather than reporting a fabricated accuracy."""

    PAPER_ID = "arithmetic"

    def __init__(self, model_manager: Any = None) -> None:
        # Retained for call-site compatibility with the other pipelines, which
        # are all constructed as `Pipeline(self.model_manager)`.
        self.model_manager = model_manager

    def run(self, model_id: str = "gpt2-small") -> Dict[str, float]:
        raise LiveUnavailable(
            "ArithmeticPipeline is not implemented. It previously returned "
            "hardcoded accuracies of 0.45 (modulo addition) and 0.85 (base-10 "
            "addition) for every caller, including callers with weights loaded, "
            "and those values were written into reproducibility reports. No "
            "arithmetic circuit measurement is performed by this platform. See "
            "the module docstring for what implementing it requires."
        )
