"""Mock Execution Prohibition — Fail-Closed Guard for Research Pipelines.

Scientific integrity invariant: no pipeline may return synthetic, simulated,
or fabricated results under any circumstances. If real model weights are not
available, execution must FAIL LOUDLY instead of producing plausible-looking
numbers.
"""

from __future__ import annotations

from typing import Any


class MockExecutionProhibitedError(RuntimeError):
    """Raised when a research pipeline is invoked without live model weights."""


def require_live_model(adapter: Any, pipeline_name: str) -> None:
    """Fails closed unless the adapter holds real, loaded model weights.

    Call this at the top of every ``run()`` method BEFORE any computation so
    that mock-mode invocations can never emit partial or synthetic output.
    """
    spec = getattr(adapter, "spec", None)
    if spec is None:
        raise MockExecutionProhibitedError(
            f"{pipeline_name}: adapter has no ModelSpec; refusing to execute."
        )
    if getattr(spec, "mock_mode", False):
        raise MockExecutionProhibitedError(
            f"{pipeline_name}: refusing to execute in mock_mode. Synthetic "
            "results must never reach EvidenceRecord, Finding, or "
            "ScientificConclusion. Construct this pipeline with "
            "mock_mode=False and ensure model weights are available."
        )
    if getattr(adapter, "_model", None) is None:
        raise MockExecutionProhibitedError(
            f"{pipeline_name}: live model weights are not loaded (adapter "
            "_model is None). Load a real model before running research "
            "pipelines; MECH does not fabricate results."
        )
