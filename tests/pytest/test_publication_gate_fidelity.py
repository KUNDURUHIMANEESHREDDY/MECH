"""The publication gate must not be bypassable by a failing fidelity check.

Why this file exists
--------------------
`publication_block_reason` concludes with `blocked_reason(gate, "Validation
gate")`. `blocked_reason` asks whether the chain was *eligible* -- live
provenance, completed, attested, eligible flags -- and returns "" when every
answer is yes.

`gate_is_live` asks a stricter question: it also requires `passed is True`. But
`blocked_reason` did not check `passed` at all. So a gate carrying every
eligibility field *but a failing fidelity measurement* produced an **empty**
reason, and `publication_block_reason` treats empty as "no block"::

    if gate is not None and not gate_is_live(gate):
        return blocked_reason(gate, "Validation gate")   # -> ""

The gate had failed and publication proceeded. Nothing reported it, because the
reason it would have reported was the empty string.

This is the shape the repository calls fail-open: the honest check
(`gate_is_live`) was right and the *explanation* was wrong, so the wrong
explanation was trusted. It was introduced while fixing the Society chain, where
the gate the supervisor assembles gained `validation_eligible` /
`publication_eligible` but its `passed` came from the critic's fidelity
comparison -- which was failing.

These tests pin the property directly, so removing the check fails them rather
than silently reopening the hole.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.agents.evidence_policy import (  # noqa: E402
    blocked_reason,
    gate_is_live,
    publication_block_reason,
    reproduction_is_live,
)


def _base_measurement_record():
    """Standard measurement record for test fixtures."""
    return {
        "model_identity": {"name": "gpt2", "content_hash": "abcd1234"},
        "input_commitment": "input123",
        "output_commitment": "output123",
        "execution_context": {"timestamp": 1234567890.0, "measurement_boundary_id": "boundary123"},
    }


def _failing_gate(**overrides):
    """A gate that is live, completed, attested, eligible -- and did not pass.

    Every field `blocked_reason` inspects before the `passed` check is
    satisfied on purpose. Only `passed` is False, so any block reported comes
    from the `passed` check and nothing else.
    """
    gate = {
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "validation_eligible": True,
        "publication_eligible": True,
        "validated": True,
        "passed": False,
        "metric": "overall_fidelity_pct",
        "value": 64.72,
        "threshold": 0.85,
        "measurement_record": _base_measurement_record(),
    }
    gate.update(overrides)
    return gate


def _trace_with_discovery():
    """A trace whose discovery and validation steps are both live and complete."""
    base_mr = _base_measurement_record()
    discovery = {
        "node": "discover", "status": "completed", "provenance": "live",
        "attested": True, "validation_eligible": True,
        "publication_eligible": True, "measured": True,
        "measurement_record": _base_measurement_record(),
    }
    validation = {
        "node": "validate", "status": "completed", "provenance": "live",
        "attested": True, "validation_eligible": True,
        "publication_eligible": True, "validated": True,
        "result": {
            "status": "completed", "provenance": "live", "attested": True,
            "validation_eligible": True, "publication_eligible": True,
            "validated": True,
            "measurement_record": _base_measurement_record(),
        },
    }
    return [discovery, validation]


def test_gate_is_live_still_refuses_a_failing_fidelity_gate():
    """The honest check. This one was never wrong."""
    assert gate_is_live(_failing_gate()) is False


def test_a_failing_gate_produces_a_nonempty_block_reason():
    """The check that was missing. Without it, the reason is ""."""
    reason = blocked_reason(_failing_gate(), "Validation gate")

    assert reason, (
        "a gate that failed its fidelity threshold produced no block reason, "
        "so `publication_block_reason` read the empty string as 'no block' "
        "and publication proceeded over a failed measurement")
    assert "fidelity gate did not pass" in reason


def test_the_block_reason_names_the_metric_value_and_threshold():
    """A reader must be told which number fell short."""
    reason = blocked_reason(_failing_gate(), "Validation gate")

    assert "overall_fidelity_pct" in reason
    assert "64.72" in reason
    assert "0.85" in reason


def test_publication_is_blocked_when_only_the_fidelity_check_fails():
    """End to end: a trace and a gate that fail *only* on fidelity."""
    trace = _trace_with_discovery()

    reason = publication_block_reason(trace, reproducibility=None,
                                      gate=_failing_gate())

    assert reason, (
        "publication was allowed when the fidelity gate failed; this is the "
        "silent-bypass hole the `passed` check closes")


def test_a_passing_gate_is_not_blocked_for_this_reason():
    """The negative control, so the check above is not just always-failing."""
    gate = _failing_gate(passed=True)

    reason = publication_block_reason(
        _trace_with_discovery(), reproducibility=None, gate=gate)

    assert reason == "", (
        f"a passing gate was blocked: {reason!r}")


def test_a_gate_with_no_metric_still_blocks():
    """The message degrades, the block does not."""
    gate = _failing_gate()
    gate.pop("metric"), gate.pop("value"), gate.pop("threshold")

    reason = blocked_reason(gate, "Validation gate")
    assert reason, "a gate with no metric detail was allowed to pass silently"
    assert "did not pass" in reason


def test_the_critic_forwards_the_measurement_record():
    """`attested: True` without the record is not attestation any more.

    The Critic rebuilds its own result dict and forwarded `attested` but not
    `measurement_record`, so `_is_attested(critic_output)` was False for a run
    whose pipeline record was complete. The Society applies exactly that
    predicate to this object -- `publication_block_reason` calls
    `reproduction_is_live(reproducibility)` on the critic's return -- so every
    Society publication was blocked at "Reproduction" while looking like it had
    passed the fidelity gate.

    The gate needs live GPT-2 weights, so this is skipped when they are absent;
    the plumbing is still pinned by the interface tests above.
    """
    from backend.interpretability.discovery.live_discovery import (
        LiveIOIDiscovery,
    )

    if not LiveIOIDiscovery.available():
        pytest.skip("live GPT-2 weights unavailable")

    from backend.agents.critic import Critic

    repro = Critic().reproduce("ioi", n_prompts=8)

    assert repro.get("status") == "completed", repro.get("reason")
    record = repro.get("measurement_record")
    assert isinstance(record, dict), (
        "the Critic did not forward the pipeline's measurement_record, so a "
        "complete record reads as unattested to every caller of this result")
    assert set(record) >= {"model_identity", "input_commitment",
                          "output_commitment", "execution_context"}, (
        "the forwarded record is missing a binding field")
    assert reproduction_is_live(repro), (
        "the Critic's own output still fails reproduction_is_live; the Society "
        "publication path applies that predicate to this object")


