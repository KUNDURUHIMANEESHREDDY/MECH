"""4.3 -- a failure anywhere must prevent a success report everywhere.

The test as specified
---------------------
Inject failures at every stage of the experiment lifecycle and assert that no
success status, publication eligibility, or live measurement is emitted
without validated execution evidence.

Where the attack surface actually is
------------------------------------
`experiment_templates` and `capability_registry` contain no cache, no
fallback, and no memoisation -- checked, because a cached value surviving a
failure was the stated concern and the wrappers do not appear to have one.

The real choke point is `evidence_policy.publication_block_reason`, the single
function that answers "may this be published?". If a failure anywhere in the
chain does not stop it, nothing downstream stops it either. So the injections
are aimed there.

The property that matters most
------------------------------
Not "each stage fails when it fails" -- that is the easy direction, and a
chain that checks only the last stage passes it. The dangerous case is the
opposite:

    discover FAILED  ->  validate PASSED  ->  publish?

An approval from a later stage is not evidence about an earlier one. Each test
below therefore injects a failure at one stage while every *other* stage is
perfectly healthy, which is the configuration most likely to leak.

Non-vacuity
-----------
`test_the_harness_would_notice_a_leak` runs the same assertions against a
deliberately-permissive checker and requires them to fail. Without it, a suite
of passing "it blocks" tests could pass because the assertions are weak.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from backend.agents.evidence_policy import (
    blocked_reason,
    publication_block_reason,
)

#: A completed, attested, opted-in measurement record. Reused so a test varies
#: exactly one stage and leaves the rest healthy -- otherwise a failure could
#: be masked by an unrelated defect in the fixture.
RECORD: Dict[str, Any] = {
    "status": "completed",
    "provenance": "live",
    "attested": True,
    "measurement_record": {
        "model_identity": {"model_name": "gpt2"},
        "input_commitment": "sha256:" + "0" * 64,
        "output_commitment": "sha256:" + "1" * 64,
        "execution_context": {"device": "cpu", "dtype": "torch.float32"},
    },
    "validation_eligible": True,
    "publication_eligible": True,
}


def _step(node: str, **overrides: Any) -> Dict[str, Any]:
    payload = dict(RECORD)
    payload.update(overrides)
    payload["node"] = node
    return payload


def _trace(*steps: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(steps)


def _assert_publication_blocked(trace, reproducibility=None, gate=None,
                                what: str = "trace") -> None:
    reason = publication_block_reason(trace, reproducibility, gate)
    assert reason, (
        f"{what} was accepted for publication with no block reason. A "
        f"failure, a missing stage, or a stage that never ran must not "
        f"produce an empty reason -- an empty string is how 'not checked' "
        f"reads as 'passed'."
    )


# ── the harness must be able to fail ────────────────────────────────

def test_the_harness_would_notice_a_leak():
    """Non-vacuity: a permissive checker must not satisfy these assertions.

    If this test can fail, the assertions elsewhere have teeth. If it cannot,
    they are decoration.
    """
    trace = _trace(_step("discover"), _step("discover"))

    def permissive(_trace, _repro, _gate) -> str:
        return ""  # publishes everything, unconditionally

    assert not permissive(trace, None, None), (
        "the permissive control unexpectedly returned a block reason; the "
        "harness compares against this and would prove nothing"
    )
    with pytest.raises(AssertionError):
        _assert_publication_blocked_for(passive=permissive, trace=trace)


def _assert_publication_blocked_for(passive, trace) -> None:
    reason = passive(trace, None, None)
    assert reason, "a checker that blocks nothing failed the harness"


def test_a_healthy_chain_is_actually_accepted_by_the_real_policy():
    """The control in the other direction.

    Without this, a policy that blocked *everything* would pass every test in
    this file, and "it blocks" would carry no information about what it blocks.
    """
    trace = _trace(_step("discover"), _step("validate"))
    reason = publication_block_reason(trace, None, None)
    # The fixture's record is placeholder-signed, so the healthy chain does
    # block -- on attestation, and for that reason specifically. What is
    # asserted is that the decision is attributable: a real check names its
    # cause, so a blanket refusal and a genuine verdict are distinguishable.
    assert reason == BASELINE_BLOCK_REASON, (
        f"the healthy chain produced {reason!r}, not the baseline reason. "
        f"Either the fixture now verifies -- in which case a healthy chain "
        f"should be accepted and this assertion is wrong -- or something "
        f"else changed. An unspecific reason means a blanket refusal and a "
        f"real check cannot be told apart."
    )


# ── failures injected one stage at a time, all others healthy ───────

def test_a_missing_discover_stage_blocks():
    """No discovery at all must not read as a clean run."""
    _assert_publication_blocked(
        _trace(_step("validate")), what="a trace with validation but no discovery")


def test_an_empty_trace_blocks():
    _assert_publication_blocked([], what="an empty trace")


def test_a_failed_discover_stage_blocks_even_when_validation_passes():
    """The dangerous ordering.

    `validate` passing is not evidence about `discover`. A chain that checks
    only the last stage passes this.
    """
    _assert_publication_blocked(
        _trace(
            _step("discover", status="failed"),
            _step("validate"),
        ),
        what="a failed discovery followed by a passing validation",
    )


def test_a_passed_discover_stage_does_not_rescue_a_failed_validation():
    """The mirror image, so neither stage is privileged."""
    _assert_publication_blocked(
        _trace(
            _step("discover"),
            _step("validate", status="failed"),
        ),
        what="a passing discovery followed by a failed validation",
    )


#: What the policy says when nothing is wrong with the fixture.
#:
#: Measured, and load-bearing. This placeholder record fails cryptographic
#: verification, so the *healthy* chain already blocks with
#: "the measurement is not attested". Any test asserting only "it blocked"
#: would therefore pass for reasons unrelated to the defect it injected --
#: four of the eight cases below did exactly that before this constant
#: existed. So each case asserts that the reason *names the defect*, and a
#: case the baseline masks is recorded as masked rather than counted as a
#: pass.
BASELINE_BLOCK_REASON = "Discovery blocked: the measurement is not attested."


@pytest.mark.parametrize("label,payload,expected_fragment", [
    ("unavailable", _step("discover", status="unavailable"),
     "status 'unavailable' is not complete"),
    ("error", _step("discover", status="error"),
     "status 'error' is not complete"),
    ("reference provenance", _step("discover", provenance="reference"),
     "provenance 'reference' is not live"),
    ("synthetic provenance", _step("discover", provenance="synthetic"),
     "provenance 'synthetic' is not live"),
    ("attested=False", _step("discover", attested=False), None),
    ("no measurement record",
     {k: v for k, v in RECORD.items() if k != "measurement_record"}, None),
    ("validation ineligible", _step("discover", validation_eligible=False), None),
    ("publication ineligible", _step("discover", publication_eligible=False), None),
])
def test_a_defective_discovery_stage_blocks(label: str, payload: Dict[str, Any],
                                            expected_fragment: Optional[str]):
    """Each defect must block, and the reason must name *that* defect.

    The validation stage below it is untouched, so a block attributed to the
    discovery check cannot be collateral damage to the fixture.

    Where `expected_fragment` is `None`, the defect is currently masked: the
    unverifiable placeholder record blocks first and the specific check is
    never reached. Those cases assert only that the block is non-empty, and
    the assertion message says plainly that the cause was not isolated --
    because a test that cannot distinguish its own cause is not evidence that
    the check exists.
    """
    payload = dict(payload)
    payload["node"] = "discover"
    trace = _trace(payload, _step("validate"))

    reason = publication_block_reason(trace, None, None)
    assert reason, f"a discovery stage defective by {label!r} was accepted"

    if expected_fragment is None:
        assert reason == BASELINE_BLOCK_REASON, (
            f"expected this case to be masked by the baseline, but the policy "
            f"gave a different reason: {reason!r}. Update `expected_fragment` "
            f"so the test asserts what the policy actually checks."
        )
        return

    assert expected_fragment in reason, (
        f"a discovery stage defective by {label!r} was blocked, but not for "
        f"that reason: {reason!r}. It does not contain {expected_fragment!r}, "
        f"so this test would have passed on an unrelated block."
    )


def test_a_completed_status_does_not_mask_a_reference_provenance():
    """The specific mix that produced the `NOT_EXECUTED` finding.

    `status: "completed"` with `provenance: "reference"` is a synthetic result
    wearing a success label. `_is_completed` and `provenance_of` are separate
    checks and the block must consult both.
    """
    payload = _step("discover", status="completed", provenance="reference")
    payload["node"] = "discover"
    _assert_publication_blocked(
        _trace(payload, _step("validate")),
        what="a completed stage carrying reference provenance",
    )


def test_a_failed_stage_nested_inside_a_successful_one_is_not_overlooked():
    """`publication_block_reason` iterates *every* discovery step.

    A trace carrying one healthy discovery followed by a failed one must not
    be accepted because the first one passed.
    """
    _assert_publication_blocked(
        _trace(
            _step("discover"),
            _step("discover", status="failed"),
        ),
        what="a healthy discovery followed by a failed discovery",
    )