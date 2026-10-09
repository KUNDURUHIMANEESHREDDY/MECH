"""A `live` label is a claim; `attested` is a vouch. They are not the same.

`attest_measurement` is the only function in the codebase permitted to set
`attested: True`, and that flag is what gates evidence:

    backend/agents/evidence_policy.py:17   .get("attested") is True
    backend/core/evidence_graph.py:86      if step.get("attested") is not True

So issuing that flag is issuing a scientific claim. The pre-fix code did this:

    existing = result.get("provenance")
    if isinstance(existing, str) and existing.strip():
        if existing.strip().lower() == LIVE:
            result["attested"] = True      # <-- from the label alone
        return result

Two defects follow from those four lines.

**Laundering.** Any component that builds a result dict can write
`{"provenance": "live", "value": 42}` and pass it through any wrapper that
calls `attest_measurement`. The function would then grant `attested: True`
without having measured anything. The docstring called this "repairing a
broken invariant", but `provenance` is an ordinary dict key -- the function
cannot tell a deliberate choice by the measurement layer from a value a caller
typed in, and treating them identically is exactly what makes the label forgeable.

**A skipped guard.** That branch `return`s at line 127, so the
`is_failure_status` check below it never runs for a pre-labelled record. A
result could claim `status: "failed"` and still come back attested.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core.provenance import (  # noqa: E402
    LIVE, UNAVAILABLE, attest_measurement, withhold,
)
from backend.agents.evidence_policy import _is_attested  # noqa: E402


# â”€â”€ the laundering path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ #

def test_a_bare_live_label_is_not_upgraded_to_attested():
    """The core defect: a label a caller typed in is not a measurement."""
    out = attest_measurement({"provenance": "live", "value": 42.0})

    assert out.get("attested") is not True, (
        "attest_measurement granted attestation from a pre-existing 'live' "
        "label alone -- anyone constructing a result dict can mint a "
        "scientific claim")
    assert out["provenance"] != LIVE


def test_the_laundered_record_carries_a_reason():
    out = attest_measurement({"provenance": "live", "value": 42.0})
    reason = str(out.get("reason", ""))
    assert reason, "the refusal must say why, not just withhold silently"
    assert "live" in reason and "did not issue" in reason


def test_a_live_label_cannot_outrank_a_failure_status():
    """The early `return` skipped the failure check entirely."""
    for status in ("error", "unavailable", "failed", "not_run"):
        out = attest_measurement({"provenance": "live", "status": status,
                                  "value": 1.0})
        assert out.get("attested") is not True, (
            f"status={status!r} was attested because the record already "
            "carried a 'live' label")
        assert out["provenance"] != LIVE, status


def test_a_forged_label_cannot_satisfy_the_evidence_gate():
    """The point of the flag: it gates evidence, so laundering it matters."""
    forged = attest_measurement({"provenance": "live", "value": 42.0,
                                 "status": "completed"})
    assert _is_attested(forged) is False, (
        "a record whose 'live' label was typed in by a caller satisfied the "
        "attestation gate")


# â”€â”€ what must keep working â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ #

def test_a_real_measurement_is_still_attested():
    out = attest_measurement({"status": "ok", "value": 1.0})
    assert out["provenance"] == LIVE
    assert out["attested"] is True
    assert _is_attested(out) is True


def test_re_wrapping_a_genuine_measurement_is_not_attested_twice():
    """There is no legitimate double-attest, and the docs say why.

    This module's contract routes wrappers and orchestrators through
    `pass_through` ("never invents `live`"), so a record arriving at
    `attest_measurement` already labelled `live` came from somewhere that had
    no right to assert it. Re-attesting one therefore withholds it -- and that
    is the safe direction to be wrong in.
    """
    once = attest_measurement({"status": "ok", "value": 1.0})
    assert once["provenance"] == LIVE and once["attested"] is True

    twice = attest_measurement(dict(once))

    assert twice.get("attested") is False, (
        "re-attesting an already-labelled record was treated as idempotent; "
        "the idempotency check itself was the laundering hole")
    assert twice["provenance"] != LIVE


def test_a_specific_non_live_label_is_still_respected():
    """`seeded` keeps its label and gains no model bookkeeping."""
    out = attest_measurement({"provenance": "seeded", "value": 1.0})
    assert out["provenance"] == "seeded"
    assert "model_mismatch" not in out
    assert out.get("attested") is not True


def test_failures_are_still_withheld_with_a_reason():
    out = attest_measurement({"status": "unavailable", "value": 1.0})
    assert out["provenance"] == UNAVAILABLE
    assert out["attested"] is False
    assert "nothing was measured" in out["reason"]


def test_model_identity_is_still_recorded():
    out = attest_measurement({"value": 1.0}, model_loaded="gpt2",
                             model_requested="gpt2-large")
    assert out["provenance"] == LIVE
    assert out["model_loaded"] == "gpt2"
    assert out["model_mismatch"] is True


def test_withhold_still_works_standalone():
    assert withhold({"status": "ok"}, reason="r")["attested"] is False


def test_a_non_dict_is_returned_untouched():
    sentinel = object()
    assert attest_measurement(sentinel) is sentinel


# â”€â”€ the flag has exactly one issuer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ #

def test_writing_both_keys_does_not_launder():
    """A tempting 'fix' is to exempt records that also carry `attested: True`.

    That is the same defect one key over: `attested` is caller-writable too, so
    `{"provenance": "live", "attested": True}` would mint a scientific claim
    just as easily. There is no exemption.
    """
    out = attest_measurement({"provenance": "live", "attested": True,
                              "value": 42.0})
    assert out.get("attested") is False, (
        "attestation was granted because the record claimed to be attested "
        "already -- but that claim is exactly as forgeable as the label")
    assert out["provenance"] != LIVE


def test_a_genuine_measurement_keeps_its_attestation_when_passed_on():
    """Wrappers are routed through `pass_through`, which must not strip it."""
    from backend.core.provenance import pass_through

    measured = attest_measurement({"status": "ok", "value": 1.0})
    assert _is_attested(pass_through(measured)) is True, (
        "pass_through discarded a genuine attestation")


@pytest.mark.parametrize("label", ["live", "LIVE", " Live ", "live "])
def test_label_matching_is_case_and_space_insensitive(label):
    """Otherwise `live` could be smuggled past by writing `Live`."""
    out = attest_measurement({"provenance": label, "value": 1.0})
    assert out.get("attested") is not True, label
    assert out["provenance"] != LIVE, label
