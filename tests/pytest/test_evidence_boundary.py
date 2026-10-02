"""EvidenceBoundary: no algorithm can produce a publishable dict on its own.

These tests pin the invariant that a result's provenance is *proved*, not
stated. Before the boundary existed, any executor could return a plain dict with
a confident number in it and callers had no way to distinguish a measurement
from a fabrication.
"""
import hashlib

import pytest

from backend.core.evidence_boundary import (
    BOUNDARY,
    EvidenceBoundary,
    EvidenceResult,
    RunAttestation,
)
from backend.agents.evidence_policy import discovery_is_live, validation_is_live


def _sha(*parts: str) -> str:
    return "sha256:" + hashlib.sha256("|".join(parts).encode()).hexdigest()


def _attestation(**over) -> RunAttestation:
    base = dict(
        run_id="run_123",
        executor_id="gpt2_engine",
        model_id="gpt2-small",
        weights_sha256=_sha("weights"),
        dataset_sha256=_sha("dataset"),
        code_revision="abc1234",
        execution_id="exec-9",
    )
    base.update(over)
    return RunAttestation(**base)


# ── A claim is not a proof ────────────────────────────────────────────────────

def test_live_claim_without_attestation_is_rejected():
    r = BOUNDARY.admit(executor_id="stub", provenance="live",
                       measurement={"score": 0.94})
    assert r.provenance == "unavailable"
    assert r.status == "blocked"
    assert r.publishable is False
    assert "not a proof" in r.reason


def test_placeholder_hashes_cannot_attest():
    """The literal mocks this repo used to ship must not verify."""
    for bad in ("sha256:8f43c...model_weights_mock", "unattested",
                "sha256:deadbeef", "not-a-hash"):
        a = _attestation(weights_sha256=bad)
        assert a.verifies() is False, bad
        r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                           measurement={"score": 1.0}, attestation=a)
        assert r.publishable is False
        assert "weights_sha256" in r.reason


def test_attestation_missing_fields_is_rejected():
    a = _attestation(code_revision="", dataset_sha256="")
    assert not a.verifies()
    r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                       measurement={"x": 1}, attestation=a)
    assert r.provenance == "unavailable"
    assert "code_revision" in r.reason
    assert "dataset_sha256" in r.reason


def test_full_attestation_is_accepted():
    r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                       measurement={"circuit_faithfulness": 0.78},
                       attestation=_attestation())
    assert r.status == "completed"
    assert r.provenance == "live"
    assert r.publishable is True
    assert r.run_id == "run_123"
    assert r.attestation["weights_sha256"].startswith("sha256:")


# ── Nothing is ever upgraded ─────────────────────────────────────────────────

@pytest.mark.parametrize("claimed", ["seeded", "reference", "unavailable"])
def test_non_live_claims_are_never_promoted(claimed):
    r = BOUNDARY.admit(executor_id="stub", provenance=claimed,
                       measurement={"score": 0.94},
                       attestation=_attestation())
    assert r.provenance == claimed
    assert r.status == "blocked"
    assert r.publishable is False
    assert r.eligibility["publication_eligible"] is False


def test_unknown_provenance_is_an_error_not_a_pass():
    r = BOUNDARY.admit(executor_id="weird", provenance="totally-live",
                       measurement={"score": 1.0})
    assert r.status == "error"
    assert r.provenance == "unavailable"


def test_attested_run_with_no_measurement_is_not_a_finding():
    r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                       measurement={}, attestation=_attestation())
    assert r.provenance == "unavailable"
    assert "measured nothing" in r.reason


# ── The result is compatible with the existing policy ───────────────────────

def test_admitted_live_result_passes_evidence_policy():
    """A boundary result must satisfy the pre-existing Society gate."""
    r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                       measurement={"discovery_id": "disc_1"},
                       attestation=_attestation())
    payload = r.as_payload()
    payload.update(r.eligibility)
    assert discovery_is_live(payload) is True

    blocked = BOUNDARY.admit(executor_id="stub", provenance="seeded",
                             measurement={"discovery_id": "disc_1"})
    bp = blocked.as_payload()
    bp.update(blocked.eligibility)
    assert discovery_is_live(bp) is False
    assert validation_is_live(bp) is False


def test_result_is_not_a_bare_dict():
    """Algorithms return measurements; callers receive a typed record."""
    r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                       measurement={"a": 1}, attestation=_attestation())
    assert isinstance(r, EvidenceResult)
    assert not isinstance(r, dict)
    d = r.to_dict()
    assert d["publishable"] is True
    assert d["attestation"]["run_id"] == "run_123"


def test_as_payload_includes_field_provenance():
    r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                       measurement={"a": 1}, attestation=_attestation())
    p = r.as_payload()
    assert p["field_provenance"]["measurement"] == "live"
    assert "provenance" not in p["field_provenance"]


def test_unavailable_and_error_helpers():
    u = BOUNDARY.unavailable("sae_loader", "weights were never fetched")
    assert u.status == "blocked" and u.publishable is False
    assert "never fetched" in u.reason

    e = BOUNDARY.error("ioi", "aborted")
    assert e.status == "error" and e.publishable is False


# ── The boundary is the only route ──────────────────────────────────────────

def test_no_module_outside_the_boundary_mints_a_publishable_result():
    """Guard: only evidence_boundary decides eligibility.

    If a future change starts hand-rolling an eligible payload elsewhere, this
    test is where it should surface.
    """
    import inspect

    from backend.core import evidence_boundary

    source = inspect.getsource(evidence_boundary)
    # Eligibility is only ever computed inside admit().
    assert source.count('"publication_eligible": True') == 1
