"""EvidenceBoundary: a publishable result cannot be produced by asserting one.

Two properties are pinned here, and they are different things.

**A claim is not a proof.** A live result requires an Ed25519 signature over a
canonical payload that includes a digest of the measurement itself. This is the
property the old boundary did not have: it checked that each hash was 64 hex
characters, which any caller can satisfy by typing a string. The docstring
claimed a caller "cannot construct a passing attestation for a run that did not
happen", and the exact counterexample in the audit worked -- so the check was
replaced rather than the counterexample being dismissed.

**A result cannot be forged by constructing one.** `EvidenceResult` carries a
capability only the boundary holds. The old class derived `publishable` from
status, provenance and eligibility and never consulted the attestation, so
`EvidenceResult(status="completed", provenance="live", executor_id="x",
eligibility={"publication_eligible": True})` was publishable without `admit()`
ever being called.

Guards here are behavioural: each forgery is actually constructed and the
boundary is actually asked. Grepping the source for banned spellings would not
have caught any of the counterexamples below.
"""
import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from backend.core import evidence_boundary as _boundary_module
from backend.core.evidence_boundary import (
    BOUNDARY,
    EvidenceBoundary,
    EvidenceResult,
    RunAttestation,
    digest_of,
    sha256_file,
)
from backend.agents.evidence_policy import discovery_is_live, validation_is_live


def _sha(*parts: str) -> str:
    return "sha256:" + hashlib.sha256("|".join(parts).encode()).hexdigest()


@pytest.fixture(scope="module")
def key() -> bytes:
    """A real Ed25519 seed, 32 raw bytes.

    Module-scoped and generated once: every attestation in this file is signed
    with the same key, which is what lets a test sign *as the attacker* and
    still be refused, because refusal here is about binding rather than about
    not holding a key.
    """
    ed25519 = pytest.importorskip(
        "cryptography.hazmat.primitives.asymmetric.ed25519")
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    return ed25519.Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )


@pytest.fixture(scope="module")
def other_key() -> bytes:
    ed25519 = pytest.importorskip(
        "cryptography.hazmat.primitives.asymmetric.ed25519")
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    return ed25519.Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )


def _signed(key: bytes, measurement: dict, **over) -> RunAttestation:
    """A genuinely signed attestation for `measurement`."""
    base = dict(
        run_id="run_123",
        executor_id="gpt2_engine",
        model_id="gpt2-small",
        measurement=measurement,
        weights_sha256=_sha("weights"),
        dataset_sha256=_sha("dataset"),
        code_revision="abc1234",
        private_key=key,
    )
    base.update(over)
    return RunAttestation.issue(**base)


def _admit(measurement: dict, key: bytes, **over):
    return BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                          measurement=measurement,
                          attestation=_signed(key, measurement, **over))


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
        a = RunAttestation(run_id="run_123", executor_id="gpt2_engine",
                           model_id="gpt2-small", weights_sha256=bad,
                           dataset_sha256=_sha("dataset"),
                           code_revision="abc1234", execution_id="exec-9",
                           model_loaded="gpt2-small",
                           measurement_sha256=_sha("m"))
        assert a.verifies() is False, bad
        r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                           measurement={"score": 1.0}, attestation=a)
        assert r.publishable is False
        assert "weights_sha256" in r.reason


def test_attestation_missing_fields_is_rejected():
    a = RunAttestation(run_id="run_123", executor_id="gpt2_engine",
                       model_id="gpt2-small", code_revision="", dataset_sha256="",
                       weights_sha256=_sha("w"), execution_id="exec-9",
                       model_loaded="gpt2-small", measurement_sha256=_sha("m"))
    assert not a.verifies()
    r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                       measurement={"x": 1}, attestation=a)
    assert r.provenance == "unavailable"
    assert "code_revision" in r.reason
    assert "dataset_sha256" in r.reason


# ── The shape check that used to be the whole check ──────────────────────────

def test_shape_correct_but_unsigned_attestation_is_refused(key):
    """The audit's counterexample, verbatim, and it must now fail.

    Every field is well-formed: 64 hex characters, no placeholders, a run id, a
    code revision. Under the old boundary this verified and became publishable.
    The only thing missing is a signature, which is the only thing that proves
    anything.
    """
    measurement = {"score": 0.94}
    forged = RunAttestation(
        run_id="anything",
        executor_id="anything",
        model_id="gpt2",
        weights_sha256="0" * 64,
        dataset_sha256="1" * 64,
        code_revision="anything",
        execution_id="anything",
        model_loaded="gpt2",
        model_requested="gpt2",
        measurement_sha256=digest_of(measurement),
    )
    # Nothing is malformed here. That is the point: the old boundary checked
    # exactly this much and called it proof.
    assert forged._hash_problem("weights_sha256", forged.weights_sha256) == []
    assert forged._identity_problem() == []
    assert forged.verifies() is False
    assert any("no signature" in p for p in forged.problems())

    r = BOUNDARY.admit(executor_id="anything", provenance="live",
                       measurement=measurement, attestation=forged)
    assert r.publishable is False
    assert r.provenance == "unavailable"
    assert "no signature" in r.reason


def test_signed_attestation_is_accepted(key):
    measurement = {"circuit_faithfulness": 0.78}
    r = _admit(measurement, key)
    assert r.status == "completed"
    assert r.provenance == "live"
    assert r.publishable is True
    assert r.run_id == "run_123"
    assert r.attestation["weights_sha256"].startswith("sha256:")
    assert r.attestation_verified is True


# ── The signature has to be about *this* payload ─────────────────────────────

def test_payload_tampered_after_signing_is_refused(key):
    """Editing a signed field must break verification."""
    a = _signed(key, {"score": 0.5})
    # Sound on its own terms, and bound to the measurement it was signed for.
    assert a.verifies() is True
    assert a.verifies({"score": 0.5}) is True
    assert a.verifies({"score": 0.9}) is False

    tampered = replace(a, weights_sha256=_sha("some_other_weights"))
    assert tampered.verifies() is False
    assert any("does not verify" in p for p in tampered.problems())
    r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                       measurement={"score": 0.5}, attestation=tampered)
    assert r.publishable is False


def test_signature_from_a_different_key_is_refused(key, other_key):
    """A signature that verifies under the wrong identity must not pass."""
    a = _signed(key, {"score": 0.5})
    forged = replace(a, signature=_signed(other_key, {"score": 0.5}).signature)
    assert any("does not verify" in p for p in forged.problems())
    assert BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                          measurement={"score": 0.5},
                          attestation=forged).publishable is False


def test_substituted_public_key_is_refused(key, other_key):
    """Swapping the verifying key must be caught by the fingerprint.

    The signature still verifies under *some* key, and the record stays
    internally consistent, which is exactly why a shape check cannot see this.
    """
    from backend.science.integrity import public_key_hex

    a = _signed(key, {"score": 0.5})
    swapped = replace(a, public_key=public_key_hex(other_key))
    assert any("substituted" in p for p in swapped.problems())
    assert BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                          measurement={"score": 0.5},
                          attestation=swapped).publishable is False


# ── An attestation belongs to one measurement and one execution ──────────────

def test_attestation_cannot_be_reused_for_a_different_measurement(key):
    """A valid signature proves who signed, not what they signed about."""
    original = {"circuit_faithfulness": 0.78}
    stolen = {"circuit_faithfulness": 0.99}
    attestation = _signed(key, original)

    assert BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                          measurement=original,
                          attestation=attestation).publishable is True

    r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                       measurement=stolen, attestation=attestation)
    assert r.publishable is False
    assert "different result" in r.reason


def test_one_execution_cannot_attest_two_measurements(key):
    """Replay guard: an execution id is spent once."""
    first = {"score": 0.5}
    second = {"score": 0.9}
    assert _admit(first, key, execution_id="exec-replay").publishable is True

    r = _admit(second, key, execution_id="exec-replay")
    assert r.publishable is False
    assert "replay" in r.reason


def test_re_admitting_the_same_result_is_idempotent(key):
    """Re-deriving the same result from the same execution is not a replay."""
    measurement = {"score": 0.5}
    a = _signed(key, measurement, execution_id="exec-idem")
    assert BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                          measurement=measurement, attestation=a).publishable is True
    assert BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                          measurement=measurement, attestation=a).publishable is True


# ── Model identity ───────────────────────────────────────────────────────────

def test_requested_model_must_equal_loaded_model(key):
    """Audit P1: a mismatch must stop the result being live, not annotate it.

    This is the failure the engine itself documents:
    `infer("Hello", "gpt2-large")` returned `model_name: "gpt2-large"` and
    `d_model: 768`, i.e. gpt2-small's tensors, and the dispatcher stamped
    live on top of it.
    """
    measurement = {"d_model": 768}
    a = _signed(key, measurement, model_id="gpt2-small",
                model_requested="gpt2-large")
    assert any("model identity mismatch" in p for p in a.problems())
    r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                       measurement=measurement, attestation=a)
    assert r.provenance == "unavailable"
    assert r.publishable is False
    assert "model identity mismatch" in r.reason


def test_matching_request_and_load_is_accepted(key):
    r = _admit({"d_model": 768}, key, model_requested="gpt2-small")
    assert r.publishable is True


# ── The record cannot be forged ──────────────────────────────────────────────

def test_evidence_result_cannot_be_constructed_directly():
    with pytest.raises(PermissionError):
        EvidenceResult(status="completed", provenance="live",
                       executor_id="x",
                       eligibility={"publication_eligible": True})


def test_the_exact_publishable_bypass_is_closed():
    """The audit's snippet, which used to return a publishable object."""
    with pytest.raises(PermissionError):
        EvidenceResult(
            status="completed",
            provenance="live",
            executor_id="x",
            eligibility={"validation_eligible": True,
                         "publication_eligible": True},
            measurement={"score": 0.99},
        )


def test_publishable_stops_being_true_if_the_attestation_is_broken(key):
    """Admission is not a permanent grant; the property re-verifies.

    The attestation is a plain dict inside a frozen record, so a caller holding
    a reference can edit it after the fact. `publishable` therefore re-checks
    the signature rather than trusting the fact that `admit()` once said yes.
    """
    r = _admit({"score": 0.5}, key)
    assert r.publishable is True

    r.attestation["weights_sha256"] = _sha("evil_weights")
    assert r.publishable is False
    assert r.attestation_verified is False


def test_no_module_outside_the_boundary_constructs_a_result():
    """Structural guard: the only `EvidenceResult(` in the backend is here.

    A per-module scan of construction sites, not a grep for a banned spelling.
    The old failure mode was a guard that looked for the word "live" in prose
    and passed while a dict literal set it.
    """
    root = Path(__file__).resolve().parents[2] / "backend"
    boundary = "evidence_boundary.py"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == boundary:
            continue
        if "EvidenceResult(" in path.read_text(encoding="utf-8", errors="replace"):
            # AggregatedEvidenceResult is a different record, not this one.
            for line in path.read_text(encoding="utf-8",
                                       errors="replace").splitlines():
                if "EvidenceResult(" in line and "AggregatedEvidenceResult(" not in line:
                    offenders.append(f"{path.name}: {line.strip()[:70]}")
    assert offenders == [], offenders


def test_only_the_boundary_grants_publication_eligibility():
    source = inspect.getsource(_boundary_module)
    assert source.count('"publication_eligible": True') == 1


# ── Nothing is ever upgraded ─────────────────────────────────────────────────

@pytest.mark.parametrize("claimed", ["seeded", "reference", "unavailable"])
def test_non_live_claims_are_never_promoted(claimed):
    r = BOUNDARY.admit(executor_id="stub", provenance=claimed,
                       measurement={"score": 0.94},
                       attestation=RunAttestation(
                           run_id="r", executor_id="e", model_id="gpt2-small",
                           weights_sha256=_sha("w"), dataset_sha256=_sha("d"),
                           code_revision="abc", execution_id="x",
                           model_loaded="gpt2-small",
                           measurement_sha256=_sha("m")))
    assert r.provenance == claimed
    assert r.status == "blocked"
    assert r.publishable is False
    assert r.eligibility["publication_eligible"] is False


def test_unknown_provenance_is_an_error_not_a_pass():
    r = BOUNDARY.admit(executor_id="weird", provenance="totally-live",
                       measurement={"score": 1.0})
    assert r.status == "error"
    assert r.provenance == "unavailable"


def test_attested_run_with_no_measurement_is_not_a_finding(key):
    a = _signed(key, {"score": 1.0})
    r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                       measurement={}, attestation=a)
    assert r.provenance == "unavailable"
    assert "measured nothing" in r.reason


# ── Artifact digests can be re-proved against real bytes ─────────────────────

def test_recorded_artifact_hash_is_recomputable(key, tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"real tensor bytes")
    measurement = {"loss": 0.4}
    a = _signed(key, measurement, weights_path=weights)
    r = BOUNDARY.admit(executor_id="gpt2_engine", provenance="live",
                       measurement=measurement, attestation=a)
    assert r.publishable is True
    assert EvidenceBoundary.verify_artifacts(r, {"weights": weights}) == []

    weights.write_bytes(b"tampered tensor bytes")
    assert EvidenceBoundary.verify_artifacts(r, {"weights": weights}) != []


def test_an_unrecorded_artifact_is_reported_not_ignored(key, tmp_path):
    stray = tmp_path / "stray.bin"
    stray.write_bytes(b"something else")
    r = _admit({"loss": 0.4}, key)
    assert EvidenceBoundary.verify_artifacts(r, {"stray": stray}) != []


# ── The result is compatible with the existing policy ───────────────────────

def test_admitted_live_result_passes_evidence_policy(key):
    """A boundary result must satisfy the pre-existing Society gate."""
    r = _admit({"discovery_id": "disc_1"}, key)
    payload = r.as_payload()
    payload.update(r.eligibility)
    assert discovery_is_live(payload) is True

    blocked = BOUNDARY.admit(executor_id="stub", provenance="seeded",
                             measurement={"discovery_id": "disc_1"})
    bp = blocked.as_payload()
    bp.update(blocked.eligibility)
    assert discovery_is_live(bp) is False
    assert validation_is_live(bp) is False


def test_result_is_not_a_bare_dict(key):
    """Algorithms return measurements; callers receive a typed record."""
    r = _admit({"a": 1}, key)
    assert isinstance(r, EvidenceResult)
    assert not isinstance(r, dict)
    d = r.to_dict()
    assert d["publishable"] is True
    assert d["attestation"]["run_id"] == "run_123"


def test_as_payload_includes_field_provenance(key):
    r = _admit({"a": 1}, key)
    p = r.as_payload()
    assert p["field_provenance"]["measurement"] == "live"
    assert "provenance" not in p["field_provenance"]


def test_unavailable_and_error_helpers():
    u = BOUNDARY.unavailable("sae_loader", "weights were never fetched")
    assert u.status == "blocked" and u.publishable is False
    assert "never fetched" in u.reason

    e = BOUNDARY.error("ioi", "aborted")
    assert e.status == "error" and e.publishable is False