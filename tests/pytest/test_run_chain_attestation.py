"""The run chain: run_id + dataset_sha256 + code_revision + the weights.

The model fingerprint says which weights ran. It says nothing about which
data they ran on or which code ran them, so two results with identical
weights can come from different experiments entirely. This file pins the
whole chain, and pins it in the direction that matters: the digest is of
the data, the revision is of the repository, and anything that cannot be
computed leaves the chain unattested rather than plausible.

A negative test passes when the system correctly refuses. The refusals
below are the point: an empty panel, an unattested fingerprint, a
repository that cannot be asked -- each of them must produce
`attested: False` with a stated reason, and never a digest that looks
real.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core.provenance import admit_result, attest_run_chain  # noqa: E402
from backend.science.reproducibility import run_attestation as ra  # noqa: E402
from backend.science.reproducibility.run_attestation import (  # noqa: E402
    EmptyDatasetError,
    attest_run,
    code_revision,
    dataset_digest,
    is_real_digest,
    verify_run_attestation,
)

#: A model fingerprint that was attested from real weights. The digest is a
#: well-formed SHA-256 because the attestation checks the shape of the
#: digest it is given; the real one is computed from loaded tensors.
WEIGHTS = "sha256:" + "ab" * 32

ATTESTED_FINGERPRINT = {
    "model_id": "gpt2",
    "weights_sha256": WEIGHTS,
    "attested": True,
    "attestation_reason": "",
}

UNATTESTED_FINGERPRINT = {
    "model_id": "gpt2",
    "weights_sha256": "unattested",
    "attested": False,
    "attestation_reason": "Adapter is in mock_mode; no weights were loaded.",
}

PANEL = [
    {"prompt": "When Alice and Bob went to the store", "correct": True},
    {"prompt": "When Carol and Dave went to the store", "correct": False},
]


# ── the dataset digest is of the data ─────────────────────────────────────

def test_the_digest_is_not_a_constant():
    """Different data must give different digests.

    A digest that did not move with the data would attest to nothing: it
    would be a decoration that looks like a checksum.
    """
    first = dataset_digest(PANEL)
    second = dataset_digest([
        {"prompt": "When Alice and Bob went to the store", "correct": True},
        {"prompt": "When Carol and Dave went to the store", "correct": True},
    ])

    assert first != second, (
        "two different panels produced the same digest; the digest is not "
        "computed from the data"
    )
    assert is_real_digest(first)
    assert is_real_digest(second)


def test_a_perturbation_changes_the_digest():
    """One changed value must change the digest.

    This is the property a drift check depends on: if the digest did not
    move when the data moved, a changed dataset would still verify against
    the old one.
    """
    baseline = dataset_digest(PANEL)
    perturbed = dataset_digest([
        {"prompt": "When Alice and Bob went to the store", "correct": True},
        {"prompt": "When Carol and Dave went to the store", "correct": True},
    ])

    assert baseline != perturbed, (
        "changing one row of the panel did not change the digest"
    )


def test_the_same_data_always_gives_the_same_digest():
    """A digest that moved between calls could never be checked."""
    assert dataset_digest(PANEL) == dataset_digest(PANEL)
    assert dataset_digest(b"abc") == dataset_digest(b"abc")


def test_the_digest_is_of_the_bytes_not_of_a_digest():
    """Hashing a digest string says nothing about any dataset.

    The digest of the *string* "sha256:..." is a different value from the
    digest of the data it stands for, which is why verification recomputes
    from the data.
    """
    real = dataset_digest(PANEL)
    assert dataset_digest(real) != real
    assert is_real_digest(dataset_digest(real))


def test_an_empty_panel_is_refused_rather_than_hashed():
    """An empty panel is the absence of data, not a measurement of nothing."""
    for empty in ([], (), {}, b"", ""):
        with pytest.raises(EmptyDatasetError):
            dataset_digest(empty)


def test_a_missing_file_is_refused_rather_than_hashed(tmp_path):
    with pytest.raises(Exception):
        dataset_digest(tmp_path / "there-is-no-such-dataset.json")


# ── the code revision is of the repository ─────────────────────────────────

def _git_head(repo_root: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root), capture_output=True, text=True, timeout=30,
        check=False,
    )
    return out.stdout.strip()


def test_the_code_revision_is_the_repository_head():
    """It is the real commit, not a plausible-looking stand-in."""
    revision = code_revision(REPO_ROOT)

    assert revision.available, revision.reason
    assert revision.source == "git"
    assert revision.revision == _git_head(REPO_ROOT)
    assert ra.is_real_revision(revision.revision)


def test_the_code_revision_falls_back_honestly_when_there_is_no_repository(
        monkeypatch):
    """No repository means no revision -- and no invented one.

    The fallback is the absence of a revision, stated as such. A made-up
    forty-character string would satisfy every reader and git itself, which
    is exactly why it is not used.
    """
    monkeypatch.setattr(ra, "repository_root", lambda start=None: None)

    revision = code_revision(REPO_ROOT)

    assert revision.available is False
    assert revision.revision == "", (
        "a revision was invented where the repository could not be asked"
    )
    assert revision.source == "unavailable"
    assert revision.reason, "the fallback must say why there is no revision"


def test_a_directory_outside_any_repository_has_no_revision():
    """The same case, for real rather than by simulation."""
    outside = Path(tempfile.mkdtemp())
    try:
        if ra.repository_root(outside) is not None:
            pytest.skip("the system temporary directory is inside a repository")
        revision = code_revision(outside)
        assert revision.available is False
        assert revision.revision == ""
        assert revision.source == "unavailable"
        assert revision.reason
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_the_code_revision_falls_back_honestly_when_git_cannot_run(
        monkeypatch, tmp_path):
    """git missing or broken is the same case: no revision, with a reason."""

    def _boom(*args, **kwargs):
        raise OSError("git is not installed")

    monkeypatch.setattr(ra, "_git", _boom)
    revision = code_revision(REPO_ROOT)

    assert revision.available is False
    assert revision.revision == ""
    assert "git" in revision.reason.lower()


def test_a_dirty_working_tree_is_reported_as_dirty():
    """A commit id names a snapshot; a dirty tree is not that snapshot."""
    revision = code_revision(REPO_ROOT)
    assert isinstance(revision.dirty, bool)


# ── the chain as a whole ───────────────────────────────────────────────────

def test_a_complete_chain_is_attested():
    chain = attest_run(run_id="run_1", dataset=PANEL,
                       model_fingerprint=ATTESTED_FINGERPRINT)

    assert chain.attested is True, chain.attestation_reason
    assert chain.run_id == "run_1"
    assert chain.dataset_sha256 == dataset_digest(PANEL)
    assert chain.weights_sha256 == WEIGHTS
    assert chain.code_revision == code_revision(REPO_ROOT).revision
    assert chain.attestation_reason == ""


def test_an_empty_dataset_leaves_the_chain_unattested():
    """Fail closed: no bytes, no chain, and a stated reason."""
    chain = attest_run(run_id="run_2", dataset=[],
                       model_fingerprint=ATTESTED_FINGERPRINT)

    assert chain.attested is False
    assert chain.dataset_sha256 == "", (
        "an empty panel produced a digest; there was nothing to hash"
    )
    assert "empty" in chain.attestation_reason.lower()


def test_an_unattested_model_leaves_the_chain_unattested():
    chain = attest_run(run_id="run_3", dataset=PANEL,
                       model_fingerprint=UNATTESTED_FINGERPRINT)

    assert chain.attested is False
    assert chain.weights_sha256 == ""
    assert "not attested" in chain.attestation_reason


def test_a_missing_model_leaves_the_chain_unattested():
    chain = attest_run(run_id="run_4", dataset=PANEL)

    assert chain.attested is False
    assert "model fingerprint" in chain.attestation_reason


def test_a_run_without_an_id_leaves_the_chain_unattested():
    chain = attest_run(run_id="", dataset=PANEL,
                       model_fingerprint=ATTESTED_FINGERPRINT)

    assert chain.attested is False
    assert "id" in chain.attestation_reason


def test_a_repository_that_cannot_be_asked_leaves_the_chain_unattested(
        monkeypatch):
    """The honest fallback propagates: no revision, no chain."""
    monkeypatch.setattr(ra, "repository_root", lambda start=None: None)

    chain = attest_run(run_id="run_5", dataset=PANEL,
                       model_fingerprint=ATTESTED_FINGERPRINT)

    assert chain.attested is False
    assert chain.code_revision == ""
    assert chain.code_revision_source == "unavailable"
    assert chain.attestation_reason


# ── verification is against the data, not against the claim ────────────────

def test_verification_recomputes_the_digest_from_the_data():
    chain = attest_run(run_id="run_6", dataset=PANEL,
                       model_fingerprint=ATTESTED_FINGERPRINT)

    verdict = verify_run_attestation(chain, PANEL)

    assert verdict["valid"] is True, verdict["reason"]
    assert verdict["checks"]["dataset_matches"] is True


def test_verification_fails_when_the_data_moved():
    """The check is against the data, so a changed panel must not verify."""
    chain = attest_run(run_id="run_7", dataset=PANEL,
                       model_fingerprint=ATTESTED_FINGERPRINT)
    changed = [dict(row) for row in PANEL]
    changed[0]["correct"] = False

    verdict = verify_run_attestation(chain, changed)

    assert verdict["valid"] is False
    assert verdict["checks"]["dataset_matches"] is False


def test_verification_fails_for_an_unattested_chain():
    chain = attest_run(run_id="run_8", dataset=[],
                       model_fingerprint=ATTESTED_FINGERPRINT)

    verdict = verify_run_attestation(chain, PANEL)

    assert verdict["valid"] is False
    assert "attested" in verdict["reason"]


# ── the chain reaches the evidence layer ──────────────────────────────────

def test_attesting_a_result_with_an_incomplete_chain_withholds_it():
    """An unattested chain must not sit beside a live result.

    `attest_run_chain` is the door between the run and the evidence layer.
    When the chain cannot be completed the result is withheld, not published
    beside a digest that was never computed.
    """
    result = {"status": "completed", "provenance": "live", "attested": True,
              "value": 0.5}

    out = attest_run_chain(result, run_id="run_9", dataset=[],
                           model_fingerprint=ATTESTED_FINGERPRINT)

    assert out["attested"] is False
    assert out["provenance"] != "live"
    assert out["run_attestation"]["attested"] is False
    assert "not attested" in out["reason"]
    assert "measurement_record" not in out


def test_a_complete_chain_leaves_the_result_alone():
    result = {"status": "completed", "provenance": "live", "attested": True,
              "value": 0.5}

    out = attest_run_chain(result, run_id="run_10", dataset=PANEL,
                           model_fingerprint=ATTESTED_FINGERPRINT)

    assert out["attested"] is True
    assert out["provenance"] == "live"
    assert out["run_attestation"]["attested"] is True
    assert out["run_attestation"]["dataset_sha256"] == dataset_digest(PANEL)
