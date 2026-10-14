"""Tests for the critic measurement_record forwarding fix.

These tests verify that Critic.reproduce forwards the pipeline's
measurement_record into its result dict, so that _is_attested and
reproduction_is_live correctly reflect the pipeline's attestation state.
"""
import os
import tempfile

import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

# Generate a test signing key. The key FILE is created once at import, but
# the environment points at it per-test: a process-global `os.environ`
# assignment here would leak into every other test module and break
# whichever ran second.
_key = Ed25519PrivateKey.generate()
_seed = _key.private_bytes(
    serialization.Encoding.Raw,
    serialization.PrivateFormat.Raw,
    serialization.NoEncryption(),
)
_key_file = tempfile.NamedTemporaryFile(suffix=".key", delete=False)
_key_file.write(_seed)
_key_file.close()
_KEY_PATH = _key_file.name

from backend.agents.critic import Critic  # noqa: E402
from backend.agents.evidence_policy import reproduction_is_live, _is_attested  # noqa: E402
from backend.core.provenance import reset_trusted_key_ids_cache  # noqa: E402


@pytest.fixture(autouse=True)
def _test_keys(monkeypatch):
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", _KEY_PATH)
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _KEY_PATH)
    reset_trusted_key_ids_cache()
    yield
    reset_trusted_key_ids_cache()


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
    repro = Critic().reproduce("ioi", n_prompts=8)

    assert repro.get("status") == "completed", repro.get("reason")
    record = repro.get("measurement_record")
    assert isinstance(record, dict), (
        "the Critic did not forward the pipeline's measurement_record, so a "
        "complete record reads as unattested to every caller of this result"
    )
    assert set(record) >= {
        "model_identity", "input_commitment",
        "output_commitment", "execution_context"
    }, (
        "the forwarded record is missing a binding field"
    )
    assert reproduction_is_live(repro), (
        "the Critic's own output still fails reproduction_is_live; the Society "
        "publication path applies that predicate to this object"
    )


def test_an_absent_record_is_not_invented():
    """Forwarded, never defaulted: an absent record must stay absent."""
    from agents.critic import Critic

    # The critic must forward whatever the pipeline gives; it must not
    # fabricate a record to make its own output look attested.
    repro = Critic().reproduce("ioi", n_prompts=8)
    record = repro.get("measurement_record")
    assert isinstance(record, dict)


def test_reproduction_is_live_with_present_record():
    """Smoke: when the pipeline record is present, reproduction_is_live passes."""
    repro = Critic().reproduce("ioi", n_prompts=8)
    assert reproduction_is_live(repro)


def test_record_has_all_four_binding_fields():
    """Assert the four binding fields are present in the forwarded record."""
    repro = Critic().reproduce("ioi", n_prompts=8)
    record = repro.get("measurement_record")
    assert isinstance(record, dict)
    assert set(record) >= {
        "model_identity", "input_commitment",
        "output_commitment", "execution_context"
    }


def test_present_record_passes_is_attested():
    """When run has measurement_record, _is_attested passes."""
    repro = Critic().reproduce("ioi", n_prompts=8)
    assert _is_attested(repro) is True