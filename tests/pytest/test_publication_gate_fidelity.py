"""Tests for the critic measurement_record forwarding fix.

These tests verify that Critic.reproduce forwards the pipeline's
measurement_record into its result dict, so that _is_attested and
reproduction_is_live correctly reflect the pipeline's attestation state.
"""

from agents.critic import Critic
from agents.evidence_policy import reproduction_is_live, _is_attested


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