"""The provenance label must mean what it says, in both directions.

An audit of this repository found the honesty invariant inverted:

* real measurements reached the client as ``status: "ok"`` carrying numbers
  under ``provenance: "unavailable"`` -- ``gpt2_engine`` never attested them,
  and the API wrapper withholds anything unattested;
* values that were never measured claimed ``live`` -- a store read in
  ``aggregates``, an asserted envelope literal in ``pipeline_contract``, and
  arithmetic over arguments in the inspectors.

Both are the same rule broken, so both directions are pinned here.
"""
from __future__ import annotations

import inspect

from backend.core import aggregates
from backend.core.provenance import LIVE, UNAVAILABLE
from backend.science.reproducibility import pipeline_contract
from backend.services import gpt2_engine


# --------------------------------------------------------------------------- #
# Direction 1: a real measurement must be attested where it is made
# --------------------------------------------------------------------------- #

def test_engine_attests_a_measurement_it_made():
    out = gpt2_engine._measure_done({"status": "ok", "matrix": [[1.0]]})
    assert out["provenance"] == LIVE
    assert out["attested"] is True
    assert out["measurement_record"], (
        "an attestation without a measurement_record is a claim, not a record")


def test_engine_does_not_attest_a_refusal():
    """A refusal stays a refusal; the API boundary withholds it as before."""
    refusal = {"status": "unavailable", "error": "no weights"}
    out = gpt2_engine._measure_done(refusal)
    assert out == refusal
    assert "provenance" not in refusal


def test_every_measurement_entry_point_is_attested():
    """The wrapper withholds what the engine did not attest.

    Each of these returns numbers the API serves as a measurement, so each has
    to pass through the attesting decorator. ``infer`` attests itself and is
    not in the list.
    """
    entry_points = (
        "architecture", "run_prompt", "generate_text", "layer_detail",
        "list_neurons", "neuron_detail", "head_detail", "attention_head",
        "activations", "patch_head", "patch_neuron", "ioi",
        "layer_activations", "logit_lens_all", "ablate_layer", "steer",
        "logit_lens",
    )
    unattested = [name for name in entry_points
                  if not hasattr(getattr(gpt2_engine, name), "__wrapped__")]
    assert not unattested, (
        f"measurement entry points that cannot attest: {unattested}")


# --------------------------------------------------------------------------- #
# Direction 2: nothing unmeasured may claim `live`
# --------------------------------------------------------------------------- #

def test_read_models_never_claim_live():
    """`/api/analytics` and `/api/scientific_health` are reads, not measurements."""
    for payload in (aggregates.aggregate_analytics(),
                    aggregates.scientific_health()):
        assert payload["provenance"] != LIVE
        assert payload["provenance"] == "reference"
        assert LIVE not in set(payload["field_provenance"].values())


def test_pipeline_envelope_is_attested_not_asserted():
    envelope = pipeline_contract.completed(
        pipeline="arithmetic",
        paper_id="arithmetic_v1",
        model_id="gpt2",
        observed_metrics={"accuracy": 0.5},
        raw_traces=[{"correct": True}],
        provenance_note="measured on loaded weights",
        validation_eligible=False,
    )
    assert envelope["provenance"] == LIVE
    assert envelope["attested"] is True, (
        "a pipeline envelope that asserts `live` without the issuer is the "
        "label this repository exists to keep checkable")
    assert envelope["measurement_record"]
    assert envelope["model_loaded"] == "gpt2"


def test_pipeline_unavailable_envelope_claims_nothing():
    envelope = pipeline_contract.unavailable(
        pipeline="arithmetic",
        paper_id="arithmetic_v1",
        model_id=None,
        reason="no weights were reachable",
    )
    assert envelope["provenance"] == UNAVAILABLE
    assert "measurement_record" not in envelope
    assert envelope["publication_eligible"] is False


def test_pipeline_contract_issues_no_live_label_of_its_own():
    """The word `live` may appear only as documentation of the issuer."""
    source = inspect.getsource(pipeline_contract)
    assert 'LIVE = "live"' not in source
    assert '"provenance": LIVE' not in source
