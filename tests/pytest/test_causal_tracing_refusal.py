"""G: the unimplemented causal-tracing module must refuse completely.

`backend.interpretability.causal.causal_tracing.CausalTracingEngine` had
**zero** test references in a 1900-test suite. Its docstring is admirably
candid -- "NOT IMPLEMENTED", it synthesises a curve from `0.12 + 0.78` and
reports `max_causal_layer: 8` unconditionally -- and the response *is* marked
`status: "unavailable"`, `provenance: "reference"`,
`trace_measured: False`, `validation_eligible: False`.

So the refusal is honest. The defect is narrower and worth stating precisely:
two fabricated numbers are still returned at the top level with **no field
provenance**, while `layer_effects` -- which carries the same synthesised
values -- *is* marked.

    "layer_effects_field_provenance": {"0": "reference", ..., "11": "reference"}
    "max_causal_layer": 8,          <- unmarked
    "max_indirect_effect": 0.90,    <- unmarked

A consumer reading `max_causal_layer` gets a confident integer with nothing to
distinguish it from a measurement. That is the specific failure this project
exists to prevent: not a fabricated success, but a fabricated *number* beside
an honest disclaimer that the disclaimer does not cover.

The contract asserted here is the one the rest of the codebase already follows
-- `HFAdapterMixin._layer_error`, the tunedish refusal in
`discovery_planner`, `evidence_policy.field_map`: a value that did not come
from a measurement must be labelled `reference` at the field level, or withheld.

Nothing here demands the module be implemented. Demanding that would be the
wrong test -- the honest refusal is the correct behaviour today.
"""

from __future__ import annotations

import pytest

from backend.interpretability.causal.causal_tracing import CausalTracingEngine

CLEAN = "The capital of France is"
CORRUPTED = "The capital of Rome is"


@pytest.fixture
def trace():
    return CausalTracingEngine().trace_causal_effect(CLEAN, CORRUPTED)


# ── the refusal itself ─────────────────────────────────────────────────

def test_an_unimplemented_engine_refuses_rather_than_reporting_success(trace):
    """The headline verdict must be a refusal.

    Asserted first and separately from the provenance checks, because a module
    that reported `status: "completed"` would make every field-provenance
    assertion below pass trivially by having nothing to mark.
    """
    assert trace["status"] != "completed", (
        "an engine that measures nothing reported success"
    )
    assert trace["status"] == "unavailable"
    assert trace["trace_measured"] is False
    assert trace["validation_eligible"] is False
    assert trace["publication_eligible"] is False
    assert trace["reason"], "a refusal must say why"


def test_the_response_does_not_claim_to_have_run_the_supplied_prompts(trace):
    """Echoing the prompts back must not read as having run them.

    The docstring records this as a past defect: the prompts were returned in a
    payload that looked like results. They may still be echoed for
    inspection, but only in a field that is explicitly labelled, and never
    under a key a consumer would read as a measurement.
    """
    # The prompts are present for inspection -- that is allowed.
    assert trace["clean_prompt"] == CLEAN
    assert trace["corrupted_prompt"] == CORRUPTED

    # But nothing may claim the supplied prompts drove a measurement.
    assert trace["method"] == "SyntheticCurve", (
        f"the method reports {trace['method']!r}; it must name the closed-form "
        f"expression rather than a measurement"
    )


def test_the_synthesised_curve_is_not_an_ablation_sweep(trace):
    """The layer effects must match the documented closed form, not the model.

    Asserted against the expression in the docstring
    (`0.12 + 0.78` for layers 8-9, else `0.05 * (l % 3)`) precisely so that a
    future implementation replacing it with a real sweep *fails this test*.
    That is the intent: the test should have to be deleted deliberately when
    the module becomes real, not quietly outgrown.
    """
    for entry in trace["layer_effects"]:
        layer = entry["layer"]
        expected = round(0.12 + (0.78 if layer in (8, 9) else 0.05 * (layer % 3)), 3)
        assert entry["indirect_effect"] == expected, (
            f"layer {layer} reports {entry['indirect_effect']}, expected the "
            f"documented closed form {expected}. If this module has become a "
            f"real ablation sweep, this test should be replaced, not relaxed."
        )


# ── the actual defect: unmarked fabricated numbers ─────────────────────

def test_every_unmeasured_number_is_marked_reference(trace):
    """No synthesised value may reach the caller unmarked.

    The defect: `layer_effects` carries a `layer_effects_field_provenance` map
    marking each layer `"reference"`, but `max_causal_layer` and
    `max_indirect_effect` are returned bare. A consumer reading either one gets
    a confident number with no indication that nothing was measured.

    Every numeric field the response returns must therefore be covered by a
    provenance map, or absent.
    """
    numeric = {
        key: value
        for key, value in trace.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    assert numeric, (
        "fixture assumption broken: the response returned no numbers, so this "
        "test cannot check their provenance"
    )

    marked = set()
    for key, value in trace.items():
        if key.endswith("_field_provenance") and isinstance(value, dict):
            marked |= {str(k) for k in value}

    unmarked = sorted(set(numeric) - marked)
    assert not unmarked, (
        f"these numbers are returned with no field provenance: {unmarked}. "
        f"They are synthesised -- the docstring's closed form, not a "
        f"measurement -- so a consumer reading them has nothing to tell them "
        f"apart from a real result. Mark them 'reference' or withhold them. "
        f"(Currently marked: {sorted(marked)})"
    )


def test_the_headline_numbers_are_not_asserted_as_facts(trace):
    """A synthesised headline must not read as a finding.

    `max_causal_layer: 8` is reported unconditionally -- it is not the argmax
    of the returned curve by construction, it is a constant. Either it is
    marked as reference, or it is withheld. Asserted separately from the
    general provenance sweep so a regression names the field directly.
    """
    marked = set()
    for key, value in trace.items():
        if key.endswith("_field_provenance") and isinstance(value, dict):
            marked |= {str(k) for k in value}

    for field in ("max_causal_layer", "max_indirect_effect"):
        if field not in trace:
            continue     # withheld is an acceptable outcome
        assert field in marked, (
            f"{field} = {trace[field]!r} is returned with no field provenance. "
            f"It is a constant from a closed-form expression, not a finding, "
            f"and an unmarked number at the top level reads as a measurement."
        )


def test_the_declared_maximum_agrees_with_the_returned_curve_or_is_marked(trace):
    """If the headline is retained, it must at least match what it summarises.

    `max_causal_layer` is documented as reported unconditionally. It happens to
    coincide with the argmax of the synthesised curve, so this cannot catch a
    real inconsistency -- which is why it is the weakest of these checks and
    stated as such. It is here to catch a future edit that changes the curve
    and forgets the constant.
    """
    layer_effects = trace.get("layer_effects") or []
    assert layer_effects, "no layer effects returned"

    derived = max(layer_effects, key=lambda e: e["indirect_effect"])
    assert trace["max_causal_layer"] == derived["layer"], (
        f"max_causal_layer is {trace['max_causal_layer']} but the curve peaks "
        f"at layer {derived['layer']}"
    )
    assert trace["max_indirect_effect"] == pytest.approx(
        derived["indirect_effect"], abs=1e-6
    ), (
        f"max_indirect_effect is {trace['max_indirect_effect']} but the "
        f"curve peaks at {derived['indirect_effect']}"
    )