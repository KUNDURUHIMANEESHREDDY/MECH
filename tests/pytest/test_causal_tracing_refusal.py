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

import hashlib
import inspect
import json as _json

import pytest

from backend.interpretability.causal.causal_tracing import CausalTracingEngine

CLEAN = "The capital of France is"
CORRUPTED = "The capital of Rome is"

#: The four values the stub's closed form can return. A closed form picks one
#: of a handful of constants; an ablation sweep measures twelve layers and
#: returns twelve arbitrary numbers. That difference is what makes "this is
#: a synthesised curve" a checkable statement rather than a label.
SYNTHETIC_GRID = (0.12, 0.17, 0.22, 0.90)


def _documented_curve(clean: str, corrupted: str):
    """The closed form documented in `causal_tracing.py`, re-derived here.

    Re-implemented rather than imported on purpose. This file exists to pin
    the *shape* of the synthetic curve, so if the module's curve changes the
    test has to fail until somebody decides the change was intended.
    Importing the module's own helper would make the test agree with
    whatever the module does, which is the opposite of a pin.
    """
    pair = _json.dumps({"clean": clean, "corrupt": corrupted}, sort_keys=True)
    digest = int(hashlib.sha256(pair.encode()).hexdigest(), 16)
    curve = []
    for layer in range(12):
        byte = (digest >> (layer * 8)) & 0xFF
        if layer in (8, 9):
            value = round(0.12 + 0.78, 3)
        else:
            value = round(0.12 + 0.05 * (byte % 3), 3)
        curve.append((layer, value))
    return curve


def _engine_is_unimplemented() -> bool:
    """Whether causal tracing is still the deliberately-unimplemented stub.

    Read from the module's own docstring, the same way
    `test_experiment_not_executed.py` reads its capability state. When someone
    replaces this module with a real ablation sweep the flag flips and the
    tests below stop demanding a synthetic curve.
    """
    return "NOT IMPLEMENTED" in (inspect.getdoc(
        __import__("backend.interpretability.causal.causal_tracing",
                   fromlist=["x"])) or "")


#: Applied to every assertion that only makes sense while the engine refuses.
#:
#: Without this, implementing the module correctly turns the suite red. That
#: is the failure this guards: a test named "the synthesised curve is not an
#: ablation sweep" would fail the day a real ablation sweep lands, and the
#: person who did the work would be pushed to delete the test rather than to
#: trust it -- which is precisely backwards. An implementation-specific
#: synthetic curve must never become a permanent scientific invariant.
requires_unimplemented = pytest.mark.skipif(
    not _engine_is_unimplemented(),
    reason="causal tracing is implemented; the refusal and synthetic-curve "
           "assertions no longer apply. `implemented_causal_effect_is_a_real_"
           "ablation_sweep` takes over.",
)


@pytest.fixture
def trace():
    return CausalTracingEngine().trace_causal_effect(CLEAN, CORRUPTED)


# â”€â”€ the refusal itself â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@requires_unimplemented
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


@requires_unimplemented
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


@requires_unimplemented
def test_the_synthesised_curve_is_not_an_ablation_sweep(trace):
    """The layer effects must match the documented closed form, not the model.

    Asserted against the expression in the module's own docstring so that a
    future implementation replacing it with a real sweep *fails this test*.
    That is the intent: the test should have to be deleted deliberately when
    the module becomes real, not quietly outgrown.

    It failed once for a reason that had nothing to do with sweeps. The pin
    said `0.05 * (layer % 3)`; the module had moved to a prompt-hash-derived
    `0.05 * (byte % 3)`. Both are synthetic -- neither consults a model --
    but they are not the same curve, so the test reported a defect that did
    not exist while saying nothing about the defects that did. A stale pin is
    not a lenient test, it is a broken one.

    So the pin is now the expression the module actually documents, and the
    property that matters -- *this is a closed form, not a measurement* -- is
    asserted separately and structurally, as the four-value grid.
    """
    expected = _documented_curve(CLEAN, CORRUPTED)
    effects = trace["layer_effects"]
    assert len(effects) == len(expected), (
        f"the curve has {len(effects)} layers; the documented closed form has "
        f"{len(expected)}"
    )

    for entry, (layer, value) in zip(effects, expected):
        assert entry["layer"] == layer, (
            f"the curve reports layer {entry['layer']} where the closed form "
            f"has {layer}"
        )
        assert entry["indirect_effect"] == value, (
            f"layer {layer} reports {entry['indirect_effect']}, expected the "
            f"documented closed form {value}. If this module has become a "
            f"real ablation sweep, this test should be replaced, not relaxed."
        )

    off_grid = [e["indirect_effect"] for e in effects
                if e["indirect_effect"] not in SYNTHETIC_GRID]
    assert not off_grid, (
        f"layers report values outside the synthetic grid {SYNTHETIC_GRID}: "
        f"{off_grid}. A closed form returns one of four constants; a "
        f"measurement returns whatever the model happened to produce."
    )


@requires_unimplemented
def test_the_synthesised_curve_is_a_function_of_the_prompt_pair(trace):
    """The curve varies with the prompts, and with nothing else.

    Two halves, both of which a measurement could not satisfy:

    * the same prompt pair always gives the same curve -- no hidden state,
      no model, no clock between calls;
    * a different prompt pair gives a different curve -- the module's
      documented reason for deriving the values from the pair at all.

    Stated as a property rather than pinned to a value, so it survives a
    change of formula: what must not change is *what the curve depends on*.
    """
    again = CausalTracingEngine().trace_causal_effect(CLEAN, CORRUPTED)
    assert [e["indirect_effect"] for e in again["layer_effects"]] == [
        e["indirect_effect"] for e in trace["layer_effects"]
    ], (
        "the same prompt pair produced two different curves; the synthetic "
        "curve is supposed to be a pure function of the pair"
    )

    other = CausalTracingEngine().trace_causal_effect(
        "The capital of Spain is", "The capital of Rome is")
    assert [e["indirect_effect"] for e in other["layer_effects"]] != [
        e["indirect_effect"] for e in trace["layer_effects"]
    ], (
        "two different prompt pairs produced the same curve; the values are "
        "supposed to be derived from the pair, which is what stops the "
        "response from being a constant dressed up as a result"
    )


# â”€â”€ the actual defect: unmarked fabricated numbers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@requires_unimplemented
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


@requires_unimplemented
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


@requires_unimplemented
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

# -- the implemented state ----------------------------------------------
#
# The refusal tests above are skipped once the engine is real. That must not
# leave the file asserting nothing, or the moment someone implements causal
# tracing this file quietly stops contributing. The tests below take over.


@pytest.mark.skipif(
    _engine_is_unimplemented(),
    reason="causal tracing is still the unimplemented stub; the assertions "
           "that describe a real ablation sweep do not apply yet",
)
def test_implemented_causal_effect_is_a_real_ablation_sweep(trace):
    """Once implemented, the response must describe a measurement.

    Deliberately written as the inverse of the refusal contract rather than
    as a check for one particular implementation: whatever the method does
    internally, the numbers it returns must be marked as measured rather than
    as reference, and must not carry the closed-form shape the stub used.
    """
    assert trace.get("trace_measured") is True, (
        f"the engine is implemented but reports trace_measured="
        f"{trace.get('trace_measured')!r}"
    )

    effects = trace.get("layer_effects") or []
    assert effects, "a real sweep returned no layers"

    measured = {
        str(k): v for k, v in trace.get("layer_effects_field_provenance", {}).items()
    }
    unmarked = [e for e in effects
                if measured.get(str(e.get("layer"))) == "reference"]
    assert not unmarked, (
        f"layers {len(unmarked)} of {len(effects)} are still marked "
        f"'reference' after implementation; an unimplemented marker left in "
        f"place would make real measurements look synthetic"
    )

    # The stub's closed form returns one of four constants; a real sweep is
    # not a constant and does not sit on that grid.
    stub_shape = sum(
        1 for e in effects
        if round(float(e["indirect_effect"]), 3) in SYNTHETIC_GRID
    )
    assert stub_shape < len(effects), (
        f"{stub_shape} of {len(effects)} layers still match the stub's "
        f"synthetic grid; this looks like the synthetic curve with a "
        f"new label on it"
    )


@pytest.mark.skipif(
    _engine_is_unimplemented(),
    reason="causal tracing is still the unimplemented stub",
)
def test_an_implemented_engine_still_labels_provenance_per_value(trace):
    """The provenance discipline outlives the stub.

    This was the defect this file found: `max_causal_layer` and
    `max_indirect_effect` came back bare while `layer_effects` was marked
    "reference". An implementation must not reintroduce it by assuming that
    measured values no longer need saying so.
    """
    fp = trace.get("field_provenance") or {}
    numeric = {
        k for k, v in trace.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    covered = set(fp) | {
        k for k in trace if k.endswith("_field_provenance")
    }
    assert not (numeric - covered), (
        f"numeric fields returned without provenance: {sorted(numeric - covered)}"
    )
