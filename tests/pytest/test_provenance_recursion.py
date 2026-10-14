"""The recursive provenance validator itself, and the gap it closes.

`_provenance.py` is shared infrastructure: one recursive rule rather than
three hand-rolled top-level checks. This file tests the validator, and then
applies it to the three responses the review names.

The concrete defect, measured on `causal_tracing`:

    {"layer": 3, "indirect_effect": 0.12, "top_causal_component": "Attention"}

The response marks the list by index --
`"layer_effects_field_provenance": {"0": "reference", ...}` -- which says
"row 3 came from the reference curve" and nothing about the fields inside
row 3. An extracted `entry["indirect_effect"]` is a bare float.

The validator recognises the per-entry form, `field_provenance` carried
*inside* the entry itself, because that is the only marking that survives
extraction. Index-based marking does not.
"""

from __future__ import annotations

import pytest

from _provenance import (
    assert_all_numbers_marked,
    collect_unmarked_numbers,
    is_derived_from_unmeasured,
    unmarked_leaves,
)


# ── the validator's own rules ─────────────────────────────────────────

def test_an_unmarked_number_is_found_at_the_top_level():
    assert collect_unmarked_numbers({"metric": 1.5}) == [("metric", 1.5)]


def test_a_top_level_field_provenance_map_marks_its_siblings():
    payload = {
        "status": "unavailable",
        "field_provenance": {"metric": "reference"},
        "metric": 1.5,
    }
    assert collect_unmarked_numbers(payload) == []


def test_a_per_key_provenance_sibling_marks_its_key():
    payload = {"metric": 1.5, "metric_field_provenance": "reference"}
    assert collect_unmarked_numbers(payload) == []


def test_a_number_nested_in_a_dict_is_found():
    assert collect_unmarked_numbers(
        {"outer": {"inner": 2.0}}) == [("outer.inner", 2.0)]


def test_a_number_nested_in_a_list_is_found_by_index():
    assert collect_unmarked_numbers(
        {"rows": [{"v": 1.0}]}) == [("rows[0].v", 1.0)]


def test_structural_keys_are_not_required_to_carry_provenance():
    """`layer` and `head` identify a measurement; they are not measurements."""
    payload = {"layer": 3, "head": 7, "status": "ok"}
    assert collect_unmarked_numbers(payload) == []


def test_per_entry_field_provenance_marks_the_fields_inside_that_entry():
    """The form that survives extraction, and the one that closes the gap."""
    payload = {
        "layer_effects": [
            {"layer": 0, "indirect_effect": 0.12,
             "field_provenance": {"indirect_effect": "reference"}},
            {"layer": 1, "indirect_effect": 0.24,
             "field_provenance": {"indirect_effect": "reference"}},
        ],
    }
    assert collect_unmarked_numbers(payload) == []


def test_index_based_marking_does_not_mark_the_fields_inside():
    """The actual defect.

    Marking the *list* by index says which row is synthetic. Extracting
    `entry["indirect_effect"]` yields a bare float, so the marking is lost at
    exactly the point a caller would use it.
    """
    payload = {
        "layer_effects": [{"layer": 3, "indirect_effect": 0.12}],
        "layer_effects_field_provenance": {"0": "reference"},
    }
    assert collect_unmarked_numbers(payload) == [
        ("layer_effects[0].indirect_effect", 0.12),
    ]


def test_the_error_message_names_every_path():
    payload = {"a": 1.0, "b": {"c": 2.0}}
    with pytest.raises(AssertionError) as excinfo:
        assert_all_numbers_marked(payload)
    message = str(excinfo.value)
    assert "a" in message and "b.c" in message, message


def test_deep_nesting_cannot_escape_the_check():
    payload = {"a": [{"b": [{"c": [{"d": 9.0}]}]}]}
    assert unmarked_leaves(payload) == ["a[0].b[0].c[0].d"]


def test_require_all_false_relaxes_the_rule_without_raising():
    payload = {"metric": 1.5}
    assert collect_unmarked_numbers(payload)
    assert collect_unmarked_numbers(payload, require_all=False) == []


# ── derived values must not inherit liveness ─────────────────────────

def test_a_value_marked_non_live_is_reported_as_derived_from_unmeasured():
    payload = {
        "status": "completed",
        "derived_score": 0.9,
        "field_provenance": {"derived_score": "reference"},
    }
    assert is_derived_from_unmeasured(payload) == ["derived_score"]


def test_a_live_value_is_not_reported():
    payload = {
        "status": "completed",
        "score": 0.9,
        "field_provenance": {"score": "live"},
    }
    assert is_derived_from_unmeasured(payload) == []


def test_the_outer_status_does_not_excuse_an_unmeasured_leaf():
    """The review's rule, stated as a test.

    `status: "completed"` at the top is not a licence for a synthesised
    score beneath it. Trust flows downward only.
    """
    payload = {
        "status": "completed",
        "provenance": "live",
        "summary": {"score": 0.97},
        "field_provenance": {"summary": "live"},
    }
    # The outer map says `summary` is live, but nothing marks the score inside
    # it -- and it is the number a reader would quote.
    assert collect_unmarked_numbers(payload) == [("summary.score", 0.97)]


# ── applied to the three responses under review ──────────────────────

def test_the_causal_tracing_response_marks_every_number_it_returns():
    """The live gap, asserted against the real response.

    Expected to fail until the engine marks its nested values. Kept as a
    failing test deliberately: it is the specification for a change nobody has
    made yet, and deleting it would lose the only statement of what is
    required.
    """
    from backend.interpretability.causal.causal_tracing import CausalTracingEngine

    trace = CausalTracingEngine().trace_causal_effect(
        "The capital of France is", "The capital of Rome is")

    assert_all_numbers_marked(
        trace, what="causal-tracing response")


def test_a_blocked_discoverer_response_leaks_no_unmarked_number(monkeypatch):
    """The blocked path, which is the one the review's rule is about.

    An engine result that fails attestation must not let any number through as
    though it were measured. This asserts that end to end on the real
    response: no numeric leaf survives without provenance, and the outer
    status is not `completed`.

    An earlier version of this test aimed at the *live* branch with
    placeholder commitments. It failed with `unavailable` and the failure was
    the guard working correctly -- `_is_attested` verifies the measurement
    record cryptographically, not structurally. Rather than forge a record,
    this asserts on the branch that is reachable, which is also the branch that
    matters: a refused measurement must not leak numbers.
    """
    import sys
    import types

    engine_module = "backend.interpretability.discovery.discovery_engine"
    unattested = {
        "status": "completed", "provenance": "live", "attested": True,
        "measurement_record": {
            "model_identity": {"model_name": "gpt2"},
            "input_commitment": "sha256:" + "0" * 64,
            "output_commitment": "sha256:" + "1" * 64,
            "execution_context": {"device": "cpu", "dtype": "torch.float32"},
        },
        "validation_eligible": True, "publication_eligible": True,
        "discovery_id": "d", "result": {"circuits": 1},
        "metrics": {"accuracy": 0.97, "loss": 0.11},
        "per_prompt": [{"score": 0.9}],
    }
    module = types.ModuleType(engine_module)

    class _Engine:
        def __init__(self, *a, **kw):
            pass

        def discover_and_orchestrate(self, **kw):
            return unattested

    module.DiscoveryEngine = _Engine
    monkeypatch.setitem(sys.modules, engine_module, module)

    from backend.agents.discoverer import Discoverer

    out = Discoverer().discover("a hypothesis")

    assert out["status"] != "completed", (
        "an unattested result reported completion; the gate did not fire"
    )
    assert_all_numbers_marked(out, what="blocked discoverer response")
    assert is_derived_from_unmeasured(out) == [], (
        "a value is marked non-live yet still presented as a measurement: "
        f"{is_derived_from_unmeasured(out)}"
    )