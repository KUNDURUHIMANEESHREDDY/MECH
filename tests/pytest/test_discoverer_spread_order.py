"""A wrapper's guarded verdict must not be overwritable by the payload it judged.

The rule under test
------------------
`Discoverer.discover()` calls the engine, decides with `discovery_is_live(res)`,
and only then builds its response. The guarded keys -- `status`, `provenance`,
`field_provenance` -- are the *verdict*.

The spread used to come after them:

    return {
        "status": "completed",
        "provenance": "live",
        "field_provenance": field_map((...), "live"),
        **res,                      # <- overwrites all three
    }

So `res` decided the verdict rather than the guard. That is the same shape as
the P0 defect: a wrapper's provenance answer that was not authoritative.

It was not an active misreport. `discovery_is_live` requires
`provenance_of(payload) == "live"` and `_is_completed(payload)`, so the two keys
the spread could overwrite had to agree with the guard already -- otherwise the
guard would not have been reached. `field_provenance` is the exception:
`discovery_is_live` does not inspect it, so it could be replaced wholesale by a
claim the guard never checked.

Which is what these tests pin. The fake engine is injected into `sys.modules`
because `discover()` imports the engine inside the function body, which keeps the
test off the ~20-module import chain and off torch.
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict

import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "backend" / "agents" / "discoverer.py"
ENGINE_MODULE = "backend.interpretability.discovery.discovery_engine"

#: The keys the guard decides. A payload spread after these can overwrite them.
GUARDED = ("status", "provenance", "field_provenance")

# Attestation needs a signed, trusted record: generate one key for the
# session. The key FILE is created once at import, but the environment
# points at it per-test: a process-global `os.environ` assignment here
# would leak into every other test module and break whichever ran second.
# Without trust configured the fixture below cannot clear `_is_attested`
# and every positive test reads as a guard defect.
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

from backend.core.provenance import (  # noqa: E402
    attest_measurement,
    reset_trusted_key_ids_cache,
)


@pytest.fixture(autouse=True)
def _test_keys(monkeypatch):
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", _KEY_PATH)
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _KEY_PATH)
    reset_trusted_key_ids_cache()
    yield
    reset_trusted_key_ids_cache()


def _fake_engine(monkeypatch, payload: Dict[str, Any]) -> None:
    """Make `discover()` see `payload` as the engine's result.

    The signature mirrors the real `DiscoveryEngine.discover_and_orchestrate`,
    which takes `n_prompts` and `n_interaction_prompts` alongside the
    hypothesis. An earlier version of this fake declared only
    `hypothesis_statement`, so when the Society's discovery began forwarding
    sample sizes the call raised `TypeError: unexpected keyword argument
    'n_prompts'`, the `except` arm in `discover()` returned
    `status: "error"`, and all nine blocked-payload tests failed on the wrong
    status -- reporting a wrapper defect that did not exist.

    Worth stating plainly: a stale test double made the guard look broken.
    The double exists only to control the engine's return value, so it must
    accept whatever the caller legitimately passes. `**kwargs` rather than two
    named parameters, so the next caller-supplied option does not repeat this.
    """
    module = types.ModuleType(ENGINE_MODULE)

    class _Engine:
        def __init__(self) -> None:
            pass

        def discover_and_orchestrate(
            self, hypothesis_statement: str, **kwargs: Any
        ) -> Dict[str, Any]:
            return payload

    module.DiscoveryEngine = _Engine
    monkeypatch.setitem(sys.modules, ENGINE_MODULE, module)


def _live_payload(**overrides: Any) -> Dict[str, Any]:
    """A payload that clears `discovery_is_live`.

    `discovery_is_live` needs a completed status, `provenance == "live"`, a
    valid *measurement record*, and `_opted_in` -- both eligibility flags true.
    Anything else short-circuits to the blocked branch and never reaches the
    spread.

    The measurement record matters here, and it must be genuine: `_is_attested`
    verifies the record's trusted-key signature, its weight-derived model
    identity, and its recomputed input/output commitments. A hand-written
    record with plausible fields satisfies none of that, so the fixture is
    built with `attest_measurement` itself (signed by the session test key,
    which is also trusted). The content hash stands in for a weight-derived
    fingerprint.

    What this file tests is the *spread ordering* -- whether a payload can
    overwrite the guard's verdict -- which does not depend on the record's
    contents beyond their validity. Whether commitments are genuine is
    `test_measurement_record.py`'s and `test_measurement_authenticity.py`'s
    subject, not this file's.
    """
    base: Dict[str, Any] = {
        "status": "completed",
        "validation_eligible": True,
        "publication_eligible": True,
        "discovery_id": "disc_test",
        "result": {"circuits": 1},
    }
    attested = attest_measurement(
        dict(base),
        model_loaded="gpt2",
        model_content_hash="sha256:test-weight-fingerprint",
        signing_key=_KEY_PATH,
    )
    payload: Dict[str, Any] = {
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "measurement_record": attested["measurement_record"],
        "validation_eligible": True,
        "publication_eligible": True,
        "discovery_id": "disc_test",
        "result": {"circuits": 1},
    }
    payload.update(overrides)
    return payload


def _discover() -> Dict[str, Any]:
    from backend.agents.discoverer import Discoverer

    return Discoverer().discover("a hypothesis about IOI")


# ── Behaviour ──────────────────────────────────────────────────────────

def test_a_payload_cannot_overwrite_the_guards_field_provenance(monkeypatch):
    """The regression, stated as an outcome.

    `field_provenance` is the key the spread could genuinely clobber: the guard
    does not inspect it, so the engine's own value used to win outright.
    """
    _fake_engine(monkeypatch, _live_payload(**{
        "field_provenance": {"status": "synthetic", "result": "synthetic"},
    }))

    out = _discover()

    assert out["provenance"] == "live"
    assert out["status"] == "completed"
    for field in GUARDED:
        assert field in out, f"the guard dropped {field}"
    assert out["field_provenance"].get("result") == "live", (
        "the engine's field_provenance overrode the guard's verdict: "
        f"{out['field_provenance']}")
    assert out["field_provenance"].get("status") == "live", (
        f"the engine's field_provenance overrode the guard's verdict: "
        f"{out['field_provenance']}")


def test_the_fake_actually_accepts_what_the_caller_passes(monkeypatch):
    """The engine interface this file substitutes must match the real one.

    Added because of the failure this file's own fake caused. The double
    declared only `hypothesis_statement`; the caller began forwarding
    `n_prompts` and `n_interaction_prompts`, so every test that expected a
    blocked verdict got `status: "error"` from the `except` arm instead. The
    guard was never reached, and the failure read as a provenance defect.

    A stale double is worse than no double, because it reports a defect in the
    code under test rather than in the test. Binding the call explicitly --
    here and via `**kwargs` -- turns that class of mistake into a failure at
    the double's own boundary instead of a wrong verdict nine tests later.

    Read from the real engine's signature, so the two cannot drift apart
    without this failing.
    """
    import inspect

    from backend.interpretability.discovery.discovery_engine import (
        DiscoveryEngine,
    )

    real_params = set(
        inspect.signature(DiscoveryEngine.discover_and_orchestrate).parameters
    ) - {"self"}

    module = types.ModuleType(ENGINE_MODULE)
    captured: Dict[str, Any] = {}

    class _Engine:
        def __init__(self) -> None:
            pass

        def discover_and_orchestrate(
            self, hypothesis_statement: str, **kwargs: Any
        ) -> Dict[str, Any]:
            captured.update(kwargs)
            return _live_payload()

    module.DiscoveryEngine = _Engine
    monkeypatch.setitem(sys.modules, ENGINE_MODULE, module)

    # Discoverer must not raise, and must not fall into its error arm.
    out = _discover()
    assert out["status"] == "completed", (
        f"discover() returned {out['status']!r}; the substituted engine was "
        f"probably called with a signature it does not accept -- error: "
        f"{out.get('error')}"
    )

    # And every required-ish engine parameter the caller could forward must be
    # something the double tolerates.
    assert real_params, "could not read the real engine's signature"
    for name in ("n_prompts", "n_interaction_prompts"):
        assert name in real_params, (
            f"the real engine no longer takes {name}; update the double and "
            f"whatever forwards it"
        )

    # The Society's discovery is supposed to forward both, and the double
    # accepts **kwargs, so confirm they actually arrived.
    assert "n_prompts" in captured, (
        f"discover() forwarded none of the sample-size parameters; the real "
        f"engine accepts {sorted(real_params)}"
    )
    assert captured["n_prompts"] >= 10, (
        "the Society asks for a sample adequate to support a publishable "
        f"finding; got {captured['n_prompts']}"
    )


def test_the_payload_is_still_spread_so_nothing_is_lost(monkeypatch):
    """The fix reorders the spread; it does not drop it.

    `discovery_id` and `result` come from the engine and the guard names both in
    its field map, so losing them would break every downstream consumer.
    """
    _fake_engine(monkeypatch, _live_payload())

    out = _discover()

    assert out["discovery_id"] == "disc_test"
    assert out["result"] == {"circuits": 1}


@pytest.mark.parametrize("payload, why", [
    ({"provenance": "unavailable"}, "the engine withheld the label"),
    ({"provenance": "mock"}, "the engine returned mock data"),
    ({"provenance": "synthetic"}, "the engine returned synthetic data"),
    ({"attested": False}, "the engine measured but did not attest"),
    ({"attested": None}, "the engine said nothing about attestation"),
    ({"validation_eligible": False}, "the engine declined validation"),
    ({"publication_eligible": False}, "the engine declined publication"),
    ({"status": "unavailable"}, "the engine did not complete"),
    ({"validation_eligible": None}, "the engine stated nothing about validation"),
    # A bare `attested: True` is no longer attestation: the record must be
    # present and carry its four binding fields. Added when that tightened,
    # because the failure mode is a payload that *looks* attested -- the flag
    # is True, the status is completed, the provenance is live -- and would
    # otherwise be promoted into validation on nothing but a boolean.
    ({"measurement_record": None},
     "the engine claimed attestation but supplied no record"),
    ({"measurement_record": {"attested": True}},
     "the record omits every binding field"),
    ({"measurement_record": {
        "attested": True,
        "model_identity": {"model_name": "gpt2"},
        "input_commitment": "sha256:" + "0" * 64,
        # output_commitment and execution_context missing
    }},
     "the record is missing its output commitment"),
])
def test_nothing_but_a_live_opted_in_payload_reaches_the_verdict(monkeypatch,
                                                                   payload, why):
    """Every condition `discovery_is_live` enforces, one at a time.

    If any of these reached the `live` verdict the guard would be decorative.

    Note what is *not* asserted: that `provenance` differs. The blocked branch
    reports `provenance_of(res)` faithfully, so a payload that ran live and then
    declined to opt in yields `status: unavailable` with `provenance: live`.
    That is not a contradiction -- the run was real, the promotion was refused --
    and it is the same reading applied in `critic._reproduce` and in the
    Society's `discover` step. What has to hold is that the result cannot be
    *used*, which is what `discovery_is_live` on the output decides.
    """
    from backend.agents.evidence_policy import discovery_is_live, provenance_of

    engine_says = provenance_of(_live_payload(**payload))
    _fake_engine(monkeypatch, _live_payload(**payload))

    out = _discover()

    assert discovery_is_live(out) is False, (
        f"{why}, yet the blocked result is still eligible for validation")
    assert out["status"] == "unavailable"
    assert out["validation_eligible"] is False
    assert out["publication_eligible"] is False
    assert out["reason"], "a blocked result must say why"
    assert out["provenance"] == engine_says, (
        "the wrapper must report the provenance the engine actually gave, "
        f"not invent one: said {engine_says!r}, reported {out['provenance']!r}")


def test_a_blocked_payload_is_not_spread_under_result(monkeypatch):
    """A synthetic result must not sit where the evidence chain looks for it.

    `result` is consumed by validation, the evidence graph and the scribe. A
    synthetic payload placed there would read as scientific evidence.
    """
    _fake_engine(monkeypatch, _live_payload(
        provenance="synthetic", result={"fabricated": True}))

    out = _discover()

    assert "result" not in out, (
        "a synthetic payload was exposed under `result`, which validation and "
        "the evidence graph treat as scientific evidence")


# ── Runtime contract: the response schema ──────────────────────────────
#
# Everything above tests the *verdict keys* -- `status`, `provenance`,
# `field_provenance`. Those are what the spread ordering governs.
#
# What it does not govern is everything the spread *adds*. `**res` copies the
# engine's payload verbatim, and `field_map` names three keys. So a live
# response carries every field the engine chose to return with no statement of
# what they are.
#
# Measured on the canonical live payload, with four extra keys a real engine
# would plausibly emit:
#
#     field_provenance covers: ['discovery_id', 'result', 'status']
#     returned keys (13): attested, discovery_id, field_provenance,
#       measurement_record, metrics, notes, per_prompt, provenance,
#       publication_eligible, raw_traces, result, status, validation_eligible
#     UNMARKED (10): attested, field_provenance, measurement_record, metrics,
#       notes, per_prompt, provenance, publication_eligible, raw_traces,
#       validation_eligible
#
# `raw_traces` and `per_prompt` are arrays of measurement-shaped values
# reaching the caller at `provenance: "live"` with nothing marking them. That
# is gap 2 of the contract review, and it is the same class the P0 defect was:
# a field whose standing the response does not state.


def _live_payload_with_extras() -> Dict[str, Any]:
    """A live payload carrying nested objects, arrays and loose scalars."""
    return _live_payload(**{
        "metrics": {"accuracy": 0.97, "loss": 0.11},
        "per_prompt": [{"idx": 0, "score": 0.9}, {"idx": 1, "score": 0.8}],
        "raw_traces": [{"layer": 0, "value": 1.23}],
        "notes": "an unmarked string",
    })


def test_every_returned_field_is_covered_by_the_field_provenance_map(monkeypatch):
    """The live response must not carry fields of unknown standing.

    This is the runtime counterpart to the AST check below. The AST check
    asserts the spread is *ordered* correctly; it says nothing about which
    fields the spread delivers, so a wrapper could pass it while handing back
    an arbitrary number of unmarked values. The trust decision has to be
    enforced on the response a caller actually receives, not only on the
    source shape that produced it.
    """
    _fake_engine(monkeypatch, _live_payload_with_extras())

    out = _discover()
    assert out["status"] == "completed", f"fixture assumption: {out['status']}"

    covered = set(out["field_provenance"])
    returned = set(out)
    # `field_provenance` cannot appear in its own map: it is built *from* the
    # response's keys and is therefore not one of them yet. That is inherent
    # to a self-describing map, not a defect, so it is excluded explicitly
    # rather than by loosening the assertion.
    uncovered = sorted(returned - covered - {"field_provenance"})

    assert not uncovered, (
        f"these fields are returned at provenance={out['provenance']!r} with no "
        f"entry in `field_provenance`: {uncovered}. A caller reading any of "
        f"them has nothing to tell a measurement from a value the engine "
        f"invented. Either mark them or withhold them."
    )


def test_field_provenance_does_not_claim_fields_the_response_does_not_carry(
    monkeypatch,
):
    """The map must describe this response, not a template.

    All three return paths name fields they do not always return -- the
    blocked path lists `result`, which it deliberately withholds; so does the
    error path. A provenance map is a lookup keyed by field name, so an entry
    with no corresponding field is a claim about something that is not there,
    and it trains a consumer to trust the map as a description of the payload.
    """
    _fake_engine(monkeypatch, _live_payload(provenance="synthetic"))

    out = _discover()
    assert out["status"] == "unavailable", f"fixture assumption: {out}"

    phantom = sorted(set(out["field_provenance"]) - set(out))
    assert not phantom, (
        f"`field_provenance` names fields this response does not contain: "
        f"{phantom}. present: {sorted(out)}"
    )


def test_a_live_response_marks_nested_objects_and_arrays_not_only_scalars(
    monkeypatch,
):
    """Nested and array fields are the ones most likely to be read as data.

    A scalar like `status` is self-describing. `raw_traces` is not: it is a
    list of dicts shaped exactly like a measurement result, and it is the field
    a downstream consumer is most likely to quote. If only top-level scalars
    are marked, that field arrives looking like evidence.
    """
    _fake_engine(monkeypatch, _live_payload_with_extras())

    out = _discover()
    nested = {
        "metrics": "a nested object",
        "per_prompt": "an array of objects",
        "raw_traces": "an array of measurement-shaped dicts",
        "measurement_record": "the nested attestation record",
    }
    for field, description in nested.items():
        assert out.get(field) is not None, (
            f"fixture assumption broken: {field} ({description}) did not "
            f"survive the response"
        )
        assert field in out["field_provenance"], (
            f"{field} ({description}) reached the caller at provenance="
            f"{out['provenance']!r} with no provenance entry. It is the kind "
            f"of field a consumer quotes as a result."
        )


# ── Malformed, missing and contradictory engine output ─────────────────


def _broken_engine(monkeypatch, value: Any) -> None:
    """Make the engine return `value` verbatim, however malformed."""
    module = types.ModuleType(ENGINE_MODULE)

    class _Engine:
        def __init__(self, *a, **kw):
            pass

        def discover_and_orchestrate(self, **kw):
            return value

    module.DiscoveryEngine = _Engine
    monkeypatch.setitem(sys.modules, ENGINE_MODULE, module)


@pytest.mark.parametrize("bad, why", [
    (None, "the engine returned None"),
    ([], "the engine returned a list"),
    ("a string", "the engine returned a bare string"),
    (42, "the engine returned a number"),
    ({}, "the engine returned an empty dict"),
    ({"status": "completed"}, "only a status, no provenance at all"),
])
def test_a_malformed_engine_result_is_refused_not_echoed(monkeypatch, bad, why):
    """Whatever the engine returns, the wrapper must return a schema.

    `discovery_is_live(res)` and `provenance_of(res)` both read `res` as a
    mapping. A non-mapping makes those calls fail, and the `except` arm turns
    that into `status: "error"` -- which is the right shape, but only because
    the exception happened to fire. The test pins the outcome for each kind of
    malformed value so that a change in how these helpers read `res` cannot
    quietly turn a malformed result into an accepted one.
    """
    _broken_engine(monkeypatch, bad)

    out = _discover()

    assert out["status"] != "completed", (
        f"{why}: the wrapper reported success\n{out}")
    assert out["provenance"] != "live", f"{why}: reported live\n{out}"
    assert out["validation_eligible"] is False
    assert out["publication_eligible"] is False
    assert out.get("reason") or out.get("error"), (
        f"{why}: a refusal must say why, got {sorted(out)}")


def test_contradictory_provenance_fields_are_refused(monkeypatch):
    """A payload claiming both live and synthetic must not be accepted.

    `provenance_of` reads one field and `_is_completed` another, so a payload
    can satisfy both while a third field contradicts them. `attested: True`
    with `validation_eligible: False` is the shape: it says a measurement
    happened and also that it is not eligible for validation.
    """
    _fake_engine(monkeypatch, _live_payload(
        provenance="live",
        attested=True,
        validation_eligible=False,
        publication_eligible=False,
    ))

    out = _discover()

    assert out["status"] != "completed", (
        "a payload that declares itself measured but ineligible was accepted\n"
        f"{out}")
    assert out["validation_eligible"] is False


def test_the_error_path_does_not_report_a_payload_as_a_result(monkeypatch):
    """The exception arm must not leak whatever it had in hand.

    `except Exception` catches everything, including a failure *after* a payload
    was obtained. The arm builds a fresh dict, so nothing should leak -- but
    "should" is what this asserts, because a future edit adding `**locals()`
    to the error branch would expose an unverified payload under `status:
    "error"`, which the evidence chain still reads.
    """
    def _explode(**kw):
        raise RuntimeError("engine blew up after building a payload")

    module = types.ModuleType(ENGINE_MODULE)

    class _Engine:
        def __init__(self, *a, **kw2):
            pass

        def discover_and_orchestrate(self, **kw2):
            return _explode()

    module.DiscoveryEngine = _Engine
    monkeypatch.setitem(sys.modules, ENGINE_MODULE, module)

    out = _discover()

    assert out["status"] == "error", out
    assert out["provenance"] == "unavailable", out
    assert out["validation_eligible"] is False
    assert "result" not in out, (
        f"the error path exposed a payload under `result`: {out}")
    assert "metrics" not in out and "raw_traces" not in out, (
        f"the error path leaked payload fields: {sorted(out)}")
    assert out["error"], "an error must carry its message"


# ── Structure ──────────────────────────────────────────────────────────

def test_no_spread_follows_a_guarded_key_in_the_return_literal():
    """The invariant, checked where it is written.

    Python dict literals preserve order and a later duplicate key wins, so the
    AST preserves exactly the property under test. Reading it from the AST
    rather than grepping keeps it immune to reformatting -- the failure mode that
    has broken text-matching guards in this repository eight times.

    Scope widened after the runtime tests above were added. This used to look
    only at dict literals that were the direct value of a `return`, which meant
    it failed closed on an unrelated refactor: moving the literal into a local
    (`out = {...}; return out`) made it report "no return dict with a **spread
    found; the shape changed" and stop guarding anything. It now examines any
    dict literal that contains both a spread and a guarded key, wherever it
    sits. That is the property that matters and it holds across both shapes.

    This remains a *structural* check. The runtime tests above are the primary
    defence -- a wrapper's trust decision has to hold for the response a caller
    receives, not only for the source that produced it -- but the structural
    invariant is cheap to state and catches the reordering at the point it is
    written, where the runtime tests would only see its symptom.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    spreads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and any(k is None for k in node.keys)
        and any(isinstance(k, ast.Constant) and k.value in GUARDED
                for k in node.keys)
    ]
    assert spreads, (
        "no dict literal combines a **spread with a guarded key; the shape "
        "this check guards has changed and it is no longer guarding anything"
    )

    offenders = []
    for node in spreads:
        keys = node.keys
        guarded_positions = [
            i for i, k in enumerate(keys)
            if isinstance(k, ast.Constant) and k.value in GUARDED
        ]
        spread_positions = [i for i, k in enumerate(keys) if k is None]
        for guarded in guarded_positions:
            for spread in spread_positions:
                if spread > guarded:
                    offenders.append(
                        f"line {node.lineno}: **spread follows "
                        f"{keys[guarded].value!r}")

    assert not offenders, (
        "a **spread follows a guarded key, so the payload decides the verdict: "
        + "; ".join(offenders))