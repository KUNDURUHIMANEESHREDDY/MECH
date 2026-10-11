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
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "backend" / "agents" / "discoverer.py"
ENGINE_MODULE = "backend.interpretability.discovery.discovery_engine"

#: The keys the guard decides. A payload spread after these can overwrite them.
GUARDED = ("status", "provenance", "field_provenance")


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

    The measurement record matters here. `_is_attested` deliberately rejects a
    bare `attested: True` boolean: the attestation has to carry the four
    binding fields -- `model_identity`, `input_commitment`,
    `output_commitment`, `execution_context` -- which is what makes it
    evidence rather than a claim. This fixture was written when a boolean
    sufficed, and when that tightened it silently produced payloads that no
    longer cleared the guard, so every test here read `status: "unavailable"`
    and reported a spread defect that did not exist.

    The fixture therefore satisfies the real contract, with placeholder
    commitments. What this file tests is the *spread ordering* -- whether a
    payload can overwrite the guard's verdict -- which does not depend on the
    record's contents beyond their presence. Whether those commitments are
    genuine is `test_measurement_record.py`'s subject, not this file's.
    """
    payload: Dict[str, Any] = {
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "measurement_record": {
            "model_identity": {"model_name": "gpt2"},
            "input_commitment": "sha256:" + "0" * 64,
            "output_commitment": "sha256:" + "1" * 64,
            "execution_context": {"device": "cpu", "dtype": "torch.float32"},
        },
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


# ── Structure ──────────────────────────────────────────────────────────

def test_no_spread_follows_a_guarded_key_in_the_return_literal():
    """The invariant, checked where it is written.

    Python dict literals preserve order and a later duplicate key wins, so the
    AST preserves exactly the property under test. Reading it from the AST
    rather than grepping keeps it immune to reformatting -- the failure mode that
    has broken text-matching guards in this repository eight times.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    spreads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        and any(k is None for k in node.value.keys)
    ]
    assert spreads, "no return dict with a **spread found; the shape changed"

    offenders = []
    for ret in spreads:
        keys = ret.value.keys
        guarded_positions = [
            i for i, k in enumerate(keys)
            if isinstance(k, ast.Constant) and k.value in GUARDED
        ]
        spread_positions = [i for i, k in enumerate(keys) if k is None]
        for guarded in guarded_positions:
            for spread in spread_positions:
                if spread > guarded:
                    offenders.append(
                        f"line {ret.lineno}: **spread follows {keys[guarded].value!r}")

    assert not offenders, (
        "a **spread follows a guarded key, so the payload decides the verdict: "
        + "; ".join(offenders))