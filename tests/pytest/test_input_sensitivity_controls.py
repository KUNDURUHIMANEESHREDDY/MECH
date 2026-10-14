"""Adversarial controls: is a "measurement" sensitive to what it claims to measure?

Finding J, stated mechanically
------------------------------
A result can satisfy every internal check at once -- arithmetic consistent,
shape right, code executed, values finite, status `ok` -- and still be
scientifically meaningless. Shape and consistency say the code ran; they say
nothing about whether the number means anything.

The sharpest available control for that is **input sensitivity**. A quantity
that claims to measure a property of its input, and returns exactly the same
thing for inputs with nothing in common, is not measuring that input. It is
returning a constant with a plausible shape.

Measured on `causal_tracing`, which reports a per-layer indirect effect for a
prompt pair:

    "The capital of France is"          -> layer_effects[0] = 0.12, [1] = 0.17 ...
    "Quantum chromodynamics describes"  -> layer_effects[0] = 0.12, [1] = 0.17 ...
    "2 + 2 ="                           -> layer_effects[0] = 0.12, [1] = 0.17 ...

Byte-identical across prompts about geography, particle physics and
arithmetic. Whatever the curve means, it is not a function of the prompt.

Why this control and not another
--------------------------------
It needs no reference implementation, so it runs against a subsystem that has
no correct answer to compare against -- which is the common case. And it
cannot be passed by being internally consistent: a constant is perfectly
consistent. It fails precisely the failure mode that shape and arithmetic
checks cannot see.

The negative control that matters
---------------------------------
This file must itself be provable. `test_a_constant_function_is_caught`
asserts the detector fires on a function that is constant by construction. A
sensitivity check that silently cannot fail is the exact defect this project
has produced repeatedly -- a green assertion attached to nothing -- so the
detector is tested against a known-bad input before being trusted against real
engines.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Sequence

import pytest

#: Two prompt pairs with no semantic relationship, chosen so that any engine
#: genuinely reading either would report something different.
DIVERSE_PAIRS = [
    ("The capital of France is", "The capital of Rome is"),
    ("Quantum chromodynamics describes", "Classical thermodynamics describes"),
    ("She gave the book to her", "She gave the book to Mary"),
]


def _strip_volatile(payload: Any, inputs: Sequence[str] = ()) -> str:
    """Canonical form, minus anything legitimately run-dependent.

    Two categories are removed, and the second one matters more:

    * **Run-dependent fields** -- timestamps and ids vary between calls for
      reasons that have nothing to do with the input.

    * **Echoed inputs** -- any string value equal to one of the arguments is
      dropped. Without this the control is defeated by an engine that simply
      returns its own prompt, which is the common wrapper shape in this
      codebase: the response differs between prompts *because it contains the
      prompt*, not because anything was measured. That failure is silent --
      the test goes green and proves nothing -- so it is pinned by
      `test_an_engine_that_only_echoes_its_input_is_caught`.

    Everything else is the claim under test.
    """
    volatile = {"timestamp", "created_at", "id", "discovery_id", "session_id",
                "elapsed", "duration", "latency_ms", "run_id"}
    echoed = {str(i) for i in inputs}

    def scrub(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: scrub(v) for k, v in node.items() if k not in volatile}
        if isinstance(node, (list, tuple)):
            return [scrub(i) for i in node]
        if isinstance(node, str) and node in echoed:
            return "<echoed-input>"
        return node

    return json.dumps(scrub(payload), sort_keys=True, default=str)


def assert_input_sensitive(
    call: Callable[[str, str], Any],
    pairs: Sequence[Sequence[str]] = DIVERSE_PAIRS,
    *,
    what: str = "engine",
) -> List[str]:
    """Every pair must produce a distinct response. Returns the duplicates.

    Raises `AssertionError` when two semantically unrelated inputs give a
    byte-identical answer, naming the offending pairs.
    """
    seen: Dict[str, List[str]] = {}
    for clean, corrupt in pairs:
        key = _strip_volatile(call(clean, corrupt), (clean, corrupt))
        seen.setdefault(key, []).append(f"{clean!r} / {corrupt!r}")

    duplicates = {k: v for k, v in seen.items() if len(v) > 1}
    if duplicates:
        detail = "; ".join(
            " == ".join(prompts) for prompts in duplicates.values())
        raise AssertionError(
            f"{what} returned a byte-identical response for semantically "
            f"unrelated inputs ({len(duplicates)} collision(s)): {detail}. "
            f"A value that does not vary with its input is a constant with a "
            f"plausible shape, not a measurement of that input."
        )
    return list(seen)


# ── the detector is tested against a known-bad input ──────────────────

def test_a_constant_function_is_caught():
    """The control on the control.

    Without this, every assertion below could be a no-op and the file would
    report clean for reasons that have nothing to do with the engines.
    """
    def constant(clean: str, corrupt: str) -> Dict[str, Any]:
        return {"status": "ok", "effect": 0.12, "top": "Attention"}

    with pytest.raises(AssertionError) as excinfo:
        assert_input_sensitive(constant, what="a deliberately constant function")

    message = str(excinfo.value)
    assert "byte-identical" in message, message


def test_a_genuinely_input_sensitive_function_passes():
    """The other direction: the check must not fire on a varying function."""
    def sensitive(clean: str, corrupt: str) -> Dict[str, Any]:
        return {"status": "ok", "len": len(clean) + len(corrupt)}

    assert_input_sensitive(sensitive, what="a sensitive function")


def test_a_timestamp_difference_does_not_trip_the_check():
    """Run-dependent fields are excluded, or every engine looks constant.

    Without this scrub, any engine that stamps `timestamp` would report as
    constant and the finding would be an artefact of the control.
    """
    counter = {"n": 0}

    def timestamped(clean: str, corrupt: str) -> Dict[str, Any]:
        counter["n"] += 1
        return {
            "status": "ok",
            "timestamp": counter["n"],
            # Varies with the input, and not merely by coincidence: the
            # control pairs include prompts of equal length
            # ("The capital of France is" and "She gave the book to her" are
            # both 24 characters), so a length-based value would collide and
            # the test would fail for a reason unrelated to timestamps.
            "v": sum(map(ord, clean)) + sum(map(ord, corrupt)),
        }

    assert_input_sensitive(timestamped, what="a timestamped function")


def test_an_engine_that_only_echoes_its_input_is_caught():
    """The failure that would have made this whole file vacuous.

    `causal_tracing` returned byte-identical `layer_effects` for every prompt
    pair, and its first version of this test still passed -- because the
    response contains the prompts, so two calls differed in the one field
    that says nothing about whether anything was measured.

    An engine that returns `{"status": "ok", "prompt": clean}` has measured
    nothing and must be reported.
    """
    def echo(clean: str, corrupt: str) -> Dict[str, Any]:
        return {"status": "ok", "provenance": "live", "prompt": clean,
                "corrupt_prompt": corrupt}

    with pytest.raises(AssertionError) as excinfo:
        assert_input_sensitive(echo, what="a pure echo")

    assert "byte-identical" in str(excinfo.value)


# ── applied to the scientific subsystems ─────────────────────────────

def test_causal_tracing_varies_with_the_prompt_pair() -> None:
    """The subsystem measured above, as a standing control."""
    from backend.interpretability.causal.causal_tracing import CausalTracingEngine

    engine = CausalTracingEngine()
    assert_input_sensitive(
        lambda clean, corrupt: engine.trace_causal_effect(clean, corrupt),
        what="CausalTracingEngine.trace_causal_effect",
    )


def test_attribution_patching_varies_with_the_prompt_pair() -> None:
    """The same control on the sibling causal engine."""
    from backend.interpretability.causal.attribution_patching import (
        AttributionPatchingEngine,
    )

    engine = AttributionPatchingEngine()
    method = next(
        (m for m in dir(engine)
         if not m.startswith("_") and callable(getattr(engine, m))
         and m != "compute_attribution"),
        "compute_attribution",
    )
    assert hasattr(engine, method), (
        f"AttributionPatchingEngine exposes no entry point this control can "
        f"call; public methods: "
        f"{[m for m in dir(engine) if not m.startswith('_')]}"
    )
    assert_input_sensitive(
        lambda clean, corrupt: getattr(engine, method)(clean, corrupt),
        what=f"AttributionPatchingEngine.{method}",
    )