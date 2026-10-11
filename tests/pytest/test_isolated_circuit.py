"""PAT-010: all-head patching, at `GPT2Adapter.run_isolated_circuit`.

`run_isolated_circuit` is called by `ioi_pipeline.py:818` and by no test. It
is the "all-head" arm of the checklist's PAT-010 -- single-head and
residual-stream patching are covered elsewhere; this one keeps a listed set of
heads and zeroes every other head in every layer.

The masking mechanism is correct. Measured, not assumed: with a circuit of
`["L0H0"]`, layer 0's head 0 survives at norm 0.805176 while heads 1-3 are
exactly 0.0, and an empty circuit drives the top logit from -92.875 (plain) to
-46.250, so the ablation genuinely fires.

The defect is in what happens to head labels it cannot read
-------------------------------------------------------------------------
The parse is wrapped in `except Exception: continue`, so an unparseable label
is dropped without a word. Measured:

    []                    top logit -46.2500
    ['garbage']           top logit -46.2500   <- identical to the empty circuit
    ['L99H99']            top logit -46.2500   <- identical to the empty circuit
    ['L9H6']              top logit -46.5000
    ['L9H6', 'garbage']   top logit -46.5000   <- identical to ['L9H6'] alone

So a caller asking for circuit `["garbage"]` receives a complete, plausible,
correctly-shaped result describing an *empty* circuit, and a caller asking for
`["L9H6", "typo"]` silently receives a different circuit than the one they
named. This is the substitution failure the project rules forbid: the
measurement is real, but it is not the measurement that was requested, and
nothing in the response distinguishes the two.

Out-of-range indices fail the same way. `L99H99` parses successfully as a
tuple, then the mask hook's `if x.shape[-1] >= end` guard means the slice is
simply never written -- so a layer that does not exist contributes nothing and
the circuit quietly shrinks.
"""

from __future__ import annotations

import pytest

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

PROMPT = "John gave a drink to Mary"


@pytest.fixture(scope="module")
def adapter():
    from backend.science.models.gpt2_adapter import GPT2Adapter

    instance = GPT2Adapter(variant="small", mock_mode=False)
    assert instance._model is not None, "adapter has no weights"
    return instance


def _top_logit(result: dict) -> float:
    return float(result["top_tokens"][0]["logit"])


def _plain_forward_top_logit(adapter, prompt: str) -> float:
    """The unpatched forward pass, run under the kernel the method pins.

    The baseline must be captured the same way `run_isolated_circuit` runs, or
    the comparison carries a constant kernel-disagreement term. GPT-2 Small's
    sdpa and eager attention differ by 0.0625 on this top logit -- exactly the
    term found in PAT-005 for `patch_head_output`. Capturing the baseline under
    the model default made this test fail by precisely that amount on a run
    that was in fact correct; that is the same mistake the PAT-005 baseline
    test made, and it is recorded here so it is not made a third time.
    """
    import torch

    config = adapter._model.config
    previous = getattr(config, "_attn_implementation", None)
    if previous != "eager":
        config._attn_implementation = "eager"
    try:
        inputs = adapter._tokenizer(prompt, return_tensors="pt").to(
            adapter._model.device
        )
        with torch.no_grad():
            logits = adapter._model(**inputs).logits[0, -1, :]
        return float(logits.max())
    finally:
        if previous is not None:
            config._attn_implementation = previous


# ── the mechanism, which is correct and must stay correct ──────────────

@needs_weights
def test_listed_heads_survive_and_every_other_head_is_zeroed(adapter):
    """The core promise: only the named heads remain active.

    Observed through a forward hook on `c_proj`, which is where the mask is
    applied -- asserting on the returned logits instead would not distinguish
    "the listed head survived" from "the listed head happened not to matter".
    """
    import torch

    d_head = adapter.spec.d_model // adapter.spec.num_heads
    seen: dict = {}
    probe = adapter._model.transformer.h[0].attn.c_proj.register_forward_hook(
        lambda _m, i, _o: seen.__setitem__("x", i[0].detach().clone())
    )
    try:
        adapter.run_isolated_circuit(PROMPT, ["L0H0"])
    finally:
        probe.remove()

    row = seen["x"][0, -1, :]
    kept = row[0:d_head]
    assert float(kept.norm()) > 0.0, (
        "the head named in the circuit was zeroed; the isolation is inverted"
    )

    for head in range(1, adapter.spec.num_heads):
        start = head * d_head
        masked = row[start:start + d_head]
        assert float(masked.abs().max()) == 0.0, (
            f"head {head} was not zeroed under circuit ['L0H0'] "
            f"(norm {float(masked.norm()):.6f}); the circuit is not isolated"
        )


@needs_weights
def test_an_unrestricted_circuit_reproduces_the_unpatched_forward_pass(adapter):
    """Naming every head must be equivalent to ablating nothing.

    This is the reference case for the whole method: it should be a no-op. If
    it is not, every restricted circuit is measured against a baseline that is
    itself perturbed.
    """
    import torch

    every = ["L%dH%d" % (l, h)
             for l in range(adapter.spec.num_layers)
             for h in range(adapter.spec.num_heads)]

    restricted = adapter.run_isolated_circuit(PROMPT, every)

    plain = _plain_forward_top_logit(adapter, PROMPT)

    assert _top_logit(restricted) == pytest.approx(plain, abs=1e-3), (
        f"naming every head gave {_top_logit(restricted)}, but the unpatched "
        f"forward pass gives {plain}. A fully-active circuit must reproduce "
        f"the baseline exactly."
    )


@needs_weights
def test_an_empty_circuit_ablates_everything(adapter):
    """The degenerate circuit is defined, and it is not the baseline.

    An explicitly empty list is a legitimate request -- isolate nothing -- not
    a parse failure. An earlier draft of the refusal path in the adapter
    conflated the two and turned this control condition into an error, so the
    assertion is pinned explicitly: an empty circuit must *measure*, and must
    say so.
    """
    plain = _plain_forward_top_logit(adapter, PROMPT)

    empty = adapter.run_isolated_circuit(PROMPT, [])

    assert empty.get("status") == "ok", (
        f"an explicitly empty circuit must measure, not refuse: {empty}"
    )
    assert empty.get("all_heads_ablated") is True
    assert _top_logit(empty) != pytest.approx(plain, abs=1e-3), (
        "an empty circuit reproduced the unpatched result; the ablation is "
        "not being applied at all"
    )


# ── the defect: unreadable head labels are dropped in silence ──────────

@needs_weights
@pytest.mark.parametrize("labels", [
    ["garbage"],
    ["L99H99"],
    ["L9"],
    ["H6"],
    ["L9H6", "garbage"],
])
def test_head_labels_that_cannot_be_read_are_reported_not_silently_dropped(
    adapter, labels
):
    """A circuit that could not be parsed must say so.

    The defect: the parse was `except Exception: continue`, so every label in
    this list was discarded without a word. `["garbage"]` and `["L99H99"]`
    measured an *empty* circuit -- byte-identical top logit to `[]` -- and
    `["L9H6", "garbage"]` measured `["L9H6"]` alone. A caller received a
    complete, plausible result describing a circuit they had not asked for.

    Whatever the method returns, it must not report a successful measurement of
    a circuit it did not run.
    """
    result = adapter.run_isolated_circuit(PROMPT, labels)

    rejected = (
        result.get("rejected_heads")
        or result.get("unparsed_heads")
        or result.get("invalid_heads")
    )
    assert rejected, (
        f"these labels were silently dropped and the result still looks like a "
        f"measurement: {labels}\n"
        f"  top_token={result.get('top_token')!r} "
        f"top_logit={result.get('top_tokens', [{}])[0].get('logit')!r}\n"
        f"  Nothing in the response says the circuit was not the one requested."
    )


@needs_weights
def test_a_fully_unreadable_circuit_is_refused(adapter):
    """If no head could be read, the honest answer is refusal, not a number.

    `["garbage"]` must not return the empty-circuit measurement as though it
    were an isolated run of a circuit the caller named. There is nothing to
    isolate, so nothing was measured.
    """
    result = adapter.run_isolated_circuit(PROMPT, ["garbage", "also-garbage"])

    assert result.get("status") != "ok" or result.get("circuit_isolated") is False, (
        "a circuit with no readable heads reported a successful isolation: "
        f"{result}"
    )
    if result.get("status") != "ok":
        assert result.get("reason"), "a refusal must say why"


@needs_weights
def test_a_partially_readable_circuit_reports_the_circuit_it_actually_ran(
    adapter,
):
    """One bad label among good ones must not shrink the circuit silently.

    `["L9H6", "garbage"]` measured exactly `["L9H6"]`. If a circuit is
    partially readable the response must name the heads that survived, so the
    result can be attributed to the circuit that was actually run rather than
    to the one that was requested.
    """
    good = ["L9H6"]
    partial = ["L9H6", "garbage"]

    result = adapter.run_isolated_circuit(PROMPT, partial)

    reported = (
        result.get("active_heads")
        or result.get("circuit")
        or result.get("heads")
    )
    if reported is not None:
        normalised = {h.upper() for h in reported} if isinstance(reported, list) else None
        if normalised is not None:
            assert all(
                label.upper() in normalised
                for label in good
            ), f"the circuit that ran {reported} omits the readable heads {good}"

    # And the bad label must still be visible.
    assert result.get("rejected_heads") or result.get("unparsed_heads") or (
        result.get("invalid_heads")
    ), f"the unreadable label in {partial} was not reported: {result}"


# ── kernel parity, the PAT-005 defect recurring ────────────────────────

@needs_weights
def test_isolation_uses_the_same_attention_kernel_as_capture(adapter):
    """This path must run the same kernel `capture_head_outputs` runs.

    The same asymmetry fixed for `patch_head_output`: `capture_head_outputs`
    forces `attn_implementation="eager"` so the captured head vector and the
    tensor patched back into it come from one kernel, while this method calls
    the model directly and so runs whatever the model is configured for.

    Measured: this path runs `sdpa` while capture runs `eager`. That puts a
    constant kernel-disagreement term inside every circuit isolation the IOI
    pipeline reports -- the same defect class as PAT-005, one function over.
    """
    import inspect

    import backend.science.models.gpt2_adapter as module

    source = inspect.getsource(module.GPT2Adapter.run_isolated_circuit)
    assert "_attn_implementation" in source or "eager" in source, (
        "run_isolated_circuit does not pin the attention implementation, so it "
        "runs whatever the model is configured for while capture_head_outputs "
        "forces eager. A circuit isolation compared against a captured "
        "activation then carries a constant kernel-disagreement term that is "
        "indistinguishable from a real circuit effect."
    )