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
    Written without fallback keys or conditional assertions.

    An earlier version tried `active_heads`, then `circuit`, then `heads`, and
    guarded the check with two nested `if`s. If none of the three keys was
    present -- or if the value found was not a list -- the entire block was
    skipped and the only thing that ran was the rejection check on the bad
    label. The test then passed while verifying nothing about whether the
    valid head survived.

    That is worse than a missing assertion, because it reads as coverage: a
    reviewer sees a test named "reports the circuit it actually ran" with
    assertions inside it. `rejected_heads` being truthy says the typo was
    noticed. It does not say `L9H6` was kept -- and a response that rejected
    every head would have passed unchanged.
    """
    good = ["L9H6"]
    partial = ["L9H6", "garbage"]

    result = adapter.run_isolated_circuit(PROMPT, partial)

    # The executed circuit must be named. No fallback keys: the contract is
    # that one field carries it, and a shim across three names is exactly the
    # flexibility that hides a contract being dropped.
    assert "active_heads" in result, (
        f"the response does not name the circuit that ran; keys are "
        f"{sorted(result)}. A partially readable circuit that does not report "
        f"what it actually executed cannot be attributed to anything."
    )

    reported = result["active_heads"]
    assert isinstance(reported, list), (
        f"active_heads is {type(reported).__name__}, expected a list of the "
        f"heads that survived: {reported!r}"
    )

    # Exact contents, not merely containment. A response listing every head
    # would satisfy a containment check while describing a different
    # experiment.
    assert [h.upper() for h in reported] == ["L9H6"], (
        f"the executed circuit was {reported}, expected exactly {good}"
    )

    # And the rejected label must be named, with its exact value.
    assert result.get("rejected_heads") == ["garbage"], (
        f"rejected_heads is {result.get('rejected_heads')!r}, expected "
        f"['garbage']"
    )

    # The measured result must correspond to the reported circuit.
    assert result.get("circuit_isolated") is True, (
        f"circuit_isolated is {result.get('circuit_isolated')!r}; the response "
        f"reports a circuit but does not claim to have isolated it"
    )

    # Adding the bad label must change only what is reported, not what ran.
    clean = adapter.run_isolated_circuit(PROMPT, good)
    assert clean["active_heads"] == result["active_heads"], (
        f"adding a rejected label changed the executed circuit from "
        f"{clean['active_heads']} to {result['active_heads']}"
    )
    assert clean["top_token"] == result["top_token"], (
        f"adding a rejected label changed the measured output from "
        f"{clean['top_token']!r} to {result['top_token']!r}; rejection must be "
        f"disclosure, not a change to the intervention"
    )


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

    Proven at runtime, not by reading the source.

    An earlier version of this test asserted that `inspect.getsource()`
    contains `_attn_implementation` or `eager`. A review correctly pointed out
    that proves nothing: the string could live in a comment, a dead branch, an
    unrelated statement, or a setting overwritten a line later. All four of
    those pass that check while the forward pass runs on sdpa.

    So the kernel is now recorded from inside the forward pass. A pre-hook on
    `c_proj` reads `config._attn_implementation` at the moment the attention
    has already run and the mask is being applied, and that reading must be
    `eager`. The source check is kept below as a tripwire for the narrow case
    where someone deletes the pinning entirely -- but it is no longer what
    establishes the property.
    """
    import torch

    model = adapter._model
    layer, head = 5, 3
    observed: list[str] = []
    config = model.config

    def _observe(module, inputs):
        observed.append(str(getattr(config, "_attn_implementation", None)))

    probe = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
        _observe)
    try:
        adapter.run_isolated_circuit(PROMPT, [f"L{layer}H{head}"])
    finally:
        probe.remove()

    assert observed, (
        "the capture hook never fired; the probe did not observe the forward "
        "pass this test is meant to inspect"
    )
    assert set(observed) == {"eager"}, (
        f"run_isolated_circuit ran attention under "
        f"{sorted(set(observed))}, not 'eager', while capture_head_outputs "
        f"forces eager. A circuit isolation compared against a captured "
        f"activation then carries a constant kernel-disagreement term that is "
        f"indistinguishable from a real circuit effect."
    )

    # The caller's configuration must be restored afterwards, or every later
    # measurement silently changes kernel. Recorded rather than asserted
    # against a literal, because a model legitimately configured for eager
    # must stay eager -- asserting "restored" by comparing to "sdpa" would
    # fail for exactly the case where nothing needed restoring.
    restored_to = str(getattr(config, "_attn_implementation", None))
    assert restored_to in ("sdpa", "eager"), (
        f"after the call the model's attention implementation reads "
        f"{restored_to!r}; the pinning must leave a usable value behind, not "
        f"None or an unknown string"
    )

    # Secondary guard: the pinning should still be written down, so that
    # deleting it is visible in review rather than only at runtime.
    import inspect

    import backend.science.models.gpt2_adapter as module

    source = inspect.getsource(module.GPT2Adapter.run_isolated_circuit)
    assert "_attn_implementation" in source, (
        "the runtime check above passed, but run_isolated_circuit no longer "
        "mentions _attn_implementation anywhere. If the pinning was removed "
        "and replaced by something equivalent this is fine; if it was removed "
        "and the model merely happens to be configured for eager, the next "
        "model loaded differently will not be."
    )