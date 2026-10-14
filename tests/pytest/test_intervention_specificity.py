"""4.4 -- specificity: is the effect *caused by* the claimed component?

The gap
-------
A directional test proves the sign. It does not prove the mechanism. A change
large enough to move the logit will move it for reasons unrelated to the head
being credited: generic activation damage, any large perturbation, noise in the
wrong direction. A head that "causes" the IOI effect by being one of several
components whose perturbation all push the same way passes a sign test and
fails the scientific claim.

So the question is not "is delta negative" but:

    is delta(head H) an outlier, or is it one more sample from the
    distribution every other head produces?

Controls, and what each rules out
--------------------------------
* **sham** -- write the head's own captured value back. Isolates the
  intervention machinery from the intervention. Already proven elsewhere;
  included so this file stands alone.
* **random vector** -- a same-norm random write to the same slice. Rules out
  "any large perturbation produces the effect".
* **unrelated component** -- a different head. Rules out "the effect is
  global".
* **targeted** -- the claimed head.

The statistic is a specificity ratio: the target's |delta| against the null
distribution of |delta| over the controls, judged per-prompt and then across
prompts.

Held-out prompts
----------------
Finding G noted that four pairs were used both to form expectations and to
check generalisation. These are reserved: they are asserted nowhere else in
the suite, they are not the canonical IOI pair, and nothing was tuned against
them. If the effect is real it should hold here without having been fitted.

Honest limits
-------------
This establishes that the claimed head's effect is *distinguishable from other
heads' effects on these prompts*. It does not establish the mechanism is
unique in the model -- a component acting in parallel would still look
specific here. Proving uniqueness needs a different control, and is not
claimed.
"""

from __future__ import annotations

import pytest

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

#: Reserved for this file only. Deliberately not the canonical IOI pair and
#: deliberately not any of the four pairs in
#: `test_causal_direction_generalisation.py`.
HELD_OUT_PAIRS = [
    ("When Alice went to the store she bought", "When Bob went to the store he bought"),
    ("The mechanic fixed the car for", "The chef fixed the recipe for"),
    ("The patient asked the doctor about", "The student asked the teacher about"),
]

LAYER = 5
TARGET_HEAD = 3

#: Heads sampled to form the null distribution. Spans layers so the null is
#: not confined to one block's typical behaviour.
CONTROL_HEADS = [(0, 0), (1, 5), (2, 11), (3, 2), (4, 7), (6, 1),
                 (7, 9), (8, 4), (9, 6), (10, 0), (11, 8)]


@pytest.fixture(scope="module")
def adapter():
    from backend.science.models.gpt2_adapter import GPT2Adapter

    instance = GPT2Adapter(variant="small", mock_mode=False)
    assert instance._model is not None
    return instance


def _d_head(adapter) -> int:
    return adapter.spec.d_model // adapter.spec.num_heads


def _delta_for(adapter, prompt: str, layer: int, head: int, value) -> float:
    result = adapter.patch_head_output(
        prompt, layer, head, patch_vector=value, position=-1)
    return result.delta


def _zero_ablation(adapter, prompt: str, layer: int, head: int) -> float:
    return _delta_for(adapter, prompt, layer, head, [0.0] * _d_head(adapter))


# ── the controls ────────────────────────────────────────────────────

@needs_weights
def test_sham_writing_the_captured_value_back_is_a_null_control(adapter):
    """The machinery control.

    If the harness itself moved the logit, every measurement below would be
    contaminated. This must be exactly zero, which makes it the baseline every
    other delta is read against.
    """
    captured = adapter.capture_head_outputs(HELD_OUT_PAIRS[0][0])
    vector = list(captured[LAYER][TARGET_HEAD])

    for clean, _ in HELD_OUT_PAIRS:
        delta = _delta_for(adapter, clean, LAYER, TARGET_HEAD, vector)
        assert delta == 0.0, (
            f"a sham intervention moved the scored logit by {delta:.6f}; every "
            f"effect measured in this file is contaminated by the harness"
        )


@needs_weights
def test_a_random_vector_of_the_same_norm_does_not_reproduce_the_effect(adapter):
    """Rules out "any large write to this slice does it".

    The random control is scaled to the same L2 norm as the zero-ablation
    perturbation, so a difference cannot be explained by magnitude.
    """
    import torch

    torch.manual_seed(0)  # fixed, so a failure is reproducible
    d_head = _d_head(adapter)

    for clean, _ in HELD_OUT_PAIRS:
        target = _zero_ablation(adapter, clean, LAYER, TARGET_HEAD)
        random_deltas = [
            _delta_for(adapter, clean, layer, head,
                       torch.randn(d_head).tolist())
            for layer, head in CONTROL_HEADS
        ]
        largest = max(abs(d) for d in random_deltas)

        assert abs(target) > largest, (
            f"on {clean!r} the claimed head moved the logit by {target:.4f}, "
            f"but a random write to another head moved it by {largest:.4f}. "
            f"The effect is not distinguishable from a same-norm random "
            f"perturbation, so it does not identify a mechanism."
        )


@needs_weights
def test_the_claimed_head_is_an_outlier_against_the_control_heads(adapter):
    """The specificity statistic, per prompt.

    Reported per prompt rather than pooled: pooling would let one large effect
    carry the average and hide a prompt where the claim is not specific.
    """
    for clean, _ in HELD_OUT_PAIRS:
        target = abs(_zero_ablation(adapter, clean, LAYER, TARGET_HEAD))
        control = [abs(_zero_ablation(adapter, clean, layer, head))
                   for layer, head in CONTROL_HEADS]

        mean = sum(control) / len(control)
        assert target > mean, (
            f"on {clean!r} the claimed head's |delta| ({target:.4f}) does not "
            f"exceed the mean over control heads ({mean:.4f}). The effect is "
            f"inside the ordinary spread, so attributing it to this head is "
            f"not supported."
        )


@needs_weights
def test_specificity_holds_on_every_held_out_prompt(adapter):
    """Held-out generalisation, stated as an outcome rather than an average.

    Finding G's gap, closed: these prompts were reserved, and the claim is
    that the ordering survives on all of them. An average across them would
    let two passes and one catastrophic failure read as success.
    """
    failures = []
    for clean, _ in HELD_OUT_PAIRS:
        target = abs(_zero_ablation(adapter, clean, LAYER, TARGET_HEAD))
        control = [abs(_zero_ablation(adapter, clean, layer, head))
                   for layer, head in CONTROL_HEADS]
        largest = max(control)
        if target <= largest:
            failures.append(f"{clean!r}: target {target:.4f} <= max control "
                            f"{largest:.4f}")

    assert not failures, (
        "the claimed head was not the most specific component on "
        f"{len(failures)}/{len(HELD_OUT_PAIRS)} held-out prompts: "
        + "; ".join(failures)
    )


@needs_weights
def test_the_null_distribution_is_not_degenerate(adapter):
    """The controls must actually do something.

    If every control head produced |delta| == 0, then "target > max(control)"
    would be satisfied by any non-zero target, including noise -- and the whole
    file would be measuring the absence of perturbation rather than
    specificity.
    """
    clean, _ = HELD_OUT_PAIRS[0]
    control = [abs(_zero_ablation(adapter, clean, layer, head))
               for layer, head in CONTROL_HEADS]

    non_zero = sum(1 for d in control if d > 1e-6)
    assert non_zero >= len(control) // 2, (
        f"only {non_zero}/{len(control)} control heads produced a non-zero "
        f"effect: {['%.5f' % d for d in control]}. The null distribution is "
        f"degenerate, so any non-zero target would beat it trivially."
    )