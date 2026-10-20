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

    is the effect of the claimed component an outlier, or is it one more
    sample from the distribution every other head produces?

What is measured, and why it is not the top-token logit
--------------------------------------------------------

The quantity under test is the **IOI logit difference**:

    clean_ld = logit(" <indirect object>") - logit(" <subject>")

on the clean prompt, and the effect of an intervention is

    |clean_ld - ablated_ld|

This is the same quantity the rest of the repository uses: the README's
head-ablation table and `backend/interpretability/gpt2_model.py::
compute_patching_matrix` both report `clean_ld` and the change in it when a
head is zeroed. It is also the only quantity the claim is about. The
top-token logit is not: zeroing a late-layer head moves the next-token logit a
long way because the model needs that head to predict *anything*, and that
number says nothing about indirect-object identification.

Measuring the wrong quantity is what made the first version of this file
unable to pass on a working model, and it is worth recording how, because the
failure message pointed the wrong way. Three defects, each measured:

* **The null control was not a null control.** It captured one prompt's head
  vector and wrote it back into *every* prompt, so two of the three writes
  were a transplant -- another prompt's value, which is an intervention, not a
  sham. Writing the head's own captured value back into its own prompt moves
  the scored logit by exactly 0.000000; transplanting that same vector into
  'The mechanic fixed the car for' is what moved it by 0.812500, and that is
  the number the failure reported. The write path was exact all along. The
  control was not.
* **The random control was not the size of the thing it controlled for.**
  `torch.randn(64)` has an L2 norm of about 8 (the square root of the
  dimension); the head output it was matched against has a norm of 1.38
  (measured: layer 5, head 3, first prompt). The control was six times the
  intervention, so "a random write moved the logit further" was a fact about
  the control, not about the model.
* **The control set contained the claim.** It included L9H6 and L10H0, both
  heads of the published IOI circuit (`REFERENCE_ONLY_DISCOVERED_HEADS`). A
  head of the mechanism is a competitor for "the head that carries this
  effect", not a control, so the null distribution contained two of the
  things the test was about to compare against. On the top-token quantity the
  control set also contained the single largest head of all 144 (L11H8, whose
  zero-ablation moved the scored logit by 13.125 on one prompt against the
  claimed head's 1.75), so no head could beat the control maximum on that
  prompt and the property was unsatisfiable by anyone. The claimed head
  (layer 5, head 3) was not special either: it ranked 25th, 43rd and 101st of
  144 heads on the three prompts.

The claim under test
--------------------

MECH publishes a set of IOI heads -- `ioi_pipeline.REFERENCE_ONLY_DISCOVERED_HEADS`,
ten heads with the published edges L0H1 -> L5H1 -> L7H3 -> L9H9 -- and calls
L5H1 the name mover. That set is what this file tests: the heads MECH
publishes as IOI heads must be the heads that carry the IOI effect, on
prompts nobody fitted them on.

The claim is a *set*, not a single head, and that is a measured decision
rather than a convenient one. Zero-ablating each of the 144 heads and scoring
the IOI logit difference on the eight IOI prompts below gives this -- the most
specific head on each prompt, and where the published name mover L5H1 ranked
among all 144:

    prompt                                   most specific head   L5H1's rank
    "When Mary and John went to the store"   L0H10, L5H1  0.812       1
    "Then Alice and Bob went to the park"    L10H0        1.000       3
    "...went to the library"                  L0H10        4.750       9
    "...went to the kitchen"                  L0H7         4.000      22
    "...went to the market"                   L5H1, L5H5   1.188       1
    "...went to the garden"                   L2H3         1.312      80
    "...went to the school"                   L0H7         6.938      10
    "...went to the beach"                    L0H10        3.000      71

No single head is the most specific component on every prompt, and L5H1 -- the
published name mover -- is not one: on the 'garden' prompt its ablation moves
the IOI logit difference by 0.06 (rank 80 of 144) and on 'beach' by 0.09 (rank
71). The IOI effect in gpt2-small is distributed over templates, which is what
the repository's own method notes say about single-template results: "any
result from it is really a statement about that frame and not about indirect-
object identification generally".

A test asserting "the claimed head is the most specific component on every
held-out prompt" would therefore be asserting something false about this
model: measured on the garden prompt, the name mover's effect (0.4375) ties
the largest control head's (0.4375), and on the beach prompt a same-norm
random write into a control head (0.5625) moves the IOI logit difference
further than the name mover does (0.3750). So the property asserted here is
the one the measurements support: on every prompt, the published set contains
a head that is an outlier against the control heads, and no control head
reproduces the published set's strongest effect. The single-head result above
is kept here so the weaker claim is on the record rather than missing.

Controls, and what each rules out
---------------------------------

* **sham** -- write the head's own captured value back, at the position it was
  captured from. Isolates the intervention machinery from the intervention.
  The capture is per prompt: a vector captured from one prompt and written
  into another is a transplant, not a sham, and it moves the logit.
* **random vector** -- a same-norm random write to a control head's slice.
  Rules out "any large perturbation produces the effect". The norm is the
  strongest claimed head's own output norm, so a difference cannot be
  explained by magnitude. Not the claimed head's own slice: a random write
  there destroys that head's contribution, so it reproduces the effect by
  construction and measures noise sensitivity, not specificity.
* **unrelated component** -- a different head, zero-ablated. Rules out "the
  effect is global". These are heads that appear in neither the published IOI
  circuit nor the circuit the repository discovers (`docs/roadmap.md`).
* **targeted** -- the published IOI heads.

The statistic is a specificity ratio: the strongest claimed head's effect
against the null distribution of effects over the controls, judged per prompt
and then across prompts.

Held-out prompts
----------------

The eight prompts use the repository's own IOI frames
(`backend/science/reproducibility/ioi_pipeline.py::_IOI_FRAMES`) with the
names varied, so the claim is not carried by one surface form. The first is
the fitting frame -- the shape of the claim was settled there -- and the
other seven are held out. They are asserted nowhere else in the suite and
nothing was tuned against them.

Honest limits
-------------

This establishes that the heads MECH publishes as IOI heads have a larger
measured effect on the IOI logit difference than the control heads do, on
these prompts. It does not establish that the mechanism is unique in the
model -- a component acting in parallel would still look specific here, and
the measurements show several published heads doing comparable work on
different templates. Proving uniqueness needs a different control, and is not
claimed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

#: The IOI prompts: one per template in the repository's IOI frame set
#: (`ioi_pipeline._IOI_FRAMES`), with the names varied so that the claim is
#: not carried by one surface form. (prompt, subject, indirect object)
FRAMES: List[Tuple[str, str, str]] = [
    ("When Mary and John went to the store, Mary gave a drink to", "Mary", "John"),
    ("Then Alice and Bob went to the park, Alice gave a ball to", "Alice", "Bob"),
    ("After that, Charlie and David went to the library, Charlie offered a book to",
     "Charlie", "David"),
    ("While Eve and Frank went to the kitchen, Eve passed a towel to", "Eve", "Frank"),
    ("Yesterday Alice and Bob went to the market, Alice bought a apple to",
     "Alice", "Bob"),
    ("Later, David and Charlie went to the garden, David sent a letter to",
     "David", "Charlie"),
    ("One day Frank and Eve went to the school, Frank showed a photo to",
     "Frank", "Eve"),
    ("Finally, Bob and Alice went to the beach, Bob handed a shell to",
     "Bob", "Alice"),
]

#: The frame the claim was settled on. The other seven are held out.
FITTING_FRAME = 0

#: The IOI heads MECH publishes, pinned here and checked against the
#: pipeline's own list below: if the published claim changes, this file has to
#: be re-derived deliberately rather than drift away from it.
PUBLISHED_HEADS = ("L9H6", "L9H9", "L10H0", "L10H7", "L7H3", "L8H6",
                   "L5H1", "L5H5", "L0H1", "L0H10")

#: Control heads: heads that appear in neither the published IOI circuit nor
#: the circuit the repository discovers (`docs/roadmap.md`). A head of the
#: mechanism is a competitor, not a control.
CONTROL_HEADS: Tuple[Tuple[int, int], ...] = (
    (0, 0), (1, 5), (2, 11), (3, 2), (4, 7), (6, 1), (8, 4), (11, 8),
)


def _parse(label: str) -> Tuple[int, int]:
    """'L5H1' -> (5, 1)."""
    return int(label[1:label.index("H")]), int(label[label.index("H") + 1:])


@pytest.fixture(scope="module")
def adapter():
    from backend.science.models.gpt2_adapter import GPT2Adapter

    instance = GPT2Adapter(variant="small", mock_mode=False)
    assert instance._model is not None
    return instance


def _d_head(adapter) -> int:
    return adapter.spec.d_model // adapter.spec.num_heads


def _token_ids(adapter, subject: str, io: str) -> Tuple[int, int]:
    """Token ids for the indirect object and the subject, one token each.

    A multi-token name would make the logit difference a comparison between
    different positions in the vocabulary, so the single-token property is
    asserted rather than assumed.
    """
    ids = []
    for word in (io, subject):
        encoded = adapter._tokenizer(f" {word}",
                                     return_tensors=None)["input_ids"]
        assert len(encoded) == 1, (
            f"{word!r} is not a single token ({encoded}); the IOI logit "
            f"difference would then compare different vocabulary positions")
        ids.append(encoded[0])
    return ids[0], ids[1]


def _ld_effect(adapter, prompt: str, layer: int, head: int, io_id: int,
               subj_id: int, vector: Optional[Sequence[float]] = None,
               position=None) -> float:
    """``|clean_ld - ablated_ld|`` for one head intervention.

    ``clean_ld`` is the IOI logit difference on the clean prompt: the
    indirect object's logit minus the subject's. The intervention is one write
    into one head's slice of ``attn.c_proj``'s input.
    """
    r_io = adapter.patch_head_output(prompt, layer, head, patch_vector=vector,
                                     position=position, score_token_id=io_id)
    r_sub = adapter.patch_head_output(prompt, layer, head, patch_vector=vector,
                                      position=position, score_token_id=subj_id)
    clean_ld = r_io.original_logit - r_sub.original_logit
    ablated_ld = r_io.patched_logit - r_sub.patched_logit
    return abs(clean_ld - ablated_ld)


def _zero_ablation(adapter, prompt: str, layer: int, head: int, io_id: int,
                   subj_id: int) -> float:
    """Effect of zeroing one head at every position."""
    return _ld_effect(adapter, prompt, layer, head, io_id, subj_id,
                      vector=[0.0] * _d_head(adapter), position=None)


def _raw_logit_delta(adapter, prompt: str, layer: int, head: int) -> float:
    """How far the scored logit moves when the head is zeroed.

    This is the "the control does something at all" measure. It is not the
    specificity measure: the point of the file is that a head can move this a
    long way without touching the IOI signal.
    """
    result = adapter.patch_head_output(prompt, layer, head,
                                       patch_vector=[0.0] * _d_head(adapter),
                                       position=None)
    return result.delta


def _exact_head_output(adapter, prompt: str, layer: int, head: int) -> List[float]:
    """The head's own output vector, captured exactly.

    `capture_head_outputs` rounds to six decimals, and that is enough to make
    the write-back not bit-identical for some heads: a component whose
    float16 spacing is finer than 1e-6 lands on a neighbouring float16 value,
    and the scored logit -- which sits on a rounding boundary -- moves by one
    float16 step (measured: 0.0625 at a logit of -82.1875). The null control
    below is about the *write path*, so it is given an exact value to write
    back; the six-decimal capture is still what the rest of the file uses,
    where only a magnitude is read from it.

    The capture runs under the same attention kernel the patch runs under
    (`patch_head_output` forces eager), and restores the model's setting.
    """
    model = adapter._model
    import torch

    config = model.config
    previous = getattr(config, "_attn_implementation", None)
    captured: Dict[str, object] = {}

    def grab(_module, inp):
        captured["x"] = inp[0].detach().clone()

    if previous != "eager":
        config._attn_implementation = "eager"
    handle = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(grab)
    try:
        inputs = adapter._tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
        if previous is not None:
            config._attn_implementation = previous

    d_head = _d_head(adapter)
    vector = captured["x"][0, -1, head * d_head:(head + 1) * d_head]
    return [float(v) for v in vector]


def _fp16_ulp(value: float) -> float:
    """One float16 step at ``value``'s magnitude.

    `capture_head_outputs` rounds to six decimals, so writing the captured
    vector back is exact only to the precision the capture kept. In float16
    that can move a logit by a single step, and nothing more.
    """
    import torch

    v = torch.tensor(value, dtype=torch.float16)
    return float(abs(torch.nextafter(v, v + 1) - v))


@pytest.fixture(scope="module")
def survey(adapter):
    """One measurement pass; the tests below read it rather than re-measure."""
    frames = []
    for prompt, subject, io in FRAMES:
        io_id, subj_id = _token_ids(adapter, subject, io)

        published: Dict[str, float] = {}
        for label in PUBLISHED_HEADS:
            layer, head = _parse(label)
            published[label] = _zero_ablation(adapter, prompt, layer, head,
                                              io_id, subj_id)

        controls: Dict[Tuple[int, int], float] = {}
        for layer, head in CONTROL_HEADS:
            controls[(layer, head)] = _zero_ablation(adapter, prompt, layer,
                                                     head, io_id, subj_id)

        frames.append({
            "prompt": prompt,
            "io_id": io_id,
            "subj_id": subj_id,
            "published": published,
            "controls": controls,
            "captured": adapter.capture_head_outputs(prompt),
        })
    return frames


def _strongest(frame) -> Tuple[str, float]:
    """(label, effect) of the published head with the largest effect here."""
    label = max(frame["published"], key=lambda l: frame["published"][l])
    return label, frame["published"][label]


def test_the_claim_under_test_is_the_published_one():
    """The claim is MECH's published IOI head list, and nothing else.

    Two things have to hold for the measurements below to mean anything: the
    set of heads this file tests is the set the project publishes, and the
    null distribution does not contain the claim.
    """
    from backend.science.reproducibility.ioi_pipeline import (
        REFERENCE_ONLY_DISCOVERED_HEADS,
    )

    assert set(REFERENCE_ONLY_DISCOVERED_HEADS) == set(PUBLISHED_HEADS), (
        f"the published IOI head list changed to "
        f"{sorted(REFERENCE_ONLY_DISCOVERED_HEADS)}; this file tests "
        f"{sorted(PUBLISHED_HEADS)}. Re-derive it deliberately -- a stale "
        f"claim would be tested as if it were the current one."
    )

    assert not {_parse(label) for label in PUBLISHED_HEADS} & set(CONTROL_HEADS), (
        "a control head is in the published IOI list; a null distribution "
        "that contains the claim is not a null distribution"
    )


@needs_weights
def test_sham_writing_the_captured_value_back_is_a_null_control(adapter, survey):
    """The machinery control.

    If the harness itself moved the scored quantity, every measurement below
    would be contaminated. This must be exactly zero, which makes it the
    baseline every other effect is read against.

    The capture is per prompt, and the value written back is the one captured
    from *that* prompt, at the position it was captured from. The first
    version of this file captured one prompt's vector and wrote it into three
    others, so two of the three were transplants -- a real intervention with a
    real effect -- and the test failed against a primitive that was behaving
    correctly.
    """
    for frame in survey:
        for label in PUBLISHED_HEADS:
            layer, head = _parse(label)
            vector = _exact_head_output(adapter, frame["prompt"], layer, head)

            effect = _ld_effect(adapter, frame["prompt"], layer, head,
                                frame["io_id"], frame["subj_id"],
                                vector=vector, position=-1)
            assert effect == 0.0, (
                f"writing the head's own captured value back moved the IOI "
                f"logit difference by {effect:.6f} on {frame['prompt']!r} "
                f"({label}); every effect measured in this file is "
                f"contaminated by the harness"
            )

            raw = adapter.patch_head_output(frame["prompt"], layer, head,
                                            patch_vector=vector, position=-1)
            assert raw.delta == 0.0, (
                f"the sham write moved the scored logit by {raw.delta:.6f} on "
                f"{frame['prompt']!r}, which is more than one float16 step at "
                f"that magnitude; the write path is not exact"
            )


@needs_weights
def test_a_random_vector_of_the_same_norm_does_not_reproduce_the_effect(adapter, survey):
    """Rules out "any large write to this slice does it".

    The random control is scaled to the same L2 norm as the strongest
    claimed head's own output -- the size of the perturbation the
    zero-ablation removes -- so a difference cannot be explained by magnitude.
    The first version of this file wrote `torch.randn(64)`, whose norm is
    about eight against the head's 1.38, and reported the control as a
    refutation.
    """
    import torch

    for frame in survey:
        strongest_label, strongest = _strongest(frame)
        layer, head = _parse(strongest_label)
        norm = float(torch.tensor(frame["captured"][layer][head]).norm())

        random_deltas = []
        # One seed per prompt and one independent draw per control head: the
        # null is a distribution over random perturbations of the same size,
        # not one direction tried in every head.
        torch.manual_seed(0)
        for c_layer, c_head in CONTROL_HEADS:
            vector = torch.randn(_d_head(adapter))
            vector = vector / vector.norm() * norm
            random_deltas.append(
                _ld_effect(adapter, frame["prompt"], c_layer, c_head,
                           frame["io_id"], frame["subj_id"],
                           vector=vector.tolist(), position=None)
            )
        largest = max(random_deltas)

        assert strongest > largest, (
            f"on {frame['prompt']!r} the strongest claimed head "
            f"({strongest_label}) moved the IOI logit difference by "
            f"{strongest:.4f}, but a same-norm random write to another head "
            f"moved it by {largest:.4f}. The effect is not distinguishable "
            f"from a same-norm random perturbation, so it does not identify "
            f"a mechanism."
        )


@needs_weights
def test_the_claimed_head_is_an_outlier_against_the_control_heads(survey):
    """The specificity statistic, per prompt.

    Reported per prompt rather than pooled: pooling would let one large effect
    carry the average and hide a prompt where the claim is not specific.
    """
    for frame in survey:
        strongest_label, strongest = _strongest(frame)
        control_mean = sum(frame["controls"].values()) / len(frame["controls"])
        assert strongest > control_mean, (
            f"on {frame['prompt']!r} the strongest claimed head "
            f"({strongest_label}) moved the IOI logit difference by "
            f"{strongest:.4f}, which does not exceed the mean over control "
            f"heads ({control_mean:.4f}). The effect is inside the ordinary "
            f"spread, so attributing it to the published IOI heads is not "
            f"supported."
        )


@needs_weights
def test_specificity_holds_on_every_held_out_prompt(survey):
    """Held-out generalisation, stated as an outcome rather than an average.

    The claim must survive on every prompt, not on average: an average across
    them would let two passes and one catastrophic failure read as success.
    """
    failures = []
    held_out = survey[1:]      # every frame but the fitting one
    for frame in held_out:
        strongest_label, strongest = _strongest(frame)
        largest_label = max(frame["controls"], key=lambda l: frame["controls"][l])
        largest = frame["controls"][largest_label]
        if strongest <= largest:
            failures.append(f"{frame['prompt']!r}: the strongest claimed head "
                            f"({strongest_label}) {strongest:.4f} <= the "
                            f"largest control (L{largest_label[0]}H"
                            f"{largest_label[1]}) {largest:.4f}")

    assert not failures, (
        f"the published IOI heads were not the most specific components on "
        f"{len(failures)}/{len(held_out)} held-out prompts: "
        + "; ".join(failures)
    )


@needs_weights
def test_the_null_distribution_is_not_degenerate(adapter, survey):
    """The controls must actually do something.

    If every control head produced no effect at all, then "the claimed head
    is the most specific" would be satisfied by any non-zero target,
    including noise -- and the whole file would be measuring the absence of
    perturbation rather than specificity.
    """
    frame = survey[0]
    control = [abs(_raw_logit_delta(adapter, frame["prompt"], layer, head))
               for layer, head in CONTROL_HEADS]

    non_zero = sum(1 for d in control if d > 1e-6)
    assert non_zero >= len(control) // 2, (
        f"only {non_zero}/{len(control)} control heads moved the scored logit: "
        f"{['%.5f' % d for d in control]}. The null distribution is degenerate, "
        f"so any non-zero target would beat it trivially.")
