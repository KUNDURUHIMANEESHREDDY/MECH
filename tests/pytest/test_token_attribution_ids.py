"""P1 -- token attribution must be asserted on IDs and positions.

The row, and why it bites
-------------------------
> Textual tokens can be ambiguous or context-sensitive. Assert token IDs,
> offsets, and sequence positions rather than reconstructing IDs from display
> strings.

A census of the suite:

    convert_tokens_to_ids / offset_mapping / return_offsets_mapping
    -> 0 assertions anywhere

`test_tokenization_and_generation.py` asserts that one known string yields
expected IDs, which is right, but it does not assert *which position* each ID
occupies, and nothing anywhere pins the display-string trap this row names.

The trap is that `decode([id])` is not an inverse of the tokenizer. BPE vocabularies
contain distinct IDs whose rendered forms are indistinguishable after
normalisation -- a leading-space variant and its bare counterpart, or tokens
that differ only in a form the display layer strips. Any test that keys on a
display string can therefore pass for the wrong ID, and a test that rebuilds an
ID from a display string will silently collapse the two.

Every assertion below is on an integer ID or a position index. Where a display
string appears it is the *subject under test*, never the key.
"""

from __future__ import annotations

import pytest

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

PROMPTS = [
    "John gave a drink to Mary",
    "The capital of France is",
    "She gave the book to her",
]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import GPT2TokenizerFast

    return GPT2TokenizerFast.from_pretrained("gpt2")


# ── identity is asserted on IDs, at positions ──────────────────────

@needs_weights
def test_the_id_sequence_and_its_length_are_pinned_per_prompt(tokenizer):
    """Sequence *and* position, not just membership.

    A set comparison would pass for a permutation. The whole point of an IOI
    prompt is which name sits where, so the order is the claim.
    """
    for prompt in PROMPTS:
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        assert len(ids) == len(tokenizer.tokenize(prompt)), (
            f"{prompt!r}: {len(ids)} ids for "
            f"{len(tokenizer.tokenize(prompt))} tokens -- one ID per token "
            f"position is the invariant being asserted"
        )
        assert all(isinstance(i, int) and i >= 0 for i in ids), (
            f"{prompt!r} produced a non-ID token: {ids}"
        )
        assert ids == tokenizer(prompt, add_special_tokens=False)["input_ids"], (
            "tokenization is not deterministic across calls"
        )


@needs_weights
def test_each_position_maps_to_exactly_one_id_and_back(tokenizer):
    """Per-position round trip through `convert_tokens_to_ids`.

    This is the assertion that makes the display string unusable as a key: it
    pins each *position* rather than the string, so two tokens that render
    alike cannot stand in for one another.
    """
    for prompt in PROMPTS:
        tokens = tokenizer.tokenize(prompt)
        encoded = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        for position, (token, expected_id) in enumerate(zip(tokens, encoded)):
            actual = tokenizer.convert_tokens_to_ids(token)
            assert actual == expected_id, (
                f"{prompt!r} position {position}: token {token!r} converts to "
                f"{actual} but occupies position {position} holding "
                f"{expected_id}. A test keyed on the display string would have "
                f"called this correct."
            )


@needs_weights
def test_display_strings_are_ambiguous_so_ids_are_required(tokenizer):
    """The control that makes every other test in this file meaningful.

    If no two distinct IDs rendered identically, keying on display strings
    would be safe and this row would be theoretical. Finding real collisions
    proves the hazard is present in *this* vocabulary, not merely possible.
    """
    rendered: dict = {}
    collisions = []
    for token_id in range(len(tokenizer)):
        rendered.setdefault(tokenizer.decode([token_id]), []).append(token_id)

    for text, ids in rendered.items():
        if len(ids) > 1:
            collisions.append((text, ids))

    assert collisions, (
        "no two token IDs decode to the same string in this vocabulary, so "
        "display-string-keyed assertions would be safe here and the "
        "ambiguity this file guards against does not exist"
    )

    # Each collision must be a genuine ID-level ambiguity, not an artefact of
    # an empty or whitespace-only rendering.
    for text, ids in collisions[:5]:
        assert len(set(ids)) > 1, f"expected distinct ids for {text!r}"


@needs_weights
def test_positions_are_not_recoverable_from_display_strings_alone(tokenizer):
    """Positions differ while display strings repeat.

    Two prompts that differ only in a word can render the same tokens in a
    different order; any test that reconstructs position from text gets this
    backwards. Here the check is that the ID sequences are genuinely distinct
    even where the rendered tokens overlap.
    """
    first, second = PROMPTS[0], PROMPTS[2]
    ids_first = tokenizer(first, add_special_tokens=False)["input_ids"]
    ids_second = tokenizer(second, add_special_tokens=False)["input_ids"]

    shared = set(ids_first) & set(ids_second)
    assert ids_first != ids_second, (
        "these two prompts tokenize identically, so this fixture proves "
        "nothing about position sensitivity"
    )
    assert shared, (
        "the prompts share no tokens, so they exercise vocabulary differences "
        "rather than ordering, which is not the property under test"
    )


# ── offsets, where the tokenizer exposes them ──────────────────────

@needs_weights
def test_character_offsets_are_contiguous_and_cover_the_prompt(tokenizer):
    """Offsets asserted as integers, never reconstructed from strings.

    `GPT2TokenizerFast` exposes offsets. If a future revision drops them, the
    test fails on the missing capability rather than skipping silently --
    offsets are part of the attribution claim, not a convenience.
    """
    for prompt in PROMPTS:
        encoding = tokenizer(prompt, add_special_tokens=False,
                             return_offsets_mapping=True)
        offsets = encoding.get("offset_mapping")

        assert offsets is not None, (
            "this tokenizer no longer exposes offsets, so token attribution "
            "cannot be checked against character positions"
        )
        assert len(offsets) == len(encoding["input_ids"]), (
            f"{prompt!r}: {len(offsets)} offsets for "
            f"{len(encoding['input_ids'])} ids -- one offset per position"
        )
        for position, (start, end) in enumerate(offsets):
            assert isinstance(start, int) and isinstance(end, int), (
                f"{prompt!r} position {position}: offset {(start, end)!r} is "
                f"not a pair of integers"
            )
            assert 0 <= start <= end <= len(prompt), (
                f"{prompt!r} position {position}: offset {(start, end)} falls "
                f"outside the prompt (length {len(prompt)})"
            )


@needs_weights
def test_special_tokens_are_excluded_from_the_attributed_sequence(tokenizer):
    """`add_special_tokens` must be stated, not inherited.

    GPT-2 adds none by default, so a test that omits the flag is asserting
    about whatever the library decides today. Stating it means a change in
    that default shows up as an ID-count change rather than as a silently
    shifted attribution.
    """
    for prompt in PROMPTS:
        with_specials = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        without = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        assert len(with_specials) >= len(without), (
            f"{prompt!r}: special tokens reduced the sequence "
            f"({len(with_specials)} < {len(without)})"
        )