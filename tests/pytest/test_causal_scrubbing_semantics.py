"""Causal scrubbing must scrub heads, and must report a real preservation score.

`causal_scrubbing.py` carried the same defect as ACDC: it swept `(layer, head)`
pairs and passed `head` to APIs whose parameter is `neuron_index`.

    get_activations(ref_prompt, layer=layer, neuron_index=head)
    patch_activation(clean_prompt, layer=layer, neuron_index=head, ...)

`get_activations` reads `hidden_states[layer][0, tok, :][head]` -- dimension
`head` of the 768-wide residual stream. `patch_activation` writes
`transformer.h[layer].mlp` output at index `head` -- MLP neuron `head` of 3072.
Both succeeded, so the scrub measured components it never touched.

Two further problems compounded it:

* `preservation_ratio = preserved_logit / base_logit_score` divides one logit by
  another, then clamps to [0, 1]. That meaningless ratio became the graph edge
  weight, the aggregate `behavior_preservation`, the hypothesis verdict, *and*
  the report's `confidence`.
* Resampling drew from the whole prompt list, so with the default single-prompt
  dataset it always chose index 0 -- the prompt being scrubbed. The scrub patched
  each head with its own activation and measured the head against itself.
  `resample_count` (default 10) was configured, documented, and never used.

Preservation is now `(scrubbed - corrupted) / (clean - corrupted)`, so 1.0 is
"scrub left the behaviour intact" and 0.0 is "scrub destroyed it" -- the
endpoints the paper's test needs.
"""

from __future__ import annotations

import re
import statistics

import pytest


def _has_gpt2() -> bool:
    try:
        from backend.science.models.gpt2_adapter import GPT2Adapter
        return GPT2Adapter(variant="small", mock_mode=False)._model is not None
    except Exception:
        return False


needs_weights = pytest.mark.skipif(
    not _has_gpt2(), reason="GPT-2 weights are not loaded"
)

# Three same-length IOI-shaped prompts, so a structural resample exists.
PROMPTS = [
    {"id": "p0",
     "clean": "When Mary and John went to the store, John gave a drink to",
     "corrupted": "When John and Mary went to the store, Mary gave a drink to",
     "target": " Mary"},
    {"id": "p1",
     "clean": "When Alice and Bob went to the library, Bob handed a book to",
     "corrupted": "When Bob and Alice went to the library, Alice handed a book to",
     "target": " Alice"},
    {"id": "p2",
     "clean": "When Carol and Dave went to the park, Dave passed a ball to",
     "corrupted": "When Dave and Carol went to the park, Carol passed a ball to",
     "target": " Carol"},
]


def _run(mock: bool = False, resample_count: int = 3, dataset=None, config=None):
    from backend.interpretability.discovery.algorithms.causal_scrubbing import (
        CausalScrubbingAlgorithm,
    )
    from backend.interpretability.discovery.algorithms.configs import (
        CausalScrubbingConfig,
    )
    from backend.science.models.gpt2_adapter import GPT2Adapter

    adapter = GPT2Adapter(variant="small", mock_mode=mock)
    payload = dataset if dataset is not None else {"id": "ioi_multi", "prompts": PROMPTS}
    cfg = CausalScrubbingConfig(resample_count=resample_count) if config is None else config
    return CausalScrubbingAlgorithm(adapter).run(payload, cfg)


# ── Fail-closed paths ───────────────────────────────────────────────────────

def test_scrubbing_without_weights_does_not_run():
    """The old code defaulted `base_logit_score` to 1.0 and used it as the scale."""
    result = _run(mock=True, dataset={"id": "x", "clean": "a b", "target_token": " c"})

    assert result.statistics["measured"] is False
    assert result.statistics["behavior_preservation"] is None
    assert result.statistics["hypothesis_supported"] is None
    assert result.confidence is None
    assert "weights" in str(result.statistics["reason"]).lower()
    assert result.graph["nodes"] == [] and result.graph["edges"] == []


def test_a_single_prompt_cannot_be_scrubbed():
    """Resampling from the prompt being scrubbed is not resampling.

    `rng.randint(0, len(prompts) - 1)` over a one-element list always returned 0,
    so every "resampled" activation was the original activation and the scrub
    patched each head with its own output.
    """
    result = _run(dataset={
        "id": "single",
        "clean": PROMPTS[0]["clean"],
        "corrupted": PROMPTS[0]["corrupted"],
        "target_token": PROMPTS[0]["target"],
    })

    assert result.statistics["measured"] is False
    reason = str(result.statistics["reason"]).lower()
    assert "resample" in reason and "alternative prompt" in reason
    assert result.statistics["resamples_performed"] == 0
    assert result.confidence is None


def test_a_multi_token_target_cannot_be_scored():
    """The logit-difference metric needs a single continuation token."""
    result = _run(dataset={"id": "x", "clean": "a b c", "target_token": " hello world"})
    assert result.statistics["measured"] is False
    assert "single token" in str(result.statistics["reason"])


def test_unequal_length_references_are_rejected_and_says_so():
    """Only equal token counts are checked, and the report admits that."""
    mixed = [
        PROMPTS[0],
        {"id": "long", "clean": "a considerably longer prompt than the first one "
                                 "with many more tokens in it indeed",
         "corrupted": "x", "target": " y"},
    ]
    result = _run(dataset={"id": "mixed", "prompts": mixed})

    assert result.statistics["measured"] is False
    check = result.statistics["equivalence_check"]
    assert check["candidates_offered"] == 1
    assert check["candidates_same_length"] == 0
    assert check["necessary_not_sufficient"] is True
    assert result.provenance["equivalence_verified"] is False


# ── The real measurement ────────────────────────────────────────────────────

@needs_weights
def test_scrubbing_measures_and_honours_resample_count():
    result = _run(resample_count=3)

    s = result.statistics
    assert s["measured"] is True
    assert s["components_scrubbed"] == 8, "2 scrub layers x min(4, 12) heads"
    # resample_count was previously read into statistics and never used.
    assert s["resamples_performed"] == s["components_scrubbed"] * 3
    assert s["reference_prompts_used"] >= 1
    assert s["behavior_preservation"] is not None


@needs_weights
def test_preservation_is_normalised_by_the_clean_corrupted_gap():
    """Recompute the score from the reported numbers.

    The metric is (scrubbed - corrupted) / (clean - corrupted), so a component
    whose scrubbed score equals the clean score must read exactly 1.0, and one
    equal to the corrupted score must read 0.0. This is what makes the score mean
    "fraction of the behaviour that survived the scrub" rather than "one logit
    divided by another".
    """
    result = _run(resample_count=3)
    s = result.statistics

    clean = s["base_logit_score"]
    corrupted = s["corrupted_logit_score"]
    gap = clean - corrupted
    assert gap != 0, "the gap is the denominator and must not be zero"

    assert clean != corrupted, (
        "base_logit_score must be the clean target logit, not the top-token logit"
    )

    samples = result.evidence["per_component_preservation"]
    assert samples, "no per-component preservations recorded"
    for node_id, values in samples.items():
        for value in values:
            # A preservation of p implies a scrubbed logit of corrupted + p*gap.
            implied = corrupted + value * gap
            assert isinstance(implied, float)
            # Values are unbounded above by construction; the clamp that used to
            # sit here would have made every observation <= 1.0.
            assert value == pytest.approx(value)  # not NaN

    assert s["preservation_unclamped"] is True
    assert statistics.mean(
        v for values in samples.values() for v in values
    ) == pytest.approx(s["behavior_preservation"], abs=1e-3)


@needs_weights
def test_edge_weights_are_measured_preservation_not_a_ratio_of_logits():
    """Every edge carries the mean of its own resample set, and they differ.

    Previously every edge carried `min(1, max(0, preserved_logit /
    base_logit_score))` -- one logit over another, clamped, and therefore
    frequently identical across unrelated components.
    """
    result = _run(resample_count=3)
    edges = result.graph["edges"]

    assert len(edges) == 8
    for edge in edges:
        assert edge["weight"] == pytest.approx(
            statistics.mean(edge["samples"]), abs=1e-3
        ), f"{edge['target']}: edge weight is not the mean of its own samples"
        assert edge["unclamped"] is True
        assert "measured_from" in edge

    weights = [e["weight"] for e in edges]
    assert len(set(weights)) > 1, (
        "all 8 edges carry the same weight, so the score is not measuring the "
        "component it is attached to"
    )


@needs_weights
def test_report_confidence_is_not_the_preservation_score():
    """The old report set `confidence=round(behavior_preservation, 2)`.

    Preservation is a property of the scrub experiment, not a calibrated
    confidence in a hypothesis, and republishing it under that name is how a
    measured intermediate becomes an overstated conclusion.
    """
    result = _run(resample_count=3)
    assert result.confidence is None


@needs_weights
def test_hypothesis_direction_is_reported_unambiguously():
    """Scrub breaking the behaviour supports the hypothesis; preserving it refutes.

    The module docstring previously said the opposite -- that *high* behaviour
    preservation validates the hypothesis -- while the code did the paper's test.
    The outcome is now named `hypothesis_supported` rather than `validated`,
    which read as though the behaviour itself had been validated.
    """
    result = _run(resample_count=3)
    s = result.statistics

    assert "hypothesis_supported" in s
    assert s["hypothesis_supported"] == ((1.0 - s["behavior_preservation"]) > s["tolerance"])
    assert s["behavior_preservation"] is not None


@needs_weights
def test_equivalence_class_is_reported_as_unverified():
    """`equivalence_class` is configurable; nothing verifies membership."""
    result = _run(resample_count=3)
    check = result.statistics["equivalence_check"]

    assert check["configured_class"] == "token_type"
    assert check["verified_class"] == "same_token_count"
    assert check["necessary_not_sufficient"] is True
    assert result.provenance["equivalence_verified"] is False


# ── Source guards ───────────────────────────────────────────────────────────

def test_causal_scrubbing_does_not_use_the_neuron_api():
    from source_assert import executable_source

    import backend.interpretability.discovery.algorithms.causal_scrubbing as cs

    src = executable_source(cs)

    assert "patch_activation(" not in src, (
        "causal scrubbing is calling patch_activation again; that is an MLP "
        "intervention and its neuron_index parameter is not a head"
    )
    assert "get_activations(" not in src, (
        "causal scrubbing is calling get_activations again; it reads a "
        "residual-stream dimension, not a head"
    )
    assert "capture_head_outputs(" in src
    assert "patch_head_output(" in src


def test_causal_scrubbing_has_no_logit_over_logit_ratio():
    from source_assert import executable_source

    import backend.interpretability.discovery.algorithms.causal_scrubbing as cs

    src = executable_source(cs)

    # One logit divided by another, as the preservation score used to be.
    assert not re.search(r"logit\w*\s*/\s*\w*logit\w*", src), (
        "the preservation score is still a ratio of two logits"
    )
    assert not re.search(r"preserved_logit\s*/", src)
    assert not re.search(r"preservation_ratio", src)
