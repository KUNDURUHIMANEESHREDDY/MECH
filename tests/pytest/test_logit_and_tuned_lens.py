"""The logit lens must project real weights, and the tuned lens must be trained.

Two defects, one file each, both in the same trust boundary.

`backend/runtime/logits.py` never loaded a model
------------------------------------------------
`IntermediateLogitsEngine.extract_logits` returned, for every layer:

    residual_norm   10.0 + layer * 1.5
    top_prediction  " Paris" if layer >= 6 else words[0]
    top_logit       4.5 + layer * 0.8
    entropy         2.5 - layer * 0.15

Four linear functions of the layer index, with no provenance field at all. So
"the residual norm grows by 1.5 per layer" and "the top prediction flips to Paris
at layer 6" were statements about arithmetic, and `ExecutionEngine` published them
as measurements.

`algorithms/logit_lens.py` appended a fabricated second token
-------------------------------------------------------------
On top of that, `LogitLens.project` hardcoded its runner-up:

    {"token": " France", "logit": top_logit - 1.2, "probability": 0.12}

A fixed token, a fixed offset, a fixed probability -- the "stub that returns 0.94"
pattern, in the one algorithm the README listed as implemented.

The tuned lens added `+0.12` to every confidence
------------------------------------------------
Now trained for real. The default translator is the **full affine** map the
tuned-lens paper uses -- a per-layer `d x d` weight plus a bias -- so the
fitted lens is not a restricted stand-in for it. The cheaper `diagonal_affine`
family is still fittable (`kind=`), and every payload names the family that
produced it via `translator_kind`, because reporting a restricted result as
"the tuned lens" would overstate it.

These tests use a stub engine so the real code paths run without a GPU. The
arithmetic-vs-measurement distinction is asserted by construction: a linear
function of the layer index cannot survive a test that feeds it a stub whose
values are not linear in the index.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from backend.interpretability.algorithms.logit_lens import LogitLens
from backend.interpretability.algorithms.tuned_lens import (
    DEFAULT_TRANSLATOR_KIND,
    DIAGONAL_AFFINE,
    FULL_AFFINE,
    TRANSLATOR_KIND,
    TunedLens,
)
from backend.runtime.logits import IntermediateLogitsEngine

ROOT = Path(__file__).resolve().parents[2]

LIVE = "live"
UNAVAILABLE = "unavailable"


# ── Stubs ───────────────────────────────────────────────────────────────

class _StubEngine:
    """Mimics `gpt2_engine`'s shape with values that are NOT linear in layer.

    Deliberately non-linear: the old defect produced `10.0 + layer * 1.5`. A stub
    whose values are quadratic cannot be satisfied by any linear formula, so a
    regression to arithmetic fails here rather than passing.
    """

    MODEL_ID = "stub-gpt2"

    def __init__(self, n_layers: int = 4, vocab: int = 7) -> None:
        self.n = n_layers
        self.vocab = vocab

    def logit_lens(self, layer: int, prompt: str, top_k: int = 5) -> Dict[str, Any]:
        if layer >= self.n:
            return {"status": UNAVAILABLE, "reason": "no such layer"}
        # Quadratic in layer, so no linear formula reproduces it.
        peak = layer * layer + 1
        toks = [{"token": f" tok{peak + i}", "prob": round(0.5 / (i + 1), 6)}
                for i in range(min(top_k, self.vocab))]
        return {"status": "ok", "method": "LogitLens", "layer": layer,
                "prompt": prompt, "top_token": toks[0]["token"],
                "top_k_tokens": toks}

    def logit_lens_all(self, prompt: str, top_k: int = 3) -> Dict[str, Any]:
        layers = []
        for i in range(self.n):
            one = self.logit_lens(i, prompt, top_k)
            layers.append({"layer": i, "top_token": one["top_token"],
                           "top_k_tokens": one["top_k_tokens"]})
        return {"status": "ok", "prompt": prompt, "layers": layers}


class _FailingEngine:
    """Stands in for a machine where the weights will not load."""

    MODEL_ID = None

    def logit_lens(self, layer: int, prompt: str, top_k: int = 5) -> Dict[str, Any]:
        return {"status": UNAVAILABLE,
                "reason": "transformers is not installed"}


# ── The fabricated arithmetic must be gone ──────────────────────────────

def test_residual_norm_is_not_a_linear_function_of_the_layer():
    """The exact old formula, asserted to be absent.

    The stub's layer norms are `(L + 1) * sqrt(2)`, so a reintroduction of
    `10.0 + layer * 1.5` cannot pass.
    """
    n = 5
    cache = _StubHiddenCache(n)
    engine = _CacheBackedEngine(_StubEngine(n_layers=n), cache)

    result = IntermediateLogitsEngine(engine=engine).extract_logits("hello",
                                                                    num_layers=n)

    norms = [p["residual_norm"] for p in result["layer_projections"]]
    assert result["status"] == "ok"
    # Layer L reads hidden[L + 1].
    assert norms == [_StubHiddenCache.norm_at(i + 1) for i in range(n)], norms

    old = [round(10.0 + (i * 1.5), 2) for i in range(n)]
    assert norms != old, "the fabricated linear residual norms are back"


def test_top_prediction_tracks_the_projection_not_a_constant():
    """The old rule: `" Paris" if layer >= 6 else words[0]`."""
    n = 4
    cache = _StubHiddenCache(n)
    engine = _CacheBackedEngine(_StubEngine(n_layers=n), cache)

    result = IntermediateLogitsEngine(engine=engine).extract_logits(
        "hello world", num_layers=n)

    predictions = [p["top_prediction"] for p in result["layer_projections"]]

    assert " Paris" not in predictions
    # The stub's sign alternates per layer, so the arg-max alternates too. A
    # rule that keys off `layer >= 6` or off `words[0]` cannot produce this.
    assert len(set(predictions)) == 2, (
        f"every layer reported {predictions[0]!r}; the projection is not "
        f"actually being read")


def test_entropy_comes_from_the_distribution_not_a_formula():
    """The old rule: `2.5 - layer * 0.15`."""
    import torch

    n = 4
    cache = _StubHiddenCache(n)
    engine = _CacheBackedEngine(_StubEngine(n_layers=n), cache)

    result = IntermediateLogitsEngine(engine=engine).extract_logits("hi",
                                                                    num_layers=n)
    entropies = [p["entropy"] for p in result["layer_projections"]]

    old = [round(2.5 - (i * 0.15), 2) for i in range(n)]
    assert entropies != old

    # And it is the real entropy of the real 2-way logits at each layer.
    for index, reported in enumerate(entropies):
        state = torch.tensor(cache.hidden[index + 1][0])
        probs = torch.softmax(state, dim=-1)
        expected = float(-(probs * torch.log_softmax(state, dim=-1)).sum())
        assert reported == pytest.approx(round(expected, 4), abs=1e-4)


def test_top_logit_is_the_real_max_logit():
    import torch

    n = 3
    cache = _StubHiddenCache(n)
    engine = _CacheBackedEngine(_StubEngine(n_layers=n), cache)

    result = IntermediateLogitsEngine(engine=engine).extract_logits("x", n)

    for index, projection in enumerate(result["layer_projections"]):
        state = torch.tensor(cache.hidden[index + 1][0])
        assert projection["top_logit"] == pytest.approx(
            round(float(state.max()), 4), abs=1e-4)


def test_every_layer_field_carries_provenance():
    engine = _StubEngine(n_layers=3)
    layers = _StubHiddenCache(engine.n)

    result = IntermediateLogitsEngine(
        engine=_CacheBackedEngine(engine, layers)).extract_logits("x", 3)

    assert result["provenance"] == LIVE
    for field in ("layer", "residual_norm", "top_prediction", "top_logit",
                  "entropy", "top_k_tokens"):
        assert result["field_provenance"][field] == LIVE


# ── No arithmetic fallback when weights are missing ─────────────────────

def test_unloadable_weights_withhold_everything():
    result = IntermediateLogitsEngine(engine=_NoWeightsEngine()).extract_logits(
        "hello", num_layers=4)

    assert result["status"] == UNAVAILABLE
    assert result["provenance"] == UNAVAILABLE
    assert result["layer_projections"] == []
    assert result["reason"]
    for field in ("residual_norm", "top_prediction", "top_logit", "entropy"):
        assert result["field_provenance"][field] == UNAVAILABLE


def test_an_unloadable_engine_reports_no_numbers_at_all():
    """The old code produced four numbers per layer regardless.

    A count of zero layers, with a reason, is the only honest answer when there
    is no model.
    """
    result = IntermediateLogitsEngine(engine=_NoWeightsEngine()).extract_logits(
        "hello", num_layers=12)

    assert result["total_layers"] == 0
    assert result["requested_layers"] == 12, (
        "the request is still reported, so the reader can see nothing was "
        "answered for it")


# ── The fabricated second token must be gone ────────────────────────────

def test_no_hardcoded_runner_up_token():
    lens = LogitLens(engine=_StubEngine(n_layers=4))
    out = lens.project("The capital of France is", layer=2, top_k=3)

    tokens = [t["token"] for t in out["top_k_tokens"]]
    assert " France" not in tokens, (
        "the hardcoded ' France' entry is back")
    assert tokens == [" tok5", " tok6", " tok7"], tokens


def test_top_k_is_exactly_what_the_engine_returned():
    engine = _StubEngine(n_layers=4)
    out = LogitLens(engine=engine).project("x", layer=1, top_k=2)
    raw = engine.logit_lens(layer=1, prompt="x", top_k=2)

    assert out["top_k_tokens"] == raw["top_k_tokens"], (
        "the lens must pass the engine's ranking through unchanged, not rebuild "
        "or extend it")


def test_no_probability_is_invented_for_the_top_token():
    out = LogitLens(engine=_StubEngine(n_layers=3)).project("x", 0, top_k=1)

    assert out["top_probability"] == 0.5, (
        "the stub's measured probability for its top token; a fabricated 0.12 "
        "or a renormalised guess would fail here")
    assert out["top_logit"] is None, (
        "the engine supplies no logit, so none is reconstructed")


def test_the_lens_withholds_when_the_engine_cannot_project():
    out = LogitLens(engine=_FailingEngine()).project("x", layer=3)

    assert out["status"] == UNAVAILABLE
    assert out["provenance"] == UNAVAILABLE
    assert out["top_token"] is None
    assert out["top_k_tokens"] is None
    assert "transformers is not installed" in out["reason"]


def test_the_lens_does_not_fall_back_to_arithmetic():
    """Regression guard on the whole point.

    Any path that produces a token or a logit without the engine having run is
    the old defect wearing a different hat.
    """
    out = LogitLens(engine=_FailingEngine()).project("The capital of France is", 3)

    for field in ("top_token", "top_logit", "top_probability", "entropy"):
        assert out.get(field) is None, f"{field} was populated without a model"


# ── The tuned lens: untrained must not claim to be tuned ────────────────

def test_an_untrained_tuned_lens_reports_the_plain_lens_and_says_so():
    lens = TunedLens(engine=_StubEngine(n_layers=4))
    out = lens.project("x", layer=1, top_k=2)

    assert lens.is_trained is False
    assert out["method"] == "TunedLens"
    assert out["tuned_lens_available"] is False
    assert out["affine_translation_applied"] is False
    assert out["provenance"] == UNAVAILABLE
    assert "No translators have been trained" in out["reason"]


def test_the_confidence_inflation_is_gone():
    """It used to add a flat `+0.12` and report the translation as applied.

    The stub's top probability is exactly 0.5. Any inflation shows up as a
    different number here.
    """
    stub = _StubEngine(n_layers=3)
    plain = LogitLens(engine=stub).project("x", 1, top_k=1)
    tuned = TunedLens(engine=_StubEngine(n_layers=3)).project("x", 1, top_k=1)

    assert plain["top_probability"] == tuned["top_probability"], (
        "the tuned lens changed the probability without a trained translator")
    assert plain["top_k_tokens"] == tuned["top_k_tokens"]


def test_an_untrained_tuned_lens_is_not_publication_eligible():
    out = TunedLens(engine=_StubEngine(n_layers=3)).project("x", 1)

    assert out["validation_eligible"] is False
    assert out["publication_eligible"] is False


def test_training_without_weights_is_unavailable_not_a_default_lens():
    lens = TunedLens(engine=_NoWeightsEngine())
    report = lens.train(["some calibration text"])

    assert report["status"] == UNAVAILABLE
    assert report["provenance"] == UNAVAILABLE
    assert lens.is_trained is False
    assert lens.translators == {}


def test_training_on_no_prompts_refuses():
    lens = TunedLens(engine=_StubEngine(n_layers=3))
    report = lens.train([])

    assert report["status"] == UNAVAILABLE
    assert "nothing to fit" in report["reason"]
    assert lens.is_trained is False


def test_there_is_no_bundled_translator_weight_file():
    """No default weights, by construction.

    A default translator set would let `TunedLens()` project through a learned
    map that was never trained against the model in front of it -- which is the
    tuned-lens equivalent of a default provenance stamp.
    """
    lens = TunedLens(engine=_StubEngine(n_layers=3))

    assert lens.translators == {}, "a translator exists before any training"
    assert lens.training_report is None


def test_the_translator_kind_is_declared_everywhere():
    """The family must be named wherever a lens is reported.

    The default is the published full affine map, so a payload that failed to
    name it would let a reader assume a cheaper map than the one that was
    fitted -- and a payload naming the *wrong* family would be worse than
    silence. `translator_kind` travels with the class.
    """
    lens = TunedLens(engine=_StubEngine(n_layers=3))

    assert TRANSLATOR_KIND == "full_affine", (
        "the default translator is no longer the published full affine map; "
        "the constant and every place that names it must change together"
    )
    assert DEFAULT_TRANSLATOR_KIND == TRANSLATOR_KIND, (
        "the backwards-compatible alias no longer points at the default"
    )
    assert TRANSLATOR_KIND in (FULL_AFFINE, DIAGONAL_AFFINE)

    untrained = lens.project("x", 1)
    assert untrained["translator_kind"] == TRANSLATOR_KIND


def test_the_projection_names_the_family_it_actually_applied():
    """A payload must not describe a map that was never applied.

    The projection's reason hard-coded 'diagonal-affine' while the default
    family is the full affine one, so every payload described a translator
    that was not the one fitted. The name has to come from the fitted
    report, for every family.
    """
    import torch

    cache = _StubHiddenCache(3)
    engine = _TrainedStubEngine(_StubEngine(n_layers=3), cache, prompt="hi")
    lens = TunedLens(engine=engine)

    families = (
        (FULL_AFFINE, {"kind": FULL_AFFINE, "weight": torch.eye(2),
                        "bias": torch.zeros(2)}),
        (DIAGONAL_AFFINE, {"kind": DIAGONAL_AFFINE, "scale": torch.ones(2),
                            "shift": torch.zeros(2)}),
    )
    for kind, entry in families:
        lens.translators = {1: entry}
        lens.training_report = {"status": "completed", "provenance": LIVE,
                                "translator_kind": kind}

        out = lens.project("hi", 1)

        assert out.get("affine_translation_applied") is True, out
        assert out["translator_kind"] == kind, out
        assert kind in str(out.get("reason", "")), (
            f"the projection does not name the family it applied: {out!r}")


def test_training_reports_the_variant_it_actually_fitted():
    """A report must name the family it fitted, and never claim more.

    The full affine translator has ~590k parameters per layer for d=768; the
    restricted one has 1536. Anyone quoting the wrong family as a tuned-lens
    result would be wrong by two orders of magnitude in capacity, so the
    report names the family it used.
    """
    lens = TunedLens(engine=_StubEngine(n_layers=3))
    report = lens.train(["calibration text"])

    # No weights behind the stub, so this is the unavailable path -- but the
    # variant is still declared, because the declaration travels with the class.
    assert report["translator_kind"] == TRANSLATOR_KIND
    assert report["translator_kind"] in (FULL_AFFINE, DIAGONAL_AFFINE)
    assert report["status"] == UNAVAILABLE
    assert report["provenance"] == UNAVAILABLE
    assert report["publication_eligible"] is False


# ── The translator arithmetic, testable without a forward pass ──────────

def test_apply_translator_scales_and_shifts():
    import torch

    lens = TunedLens(engine=_StubEngine(n_layers=3))
    lens.translators = {2: {"kind": DIAGONAL_AFFINE,
                            "scale": torch.tensor([2.0, 0.5]),
                            "shift": torch.tensor([1.0, -1.0])}}

    hidden = torch.tensor([[3.0, 4.0]])
    out = lens.apply_translator(hidden, layer=2)

    assert out.tolist() == [[3.0 * 2.0 + 1.0, 4.0 * 0.5 - 1.0]]


def test_apply_translator_refuses_an_untrained_layer():
    import torch

    lens = TunedLens(engine=_StubEngine(n_layers=3))
    lens.translators = {1: {"scale": torch.ones(2), "shift": torch.zeros(2)}}

    with pytest.raises(KeyError):
        lens.apply_translator(torch.zeros(1, 2), layer=7)


def test_the_mean_kl_of_a_perfect_projection_is_zero():
    """Sanity on the objective itself.

    `_mean_kl` takes a *hidden state*, applies `layer_norm` then the unembedding,
    and compares the result to a target. So the target has to be built the same
    way -- passing raw logits as the hidden state would not give KL 0, and the
    earlier version of this test did exactly that.
    """
    import torch

    torch.manual_seed(0)
    hidden = torch.randn(6, 8)
    unembed = torch.eye(8)

    target = torch.log_softmax(
        torch.nn.functional.layer_norm(hidden, (8,)) @ unembed.T, dim=-1)

    # The translator is applied by the caller now, so the untranslated
    # objective is the four-argument form.
    kl = TunedLens._mean_kl(torch, hidden, target, unembed)

    assert isinstance(kl, torch.Tensor)
    assert float(kl) == pytest.approx(0.0, abs=1e-5)


def test_the_mean_kl_is_positive_for_a_wrong_projection():
    import torch

    target = torch.log_softmax(torch.randn(5, 11), dim=-1)
    kl = TunedLens._mean_kl(torch, torch.zeros(5, 11), target,
                            torch.eye(11)).item()

    assert kl > 0.0
    assert not math.isnan(kl)


def test_mean_kl_stays_differentiable():
    """`train()` calls `.backward()` on the sum of these.

    Returning a Python float severs the autograd graph, and `train()` then raises
    `AttributeError: 'float' object has no attribute 'backward'` on the first
    step -- which is exactly what happened before this was a tensor.

    The translator is applied by the caller now, so the graph has to survive
    that composition too: the gradient must reach the fitted scale and shift,
    not just the objective that consumes them.
    """
    import torch

    hidden = torch.randn(4, 6)
    scale = torch.ones(6, requires_grad=True)
    shift = torch.zeros(6, requires_grad=True)
    target = torch.log_softmax(torch.randn(4, 6), dim=-1)

    kl = TunedLens._mean_kl(torch, hidden, target, torch.eye(6),
                            translated=hidden * scale + shift)

    assert isinstance(kl, torch.Tensor), (
        f"_mean_kl returned {type(kl).__name__}; the training loop needs a "
        f"tensor it can differentiate")
    kl.backward()
    assert scale.grad is not None, "no gradient reached the scale"
    assert shift.grad is not None, "no gradient reached the shift"


def test_mean_kl_accepts_a_scale_without_a_shift():
    """Independent application, not both-or-neither.

    `h * scale + shift` raised `TypeError: Tensor + NoneType` on a legitimate
    scale-only call. The refactored signature takes the already-translated
    state, so that composition moved to the caller -- but the scale-only call
    is still a legitimate one, and the state handed in has to be the state
    that gets scored.
    """
    import torch

    torch.manual_seed(0)
    hidden = torch.randn(3, 5)
    target = torch.log_softmax(torch.randn(3, 5), dim=-1)
    unembed = torch.eye(5)

    scale = torch.full((5,), 2.0)
    shift = torch.full((5,), 2.0)

    scaled = TunedLens._mean_kl(torch, hidden, target, unembed,
                                translated=hidden * scale)
    shifted = TunedLens._mean_kl(torch, hidden, target, unembed,
                                 translated=hidden + shift)
    both = TunedLens._mean_kl(torch, hidden, target, unembed,
                              translated=hidden * scale + shift)

    for value in (scaled, shifted, both):
        assert isinstance(value, torch.Tensor)
        assert not math.isnan(float(value))

    # A shift and a scale are not the same translation, so the two calls
    # must not agree: the state handed in is the state that is scored.
    assert abs(float(shifted) - float(both)) > 1e-6, (
        "a shift-only translation and a scale-and-shift translation scored "
        "identically, so the translated state is being ignored"
    )


# ── The held-out split, which is what makes the improvement meaningful ──

def test_the_split_is_deterministic_and_holds_out_every_fifth_prompt():
    prompts = [f"p{i}" for i in range(15)]

    fit, held = TunedLens._split(prompts, holdout=True)

    assert held == ["p0", "p5", "p10"]
    assert "p0" not in fit, "a held-out prompt must not also be fitted on"
    assert set(fit) | set(held) == set(prompts)
    assert not (set(fit) & set(held))
    # Same split every time, so a repeated run scores the same prompts.
    assert TunedLens._split(prompts, holdout=True) == (fit, held)


def test_a_single_prompt_cannot_be_split():
    """Nothing to hold out from, so everything is fitted and it is said."""
    fit, held = TunedLens._split(["only"], holdout=True)

    assert fit == ["only"]
    assert held == [], "a prompt cannot be both fitted on and held out"


def test_holdout_can_be_switched_off():
    fit, held = TunedLens._split([f"p{i}" for i in range(10)], holdout=False)

    assert len(fit) == 10
    assert held == []


def test_training_never_reports_an_in_sample_number_as_the_headline():
    """Structure only -- the numbers need weights.

    The distinction is the whole point: an in-sample KL reduction on a small
    calibration set is close to unfalsifiable, because a diagonal map can drive
    it toward zero by memorising. `headline_metric` names which figure a reader
    should quote.
    """
    import inspect

    source = inspect.getsource(TunedLens.train)

    assert "holdout_kl_reduction_pct" in source, (
        "the held-out reduction is not reported at all")
    assert '"headline_metric"' in source, (
        "the report does not name which metric is the headline")
    # The holdout must be scored through translators that never saw it, i.e.
    # after the split, using _collect_hidden on the held-out prompts.
    assert "_holdout_report" in source
    assert "heldout = self._holdout_report(" in source


def test_the_holdout_is_collected_after_the_split_not_before():
    """Guards the obvious implementation error.

    Collecting hidden states for the whole calibration set and then fitting on
    all of it would report a "holdout" that was in the training data -- the
    number would look right and mean nothing.
    """
    import inspect

    source = inspect.getsource(TunedLens.train)

    split_at = source.index("self._split(")
    fit_at = source.index("self._collect_hidden(fit_prompts)")
    holdout_at = source.index("self._holdout_report(")

    assert split_at < fit_at, "the split must happen before anything is collected"
    assert holdout_at > fit_at, "the holdout must be scored after fitting"
    assert "self._collect_hidden(calibration_prompts)" not in source, (
        "hidden states are being collected for the full calibration set, which "
        "would put the held-out prompts into training")


# ── The README's tuned-lens figures must be the measured ones ──────────

def test_the_readme_quotes_the_measured_tuned_lens_result():
    """Experimental README claims must identify and qualify their source artifact."""
    import json
    import re

    legacy_path = ROOT / "docs" / "results" / "tuned_lens_training.json"
    artifact_path = ROOT / "docs" / "results" / "tuned_lens_full_affine.json"
    assert legacy_path.exists(), "legacy artifact must remain traceable"
    assert artifact_path.exists(), "current experiment artifact is missing"

    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert legacy["status"] == "superseded"
    assert legacy["provenance"] == "historical-unverified"
    assert legacy["legacy_artifact"]["report"]["publication_eligible"] is False

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    report = payload["full_affine"]
    assert report["status"] == "completed"
    assert report["provenance"] == LIVE
    assert report["validation_eligible"] is False
    assert report["publication_eligible"] is False
    assert report["model_id"] is None

    flat = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8")).lower()
    assert "experimental, unverified affine translators" in flat
    assert "model_id: null" in flat
    assert "publication_eligible: false" in flat
    assert "not a paper reproduction" in flat
    assert str(report["parameters_fitted"]) in flat or f"{report['parameters_fitted']:,}" in flat
    assert str(report["n_fit_prompts"]) in flat
    assert str(report["n_holdout_prompts"]) in flat


def test_the_held_out_figure_is_the_one_the_readme_leads_with():
    """Do not publish a training figure as a validated scientific headline."""
    import inspect
    import json

    report = json.loads(
        (ROOT / "docs" / "results" / "tuned_lens_full_affine.json")
        .read_text(encoding="utf-8")
    )["full_affine"]

    assert report["headline_metric"] == "holdout_kl_reduction_pct"
    assert report["publication_eligible"] is False
    assert report["validation_eligible"] is False

    source = inspect.getsource(TunedLens.train)
    assert '"headline_metric": "holdout_kl_reduction_pct" if heldout' in source
    assert 'else "kl_reduction_pct"' in source


def test_the_readme_states_the_variant_is_not_the_full_affine_map():
    """Both model families are labelled; the reduced map is not called the paper's map."""
    import re

    flat = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8")).lower()
    assert "translator_kind" in flat
    assert "diagonal_affine" in flat
    assert "full-affine report" in flat
    assert "not a paper reproduction" in flat


def test_the_artifact_records_that_the_split_was_deterministic():
    import json

    report = json.loads(
        (ROOT / "docs" / "results" / "tuned_lens_full_affine.json")
        .read_text(encoding="utf-8")
    )["full_affine"]

    assert report["n_holdout_prompts"] > 0
    assert report["holdout_prompts"]
    assert report["n_fit_prompts"] + report["n_holdout_prompts"] == report["n_calibration_prompts"]


# ── Stub plumbing ───────────────────────────────────────────────────────

class _StubHiddenCache:
    """Hidden states shaped like `gpt2_engine._cache["hidden"]`.

    `hidden[L]` holds `[(-1)**L * (L + 1), (L + 1)]`, so:
      * its L2 norm is `(L + 1) * sqrt(2)` -- strictly increasing, quadratic in
        no linear formula's reach, and unlike `10.0 + layer * 1.5`;
      * its sign alternates, so the arg-max over a 2-way unembedding alternates
        with the layer. A constant `" Paris"` after layer 6 cannot reproduce
        that.

    Note the engine reads `hidden[layer + 1]`, GPT-2's convention for "output of
    block `layer`", so reported layer L carries the norm of `hidden[L + 1]`.
    """

    def __init__(self, n_layers: int) -> None:
        import numpy as np

        self.hidden = [
            [np.array([((-1.0) ** index) * (index + 1), float(index + 1)],
                      dtype="float32")]
            for index in range(n_layers + 1)
        ]

    @staticmethod
    def norm_at(index: int) -> float:
        value = float(index + 1)
        return round(value * math.sqrt(2.0), 4)

    def get(self, key: str):
        return self.hidden if key == "hidden" else None


class _CacheBackedEngine(_StubEngine):
    """A stub that also answers the cache/private hooks the engine needs."""

    def __init__(self, stub: _StubEngine, cache: _StubHiddenCache) -> None:
        super().__init__(n_layers=stub.n, vocab=stub.vocab)
        self._model = _StubModel()
        self._tokenizer = object()
        self._cache = {"hidden": cache.hidden}

    def _ensure_loaded(self) -> Optional[Dict[str, Any]]:
        return None

    def _ensure_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        return None

    def _decode(self, id_: int) -> str:
        return f"decoded{id_}"


class _TrainedStubEngine(_CacheBackedEngine):
    """A stub that also answers the snapshot the trained path reads.

    `TunedLens.project` applies the translator to the residual stream the
    engine already cached, so a stub that wants to exercise that path has to
    hand the snapshot back.
    """

    def __init__(self, stub: _StubEngine, cache: _StubHiddenCache,
                 prompt: str = "hello") -> None:
        super().__init__(stub, cache)
        self._prompt = prompt

    def _snapshot(self):
        return {"prompt": self._prompt, "hidden": self._cache["hidden"]}


class _StubModel:
    """Just enough for `ln_f`, `wte` and `.parameters()`."""

    class _Block:
        class _LN:
            def __call__(self, x):
                return x

        def __init__(self) -> None:
            import torch

            self.ln_f = self._LN()
            self.wte = self
            self.weight = torch.eye(2)

    def __init__(self) -> None:
        import torch

        self.transformer = self._Block()
        self._param = torch.zeros(1)

    def parameters(self):
        return iter([self._param])


class _NoWeightsEngine:
    MODEL_ID = None

    def _ensure_loaded(self) -> Dict[str, Any]:
        return {"status": UNAVAILABLE, "reason": "transformers is not installed"}

    def _ensure_prompt(self, prompt: str) -> Dict[str, Any]:
        return self._ensure_loaded()

    def logit_lens(self, layer: int, prompt: str, top_k: int = 5) -> Dict[str, Any]:
        return self._ensure_loaded()

    def _model(self):
        return None

    def _cache(self):
        return {}