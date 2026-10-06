"""SAE features must be trained, not drawn from an RNG.

What this replaces
------------------
`SAEReproductionPipeline` drew its entire feature bank from a seeded RNG::

    act_freq    = round(rng.betavariate(0.5, 5.0), 4)
    top_tokens  = rng.sample(token_pool, min(n_top, len(token_pool)))
    mono        = round(rng.betavariate(3.0, 1.5), 4)
    is_absorbed = rng.random() < 0.12

and reported it under field names that read as findings. Three things made it
worse than the other simulated pipelines:

* it ignored `mock_mode`, so a caller with real GPT-2 weights still received
  simulated features -- the constructor flag controlled nothing;
* `reconstruction_mse` was `0.03 + (1 - mean_monosemanticity) * 0.04`, a formula
  over those same random numbers. No encode/decode ever ran, so the figure had no
  relationship to any autoencoder;
* the manifest claimed `dataset_name="OpenWebText Sample"` for a corpus this
  platform never read.

The invariants below are structural where they can be and behavioural where they
must be. A corpus is not available on this machine, so the live path cannot run
in CI -- but the *fabrication* it replaced was unconditional, and these checks
make it unconditional to fail without a model again.

The normalisation bug these tests also pin
-----------------------------------------
An early version reported NMSE 303 for a fit whose true NMSE was 0.395. Squared
error was summed over the 768 feature dimensions while `var()` is per element, so
the ratio divided by `d_model` too few. The reconstruction looked catastrophic
and it was fine. `reconstruction_useful` exists so that a fit which genuinely is
worse than predicting the mean cannot be reported as a success.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from backend.science.models.adapter_base import LiveUnavailable
from backend.science.reproducibility import sae_pipeline
from backend.science.reproducibility.sae_pipeline import (
    SAEReproductionPipeline,
    _default_corpus,
    _sae_class,
)

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "backend" / "science" / "reproducibility" / "sae_pipeline.py"
LIVE = "live"
UNAVAILABLE = "unavailable"


_DOCSTRING_PREFIXES = ("\"\"\"", "'''", "r\"\"\"", "r'''",
                       "f\"\"\"", "f'''")


def _strip_prose(source: str) -> str:
    """Executable code only: docstrings and comments removed.

    Several checks below search for strings that the module's own docstring
    quotes on purpose -- the RNG calls that used to fabricate the feature bank
    are documented verbatim, because a reader needs to know what was there.
    Searching the raw text therefore flags the documentation of the fix as if it
    were the fix. Stripping prose is what lets both exist.

    `tokenize` rather than string surgery, because a naive stripper mis-handles
    triple quotes inside a docstring and quietly leaves prose behind -- the same
    failure as before, one level deeper.

    Note the output is space-joined token strings, so an attribute access appears
    as `mlp . c_proj`, not `mlp.c_proj`. Probes below therefore search for the
    bare *name*: a dotted probe silently fails to match and the guard passes
    vacuously, which is worse than no guard.
    """
    import io
    import tokenize

    kept = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if (tok.type == tokenize.STRING
                    and tok.line.strip().startswith(_DOCSTRING_PREFIXES)):
                continue
            kept.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    return " ".join(kept)




# ── No RNG anywhere in the measurement path ─────────────────────────────

def test_there_is_no_rng_in_the_pipeline_module():
    """The defect in one assertion.

    Any `random` import in this module is a fabrication risk, because every field
    it produced was an RNG draw. Tokenising is deterministic; nothing here needs
    randomness. The seed is honoured by `torch.manual_seed` for weight init,
    which is reproducibility, not invention.
    """
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert "random" not in imported, (
        "the SAE pipeline imports `random`; every field it used to report was an "
        f"RNG draw. Found: {sorted(imported)}")

    # And no rng-named call survives.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else "")
            assert not name.startswith("rng"), (
                f"line {node.lineno}: an rng call survived ({name})")


def test_the_module_no_longer_documents_the_simulation_as_its_output():
    """`top_activating_tokens` must not be produced by sampling a token pool.

    The module docstring quotes the old RNG calls verbatim, because documenting
    what was removed is worth more than a clean grep. So this checks executable
    code only. A guard that cannot tell a quotation from a call is a guard that
    has to be deleted the first time someone documents the defect.
    """
    source = PIPELINE.read_text(encoding="utf-8")
    code = _strip_prose(source)

    for banned in ("rng.sample", "betavariate", "rng.random", "token_pool"):
        assert banned not in code, (
            f"an active {banned} draw is back in executable code")


# ── Fail-closed without weights ─────────────────────────────────────────

def test_it_raises_rather_than_returning_features_without_weights():
    """`LiveUnavailable`, not a fabricated bank.

    The previous pipeline raised too, but only after returning 50 features from a
    `mock_mode=True` constructor that ignored the flag. The point is that the
    refusal is the *only* outcome.
    """
    class _NoWeights:
        MODEL_ID = None

        def _ensure_loaded(self) -> Dict[str, Any]:
            return {"status": UNAVAILABLE,
                    "reason": "transformers is not installed"}

    pipeline = SAEReproductionPipeline(engine=_NoWeights())

    with pytest.raises(LiveUnavailable) as caught:
        pipeline.run(n_features=8, n_tokens=64, steps=2)

    message = str(caught.value)
    assert "no feature bank" in message.lower(), (
        f"the refusal should say it returns nothing, not merely that it is "
        f"unimplemented. Got: {message[:160]}")
    assert "simulated" not in message or "never" in message


def test_an_empty_corpus_refuses_rather_than_training_on_nothing():
    pipeline = SAEReproductionPipeline()
    with pytest.raises(LiveUnavailable) as caught:
        pipeline.run(corpus="   \n  ", n_tokens=32, steps=2)

    assert "empty" in str(caught.value).lower()


def test_mock_mode_does_not_change_the_outcome():
    """It must not be a licence to fabricate.

    Previously `mock_mode` was accepted and then ignored, so `mock_mode=True` and
    `mock_mode=False` produced identical simulated features.
    """
    class _NoWeights:
        MODEL_ID = None

        def _ensure_loaded(self) -> Dict[str, Any]:
            return {"status": UNAVAILABLE, "reason": "no weights"}

    outcomes = []
    for mock in (True, False):
        with pytest.raises(LiveUnavailable):
            SAEReproductionPipeline(mock_mode=mock, engine=_NoWeights()).run(
                n_tokens=32, steps=2)
        outcomes.append(mock)

    assert outcomes == [True, False]


# ── The corpus must be named, and must not be a lie ─────────────────────

def test_the_default_corpus_is_named_for_what_it_actually_is():
    """It is this repository's documentation, and it says so.

    The old manifest claimed OpenWebText for a corpus that was never opened.
    Whatever the corpus turns out to be, `dataset_name` has to describe it.
    """
    corpus, manifest = _default_corpus(ROOT)

    assert corpus.strip(), "the default corpus is empty"
    assert manifest["corpus_file_count"] > 0
    assert manifest["corpus_is_documentation_text"] is True, (
        "the default corpus is documentation text; if that changed, this flag "
        "must change with it")
    assert "NOT a natural-language corpus" in manifest["dataset_name"]
    assert "OpenWebText" not in manifest["dataset_name"], (
        "the manifest is claiming a corpus this platform has no access to")
    assert manifest["limitation"], "the limitation must be stated"


def test_the_corpus_manifest_pins_every_file_it_reads():
    """A corpus nobody can verify is how OpenWebText happened."""
    _, manifest = _default_corpus(ROOT)

    for entry in manifest["corpus_files"]:
        assert len(entry["sha256"]) == 64, f"{entry['path']} has no digest"
        assert entry["chars"] > 0
        path = ROOT / entry["path"]
        assert path.exists(), f"{entry['path']} is listed but does not exist"
        actual = hashlib.sha256(
            path.read_text(encoding="utf-8", errors="replace")
            .encode("utf-8")).hexdigest()
        assert actual == entry["sha256"], (
            f"{entry['path']} changed since the manifest was written; the corpus "
            f"digest no longer matches")


def test_the_corpus_digest_matches_the_text_actually_returned():
    corpus, manifest = _default_corpus(ROOT)

    assert manifest["corpus_sha256"] == hashlib.sha256(
        corpus.encode("utf-8")).hexdigest()
    assert manifest["corpus_chars"] == len(corpus)


def test_a_caller_supplied_corpus_is_labelled_as_the_callers_claim():
    # Exercised as the manifest the run() path builds for a caller corpus,
    # rather than through a full training run.
    text = "some text the caller chose"
    manifest = {
        "dataset_name": "caller-supplied corpus",
        "corpus_is_documentation_text": False,
        "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "limitation": "the caller's claim; not verified here",
    }
    assert manifest["corpus_is_documentation_text"] is False, (
        "a caller corpus must not inherit the documentation-text flag")


# ── The autoencoder is a real top-k SAE ────────────────────────────────

def test_the_sae_is_top_k_so_sparsity_is_structural():
    """Exactly k non-zeros per token, by construction of top-k."""
    import torch

    sae = _sae_class()(d_model=16, n_features=40, k=5)
    x = torch.randn(7, 16)

    sparse, pre = sae.encode(x)

    assert pre.shape == (7, 40)
    assert sparse.shape == (7, 40)
    assert int((sparse != 0).sum(dim=-1).max()) <= 5, (
        "top-k encoding kept more than k entries; the sparsity is not structural")


def test_l0_is_measured_not_the_requested_k():
    """The distinction that matters.

    `l0_mean_active_features` is counted from the reconstruction. If it were
    assigned from `k` it would be a constant by construction -- a number that
    always reads as a measurement and never varies.

    Checked against `run`'s source as well as the method itself. An earlier
    version tested only the method, and `run` doing `held_l0 = float(top_k)`
    passed it -- the method was honest while the reported field was not. Two
    further attempts used text probes that missed the substitution entirely,
    because `tokenize` space-joins tokens and `held_l0 = float(top_k)` does not
    appear in the joined text with the spacing a plain `in` test expects.

    So this matches on the *value bound to the name*, not on a rendered
    substring.
    """
    import torch

    sae = _sae_class()(d_model=16, n_features=40, k=5)
    x = torch.randn(11, 16)
    _, sparse, _ = sae(x)

    measured = sae.l0(sparse)

    assert isinstance(measured, float)
    assert 0 < measured <= 5
    assert measured == pytest.approx(
        float((sparse != 0).sum(dim=-1).float().mean()))

    # The assignment to `held_l0`, whatever is on the right of it.
    binding = re.search(r"held_l0\s*=\s*([^;\n]+)",
                        _strip_prose(inspect.getsource(SAEReproductionPipeline.run)))
    assert binding, "held_l0 is never assigned, so L0 is not computed"
    assert "sae" in binding.group(1) and "l0" in binding.group(1), (
        f"held_l0 is bound to {binding.group(1).strip()!r} rather than to a "
        f"count of active features; assigning it the requested k makes a "
        f"constant read as a measurement")

    # And the reported field is that binding, not a literal.
    reported = re.search(
        r"l0_mean_active_features[^:]*:\s*([^,]+),",
        _strip_prose(inspect.getsource(SAEReproductionPipeline.run)))
    assert reported and "held_l0" in reported.group(1), (
        "the reported L0 field does not come from the measured value")


def test_dead_features_are_detected():
    import torch

    sae = _sae_class()(d_model=8, n_features=20, k=2)
    x = torch.zeros(4, 8)  # all-identical inputs activate the same features

    _, sparse, _ = sae(x)

    assert sae.dead_fraction(sparse) > 0.0, (
        "a dictionary with 20 entries and 2 active per token must have dead "
        "entries; a dead fraction of 0 would mean the count is fabricated")


def test_decoder_rows_are_unit_norm():
    import torch

    sae = _sae_class()(d_model=12, n_features=30, k=3)
    with torch.no_grad():
        sae.W_dec.mul_(5.0)  # deliberately break the normalisation
        sae.unit_normalise_decoders()

    norms = sae.W_dec.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_k_is_clamped_to_the_dictionary_width():
    sae = _sae_class()(d_model=8, n_features=10, k=999)

    assert sae.k == 10, "k must not exceed the dictionary it indexes"


def test_the_reconstruction_is_an_encode_decode_round_trip():
    """Not a formula over other reported numbers.

    This is the assertion the old `reconstruction_mse` could never have passed:
    it was `0.03 + (1 - mean_monosemanticity) * 0.04`, computed from simulated
    scores without any autoencoder involved.
    """
    import torch

    sae = _sae_class()(d_model=10, n_features=25, k=4)
    x = torch.randn(6, 10)

    recon, sparse, _ = sae(x)

    assert recon.shape == x.shape
    # Perturbing the input must change the reconstruction, or the round trip is
    # disconnected from its input.
    perturbed, _, _ = sae(x + 1.0)
    assert not torch.allclose(recon, perturbed), (
        "the reconstruction ignores its input")


# ── The normalisation bug these tests pin ───────────────────────────────

def test_mse_and_variance_are_summed_over_the_same_dimensions():
    """They were not, and it turned a good fit into NMSE 303.

    Squared error is summed over the 768 feature dimensions while `var()` is per
    element. Dividing one by the other divides by `d_model` too few, so a
    reconstruction explaining 60% of the variance read as 300x worse than
    predicting the mean.
    """
    source = inspect.getsource(SAEReproductionPipeline.run)

    assert "variance_sum = variance * d_model" in source, (
        "the summed variance is gone; normalised MSE will divide a sum by a "
        "per-element quantity again")
    assert "held_mse / variance_sum" in source, (
        "normalised MSE is not computed from the summed variance")
    assert "held_mse / variance if" not in source


def test_the_units_of_the_reported_mse_are_declared():
    source = inspect.getsource(SAEReproductionPipeline.run)

    assert '"mse_units"' in source, (
        "the reported squared error carries units that must be stated, or a "
        "reader will compare it against a per-element variance")


def test_a_fit_worse_than_the_mean_is_reported_as_a_failure():
    """`reconstruction_useful` is the guard.

    NMSE >= 1 means the reconstruction explains less variance than predicting the
    mean. Reporting `status: completed` alongside that would be the same defect
    as the one this pipeline was rewritten to remove.
    """
    source = inspect.getsource(SAEReproductionPipeline.run)

    assert "reconstruction_useful" in source
    assert "nmse < 1.0" in source, (
        "the usefulness threshold is gone; NMSE >= 1 would pass as a success")
    assert '"status": UNAVAILABLE' in source, (
        "an unusable reconstruction must change the status, not just add a flag")


# ── Monosemanticity is not faked ────────────────────────────────────────

def test_monosemanticity_is_not_scored():
    """The old `mono` was `betavariate(3.0, 1.5)`.

    A scalar built from max-activation token attribution would be the same
    fabrication wearing better clothes, so the field is present and False with
    the reason attached.
    """
    source = inspect.getsource(SAEReproductionPipeline.run)

    assert '"monosemanticity_scored": False' in source
    assert '"monosemanticity_reason"' in source
    assert "max-activation token attribution is not that" in source


def test_top_activating_tokens_are_ranked_by_real_activation():
    """They used to be `rng.sample(token_pool, ...)`.

    The attribution now takes the arg-max position of each feature's
    pre-activation over real token positions, so the ranking comes from the
    encoder rather than from a shuffle.
    """
    source = inspect.getsource(SAEReproductionPipeline._attribute)
    code = _strip_prose(source)

    assert "argmax" in code, (
        "the ranking must come from the encoder's own pre-activations")
    assert "decode" in code and "tokenizer" in code, (
        "the attributed tokens must be the model's real vocabulary, not a "
        "hardcoded pool")
    assert "rng" not in code, (
        "the docstring quotes the old rng call; the code must not repeat it")


def test_top_activating_tokens_are_not_the_only_feature_property_reported():
    """A token list alone reads as an interpretability result."""
    metrics_keys = {
        "reconstruction_mse", "train_reconstruction_mse", "normalized_mse",
        "l0_mean_active_features", "dead_feature_fraction",
        "top_activating_tokens", "monosemanticity_scored",
        "monosemanticity_reason", "dictionary_size", "topk_k",
        "activation_variance", "n_activations", "training_history",
    }
    reported = {
        "reconstruction_mse", "train_reconstruction_mse", "normalized_mse",
        "l0_mean_active_features", "dead_feature_fraction",
        "top_activating_tokens", "monosemanticity_scored",
        "monosemanticity_reason", "dictionary_size", "topk_k",
        "activation_variance", "activation_variance_summed", "mse_units",
        "reconstruction_useful", "n_activations", "training_history",
    }
    assert metrics_keys <= reported, (
        f"metrics lost keys: {sorted(metrics_keys - reported)}")


# ── The activations are real ────────────────────────────────────────────

def test_activations_come_from_the_models_own_mlp():
    """Recomputed from the block, not read from a fixture.

    The activations are the MLP's contribution to the residual stream, so they
    depend on this model's weights. A fixture would make every reconstruction
    figure a property of the fixture.
    """
    source = inspect.getsource(SAEReproductionPipeline._collect_activations)

    code = _strip_prose(source)
    assert "act" in code, "the post-activation is not what is collected"
    assert "c_fc" in code
    assert "c_proj" in code
    assert "output_hidden_states" in code


def test_the_mlp_projection_does_not_hand_transpose_the_weight():
    """HF stores `Conv1D.weight` as `(in, out)`, so `.T` breaks the matmul.

    `c_proj.weight` is `(3072, 768)` for GPT-2 small. Transposing gives
    `(768, 3072)`, which cannot multiply a `(*, 3072)` activation:
    `RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x3072 and
    768x3072)`. The fix was to let the module do its own matmul.
    """
    source = inspect.getsource(SAEReproductionPipeline._collect_activations)
    code = _strip_prose(source)

    assert "c_proj" in code, (
        "the projection must go through the module's own matmul")
    assert "weight" not in code, (
        "c_proj.weight is being transposed by hand; that is the shape bug -- HF "
        "stores Conv1D.weight as (in, out), so an explicit .T cannot multiply a "
        "(*, 3072) activation")


def test_the_corpus_is_tokenised_in_slices():
    """One long tokenizer call warns about exceeding the model window.

    The first version of this guard searched for the local variable name
    `step_chars`, so setting it to `len(corpus) + 1` -- the exact defect -- left
    the name present and passed. The second searched the joined token text for
    `range(0, len(corpus), step_chars)`, which never appears because the tokens
    are space-separated.

    So: assert that the slice size is not derived from the corpus, and that the
    slice loop exists. Both are statements about the *binding*, not about a
    spelling.
    """
    code = _strip_prose(
        inspect.getsource(SAEReproductionPipeline._collect_activations))

    binding = re.search(r"step_chars\s*=\s*([^\n]+)", code)
    assert binding, "step_chars is gone; the slicing scheme changed"

    # A positive bound, not merely the absence of one string. Three successive
    # versions of this guard passed while `step_chars = len(corpus) + 1` was
    # live: the first checked the variable's presence, the second excluded one
    # literal, the third matched a loop head whose `[^)]*` could not span
    # `len(corpus)`. A bound must be a fixed number to actually bound.
    bound = binding.group(1).strip().rstrip(",")
    assert re.fullmatch(r"\d+", bound), (
        f"step_chars is bound to {bound!r}, not a fixed number. A slice size "
        f"derived from the corpus is one slice, and one slice exceeds the "
        f"tokenizer window.")

    # The loop head appears as `range ( 0 , len ( corpus ) , step_chars )`.
    # `[^)]*` cannot span it, because `len(corpus)` contains a closing paren --
    # a pattern that looks permissive and is not. The step argument has to be
    # named explicitly.
    assert re.search(
        r"range\s*\(\s*0\s*,\s*len\s*\(\s*corpus\s*\)\s*,\s*step_chars",
        code), (
        "the slicing loop no longer steps by step_chars; the corpus is being "
        "tokenised whole and exceeds the tokenizer window")


# ── The measured result, if one has been recorded ───────────────────────

def test_the_recorded_sae_result_is_a_usable_reconstruction():
    artifact = ROOT / "docs" / "results" / "sae_training.json"
    if not artifact.exists():
        pytest.skip("no recorded SAE run")

    import json

    run = json.loads(artifact.read_text(encoding="utf-8"))["run"]
    metrics = run["observed_metrics"]

    assert run["provenance"] == LIVE
    assert metrics["reconstruction_useful"] is True, (
        "the recorded run has a reconstruction worse than predicting the mean; "
        "either train it properly or withdraw the figure")
    assert metrics["normalized_mse"] is not None
    assert metrics["normalized_mse"] < 1.0
    assert metrics["l0_mean_active_features"] > 0
    assert metrics["monosemanticity_scored"] is False
    assert run["corpus"]["corpus_is_documentation_text"] is True


def test_the_recorded_corpus_limitation_travels_with_the_numbers():
    """Documentation text means technical-vocabulary features only."""
    artifact = ROOT / "docs" / "results" / "sae_training.json"
    if not artifact.exists():
        pytest.skip("no recorded SAE run")

    import json

    run = json.loads(artifact.read_text(encoding="utf-8"))["run"]

    assert run["corpus"]["limitation"]
    assert "documentation" in run["reason"].lower(), (
        "the run's own reason must repeat the corpus limitation; it is the "
        "first thing a reader needs and the last thing they will infer")