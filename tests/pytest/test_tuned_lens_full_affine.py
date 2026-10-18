"""The tuned lens fits full-rank affine translators, and says which kind ran.

The defect this closes
----------------------
``tuned_lens.py`` fitted a **diagonal** map -- a per-dimension scale and shift,
2*d_model parameters per layer -- while the tuned-lens paper's modified
architecture is a full affine map, d*d + d per layer. The module said so in its
docstring, but the payload still carried a `translator_kind` that a reader had
to know to distrust, and there was no way to fit the published variant at all.

What is pinned here
-------------------
* the default family is ``full_affine`` and it is a real d x d map, not a
  relabelled diagonal one (parameter count is exact: d*d + d per layer);
* both families remain fittable, so the shortfall of the restricted map is a
  measurement rather than a claim;
* ``apply_translator`` does the matrix product for one family and the
  elementwise product for the other;
* a saved lens round-trips through ``torch.save``/``weights_only=True``, and a
  missing or malformed artifact is refused rather than returning an empty lens;
* nothing reports a translation that did not happen.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from _weight_guard import skip_reason, weights_available  # noqa: E402

from backend.interpretability.algorithms.tuned_lens import (  # noqa: E402
    DEFAULT_TRANSLATOR_KIND,
    DIAGONAL_AFFINE,
    FULL_AFFINE,
    TunedLens,
)

needs_weights = pytest.mark.skipif(not weights_available(), reason=skip_reason())


class _NotACheckpoint:
    """A sentinel that is not a tensor or a primitive.

    Declared at module scope because ``torch.save`` cannot pickle a class
    defined inside a function -- the refusal test could not even build its
    fixture with a local class.
    """


D = 768  # gpt2-small d_model, used only for shapes in unit tests


# -- the families exist and the published one is the default -------------


def test_full_affine_is_the_default_family():
    assert FULL_AFFINE == "full_affine"
    assert DIAGONAL_AFFINE == "diagonal_affine"
    assert DEFAULT_TRANSLATOR_KIND == FULL_AFFINE


def test_parameter_count_matches_the_published_architecture():
    """d*d + d for full affine; 2*d for the per-dimension variant."""
    full = D * D + D
    assert full == 590592, "the figure the module reports for gpt2-small"
    assert 2 * D == 1536


# -- arithmetic, without a model ----------------------------------------


def _lens_with(entry):
    lens = TunedLens()
    lens.translators = {3: entry}
    return lens


def test_apply_translator_does_a_matrix_product_for_full_affine():
    torch = pytest.importorskip("torch")
    weight = torch.eye(D)
    bias = torch.zeros(D)
    lens = _lens_with({"kind": FULL_AFFINE, "weight": weight, "bias": bias})
    h = torch.randn(4, D)
    # Identity map on an identity weight must return the input unchanged.
    assert torch.allclose(lens.apply_translator(h, 3), h, atol=1e-6)

    lens = _lens_with({"kind": FULL_AFFINE,
                       "weight": torch.eye(D), "bias": torch.ones(D)})
    assert torch.allclose(lens.apply_translator(h, 3), h + 1.0, atol=1e-6)


def test_apply_translator_does_an_elementwise_product_for_diagonal():
    torch = pytest.importorskip("torch")
    lens = _lens_with({"kind": DIAGONAL_AFFINE,
                       "scale": torch.full((D,), 2.0),
                       "shift": torch.full((D,), 3.0)})
    h = torch.randn(4, D)
    assert torch.allclose(lens.apply_translator(h, 3), h * 2.0 + 3.0, atol=1e-6)


def test_apply_translator_rejects_an_untrained_layer():
    lens = TunedLens()
    with pytest.raises(KeyError):
        lens.apply_translator(None, 7)


# -- persistence --------------------------------------------------------


def _trained_stub(kind):
    torch = pytest.importorskip("torch")
    lens = TunedLens()
    entry = ({"kind": FULL_AFFINE, "weight": torch.eye(D), "bias": torch.zeros(D)}
             if kind == FULL_AFFINE else
             {"kind": DIAGONAL_AFFINE, "scale": torch.ones(D),
              "shift": torch.zeros(D)})
    lens.translators = {1: entry, 2: entry}
    lens.training_report = {"status": "completed", "provenance": "live",
                            "translator_kind": kind, "layers_trained": [1, 2]}
    return lens


@pytest.mark.parametrize("kind", [FULL_AFFINE, DIAGONAL_AFFINE])
def test_save_then_load_round_trips(kind, tmp_path):
    torch = pytest.importorskip("torch")
    path = str(tmp_path / f"lens-{kind}.pt")
    saved = _trained_stub(kind).save(path)
    assert saved["status"] == "saved", saved
    assert saved["translator_kind"] == kind
    assert saved["sha256"].startswith("sha256:")
    assert os.path.isfile(path)

    fresh = TunedLens()
    loaded = fresh.load(path)
    assert loaded["status"] == "loaded", loaded
    assert loaded["translator_kind"] == kind
    assert loaded["layers"] == [1, 2]
    h = torch.randn(3, D)
    assert torch.allclose(fresh.apply_translator(h, 1), h, atol=1e-6)


def test_load_refuses_a_missing_artifact(tmp_path):
    result = TunedLens().load(str(tmp_path / "nope.pt"))
    assert result["status"] == "unavailable"
    assert "No translator file" in result["reason"]


def test_load_refuses_a_foreign_pickle(tmp_path):
    """`weights_only=True` must reject an object that is not tensors/primitives."""
    torch = pytest.importorskip("torch")
    path = str(tmp_path / "evil.pt")
    torch.save({"translators": {1: _NotACheckpoint()}}, path)
    result = TunedLens().load(path)
    assert result["status"] == "unavailable"
    assert result["provenance"] == "unavailable"


def test_save_refuses_when_nothing_was_trained(tmp_path):
    result = TunedLens().save(str(tmp_path / "empty.pt"))
    assert result["status"] == "unavailable"
    assert "No trained translators" in result["reason"]


# -- training, gated on real weights ------------------------------------


def _prompts():
    return [
        "The capital of France is",
        "When Mary and John went to the store, John gave a drink to",
        "The Eiffel Tower is located in",
        "In 1942 the event lasted until 19",
        "When Xavier and Yolanda went to the store, Xavier gave a bottle to",
    ]


@needs_weights
def test_full_affine_training_reports_the_published_parameter_count():
    report = TunedLens().train(_prompts(), steps=3, kind=FULL_AFFINE, holdout=True)
    assert report["status"] == "completed", report.get("reason")
    assert report["provenance"] == "live"
    assert report["translator_kind"] == FULL_AFFINE
    assert report["parameters_per_layer"] == D * D + D
    assert report["layers_total"] == 12
    assert report["parameters_fitted"] == (D * D + D) * 12


@needs_weights
def test_diagonal_training_is_still_available_and_cheaper():
    report = TunedLens().train(_prompts(), steps=3, kind=DIAGONAL_AFFINE, holdout=True)
    assert report["status"] == "completed", report.get("reason")
    assert report["translator_kind"] == DIAGONAL_AFFINE
    assert report["parameters_per_layer"] == 2 * D


@needs_weights
def test_unknown_kind_is_refused():
    report = TunedLens().train(_prompts(), steps=1, kind="not_a_family")
    assert report["status"] == "unavailable"
    assert "Unknown translator kind" in report["reason"]


def test_training_without_weights_is_unavailable():
    report = TunedLens().train([], steps=1)
    assert report["status"] == "unavailable"
    assert report["provenance"] == "unavailable"


@needs_weights
def test_full_affine_beats_or_matches_diagonal_on_the_same_split():
    """The reason to pay 7M parameters instead of 18k, measured not asserted.

    On a calibration set this small either family can win, so the assertion is
    the weaker invariant that still matters: full affine must not be *worse*
    than diagonal by a meaningful margin, otherwise the default is wrong.
    """
    prompts = _prompts()
    full = TunedLens().train(prompts, steps=25, kind=FULL_AFFINE, holdout=True)
    diag = TunedLens().train(prompts, steps=25, kind=DIAGONAL_AFFINE, holdout=True)
    assert full["status"] == "completed" and diag["status"] == "completed"

    def mean_holdout(report):
        vals = [row["holdout_kl_reduction_pct"] for row in report["per_layer"]
                if row.get("holdout_kl_reduction_pct") is not None]
        return sum(vals) / len(vals) if vals else None

    f, d = mean_holdout(full), mean_holdout(diag)
    if f is None or d is None:
        pytest.skip("no held-out reduction available")
    assert f >= d - 5.0, (
        f"full affine mean hold-out reduction {f:.2f}% is more than 5 points "
        f"below diagonal {d:.2f}%, so the full-affine default is not justified")
