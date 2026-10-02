"""Tests for AI 3 Mechanistic Interpretability Sprint 2 epics."""
import pytest

from api.dispatcher import build_dispatcher


def test_sae_checkpoint_loading(tmp_path):
    """Dims must come from the real encoder matrix, not a config literal.

    This used to load "sae_gpt2.pt" -- a file that has never existed here --
    and assert ``status == "loaded"`` with ``d_sae == 16384``. That pair was
    only consistent while the loader reported success for a config-only SAE
    holding no tensors, so the assertion was really pinning the fabrication.
    """
    torch = pytest.importorskip("torch")

    d_in, d_sae = 768, 2048
    torch.manual_seed(0)
    path = tmp_path / "sae.pt"
    torch.save({"W_enc": torch.randn(d_in, d_sae),
                "b_pre": torch.zeros(d_in),
                "b_enc": torch.zeros(d_sae)}, path)

    dispatcher = build_dispatcher()
    res = dispatcher["interpretability/sae/load"]({"checkpoint_path": str(path)})

    assert res["status"] == "loaded"
    assert res["weights_loaded"] is True
    assert res["provenance"] == "live"
    # Read off the weights, so a wrong shape fails here rather than silently
    # reporting the default.
    assert res["d_in"] == d_in
    assert res["d_sae"] == d_sae
    assert res["weights_sha256"].split(":", 1)[1].__len__() == 64


def test_sae_load_of_absent_checkpoint_is_not_loaded():
    dispatcher = build_dispatcher()
    res = dispatcher["interpretability/sae/load"]({"checkpoint_path": "sae_gpt2.pt"})
    assert res["status"] == "unavailable"
    assert res["weights_loaded"] is False
    assert res["publication_eligible"] is False
    assert res["reason"]


def test_feature_inspection_dto():
    dispatcher = build_dispatcher()
    res = dispatcher["interpretability/features/inspect"]({"feature_id": 1402})
    assert res["feature_id"] == 1402
    assert "connected_neurons" in res
    assert "dataset_examples" in res
    assert "statistics" in res


def test_logit_lens_and_tuned_lens():
    dispatcher = build_dispatcher()
    logit = dispatcher["interpretability/projections/logit_lens"]({"prompt": "France is", "layer": 11})
    assert logit["method"] == "LogitLens"
    assert "top_token" in logit
    assert len(logit["top_k_tokens"]) >= 1

    tuned = dispatcher["interpretability/projections/tuned_lens"]({"prompt": "France is", "layer": 11})
    assert tuned["method"] == "TunedLens"
    # Honest: no trained per-layer translators ship with this repo, so the
    # tuned-lens route returns the raw LogitLens projection and says so.
    assert tuned["affine_translation_applied"] is False
    assert "no trained translator" in tuned.get("note", "")


def test_attention_head_ranking():
    dispatcher = build_dispatcher()
    ranked = dispatcher["interpretability/ranking/heads"]({"metric": "importance"})
    assert isinstance(ranked, list)
    assert len(ranked) > 0
    assert ranked[0]["metric"] == "importance"


def test_activation_and_feature_search():
    dispatcher = build_dispatcher()
    acts = dispatcher["interpretability/search/activations"]({"threshold": 1.0})
    assert isinstance(acts, list)

    # Federated search over real stores (KG labels, repo tokens, SAE dict).
    feats = dispatcher["interpretability/features/search"]({"query": "France"})
    assert isinstance(feats, list)
    assert all("source" in f for f in feats)

    # Direct SAE inspection still resolves the canonical demo feature, but with
    # no inspector loaded it must report no examples rather than the two fixed
    # activations (4.2, 3.8) it used to return for every feature id.
    insp = dispatcher["interpretability/features/inspect"]({"feature_id": 1402})
    assert insp["feature_id"] == 1402
    assert insp["dataset_examples"] == []
    assert insp["provenance"] == "unavailable"
    assert insp["statistics"]["n_examples"] == 0
