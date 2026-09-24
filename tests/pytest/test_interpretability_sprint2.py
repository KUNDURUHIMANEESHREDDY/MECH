"""Tests for AI 3 Mechanistic Interpretability Sprint 2 epics."""
from api.dispatcher import build_dispatcher


def test_sae_checkpoint_loading():
    dispatcher = build_dispatcher()
    res = dispatcher["interpretability/sae/load"]({"checkpoint_path": "sae_gpt2.pt"})
    assert res["status"] == "loaded"
    assert res["d_sae"] == 16384


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

    # Direct SAE inspection still resolves the canonical demo feature.
    insp = dispatcher["interpretability/features/inspect"]({"feature_id": 1402})
    assert insp["feature_id"] == 1402
    assert len(insp["dataset_examples"]) >= 1
