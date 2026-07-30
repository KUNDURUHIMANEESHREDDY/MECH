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

    tuned = dispatcher["interpretability/projections/tuned_lens"]({"prompt": "France is", "layer": 11})
    assert tuned["method"] == "TunedLens"
    assert tuned["affine_translation_applied"] is True


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

    feats = dispatcher["interpretability/features/search"]({"query": "Indirect Object"})
    assert len(feats) >= 1
    assert feats[0]["feature_id"] == 1402
