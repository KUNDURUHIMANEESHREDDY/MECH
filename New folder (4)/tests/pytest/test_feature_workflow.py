"""Integration Test: SAE Feature Workflow.

Validates: SAE -> Feature Search -> Feature Inspection DTO -> Dataset Examples.
"""
from api.dispatcher import build_dispatcher


def test_feature_workflow():
    dispatcher = build_dispatcher()

    # 1. Load SAE checkpoint
    sae_load = dispatcher["interpretability/sae/load"]({"checkpoint_path": "sae_gpt2.pt"})
    assert sae_load["status"] == "loaded"

    # 2. Search features matching query string
    search_res = dispatcher["interpretability/features/search"]({"query": "Capital"})
    assert len(search_res) >= 1
    feat_id = search_res[0]["feature_id"]

    # 3. Perform detailed feature inspection
    inspect_res = dispatcher["interpretability/features/inspect"]({"feature_id": feat_id})
    assert inspect_res["feature_id"] == feat_id
    assert len(inspect_res["dataset_examples"]) >= 1
    assert len(inspect_res["connected_neurons"]) >= 1
