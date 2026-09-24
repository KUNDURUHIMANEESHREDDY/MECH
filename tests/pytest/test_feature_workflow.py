"""Integration Test: SAE Feature Workflow.

Validates: SAE -> Feature Search -> Feature Inspection DTO -> Dataset Examples.
"""
from api.dispatcher import build_dispatcher


def test_feature_workflow():
    dispatcher = build_dispatcher()

    # 1. Load SAE checkpoint
    sae_load = dispatcher["interpretability/sae/load"]({"checkpoint_path": "sae_gpt2.pt"})
    assert sae_load["status"] == "loaded"

    # 2. Search features matching query string (federated over KG,
    # activation repository, and SAE dictionary)
    search_res = dispatcher["interpretability/features/search"]({"query": "France"})
    assert isinstance(search_res, list)
    assert all("source" in f for f in search_res)

    # 3. Perform detailed feature inspection on the canonical feature
    inspect_res = dispatcher["interpretability/features/inspect"]({"feature_id": 1402})
    assert inspect_res["feature_id"] == 1402
    assert len(inspect_res["dataset_examples"]) >= 1
    assert len(inspect_res["connected_neurons"]) >= 1
