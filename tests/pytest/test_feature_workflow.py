"""Integration Test: SAE Feature Workflow.

Validates: SAE -> Feature Search -> Feature Inspection DTO -> Dataset Examples.

This test used to point the loader at "sae_gpt2.pt", a file that has never
existed in this repository, and assert ``status == "loaded"``. That assertion
was only ever satisfiable while the loader reported success for a config-only
SAE with no tensors in it. The loader now fails closed, so the test builds a
real checkpoint and exercises the actual code path.
"""
import pytest

from api.dispatcher import build_dispatcher


torch = pytest.importorskip("torch")


@pytest.fixture
def sae_checkpoint(tmp_path):
    """A genuine SAE checkpoint: encoder weights plus both bias terms."""
    torch.manual_seed(0)
    path = tmp_path / "sae.pt"
    torch.save(
        {
            "W_enc": torch.randn(8, 16),
            "b_pre": torch.zeros(8),
            "b_enc": torch.zeros(16),
        },
        path,
    )
    return str(path)


def test_feature_workflow(sae_checkpoint):
    dispatcher = build_dispatcher()

    # 1. Load a SAE checkpoint that actually exists.
    sae_load = dispatcher["interpretability/sae/load"](
        {"checkpoint_path": sae_checkpoint}
    )
    assert sae_load["status"] == "loaded"
    assert sae_load["weights_loaded"] is True
    assert sae_load["provenance"] == "live"
    # The hash must be over real tensors, not derived from the filename.
    digest = sae_load["weights_sha256"].split(":", 1)[1]
    assert len(digest) == 64

    # 2. Search features matching query string (federated over KG,
    # activation repository, and SAE dictionary)
    search_res = dispatcher["interpretability/features/search"]({"query": "France"})
    assert isinstance(search_res, list)
    assert all("source" in f for f in search_res)

    # 3. Detailed feature inspection must not invent activations. No SAE
    # inspector is loaded, so the honest answer is "no evidence" -- previously
    # this returned fixed weights (0.85, 0.62) and activations (4.2, 3.8) for
    # every feature id and derived statistics from them.
    inspect_res = dispatcher["interpretability/features/inspect"]({"feature_id": 1402})
    assert inspect_res["feature_id"] == 1402
    assert inspect_res["provenance"] == "unavailable"
    assert inspect_res["inspected"] is False
    assert inspect_res["validation_eligible"] is False
    assert inspect_res["publication_eligible"] is False
    assert inspect_res["reason"]

    # max_act was previously computed as max(4.2, 3.8) from the fixed
    # activations. With no analysis there is no statistic.
    assert inspect_res["statistics"]["max_act"] is None
    assert inspect_res["statistics"]["n_examples"] == 0
    assert inspect_res["dataset_examples"] == []
    assert inspect_res["connected_neurons"] == []


def test_loading_an_absent_checkpoint_reports_unavailable():
    """The fail-closed path, asserted rather than assumed."""
    dispatcher = build_dispatcher()
    res = dispatcher["interpretability/sae/load"](
        {"checkpoint_path": "definitely_not_here.pt"}
    )
    assert res["status"] == "unavailable"
    assert res["weights_loaded"] is False
    assert res["validation_eligible"] is False
    assert res["publication_eligible"] is False
    assert res["reason"]
