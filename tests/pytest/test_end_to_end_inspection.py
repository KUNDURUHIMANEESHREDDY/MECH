"""End-to-end integration test verifying the full pipeline flow:
Runtime -> Repository -> Inspector -> Dispatcher -> JSON Response.
"""
from api.dispatcher import build_dispatcher
from repository import get_activation_repository


def test_end_to_end_inspection_pipeline():
    dispatcher = build_dispatcher()

    # 1. Query runtime status via dispatcher
    status = dispatcher["runtime:status"]({})
    assert status["status"] == "ready"

    # 2. Query repository activations via dispatcher
    repo_results = dispatcher["repository:query"]({"min_activation": 1.0, "sort_by": "activation"})
    assert isinstance(repo_results, list)
    assert len(repo_results) > 0

    # 3. Perform inspector analysis via dispatcher
    neuron_res = dispatcher["inspectors:neuron"]({"layer": repo_results[0]["layer"], "neuron_index": repo_results[0].get("neuron_index", 0)})
    assert neuron_res["neuron_id"].startswith("L")

    # 4. Perform search via dispatcher
    search_res = dispatcher["search"]({"type": "neuron", "query": "402"})
    assert isinstance(search_res, list)

    # 5. Fetch repository metrics via dispatcher
    metrics = dispatcher["repository:metrics"]({})
    assert metrics["hits"] >= 1
    assert "hit_ratio" in metrics
