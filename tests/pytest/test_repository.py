"""Tests for Activation Repository query engine, search, and metrics."""
from repository import get_activation_repository


def test_repository_query_filtering():
    repo = get_activation_repository()
    # Filter by layer 8
    res = repo.query(layer=8)
    assert len(res) >= 1
    assert res[0]["layer"] == 8

    # Filter by min_activation / threshold
    res_thresh = repo.query(min_activation=2.0)
    assert all(r["activation"] >= 2.0 for r in res_thresh)


def test_repository_search_api():
    repo = get_activation_repository()
    neurons = repo.search("neuron", query="402")
    assert len(neurons) >= 1

    tokens = repo.search("token", query="France")
    assert len(tokens) >= 1
    assert tokens[0]["token"] == "France"


def test_repository_aggregate_and_stats():
    repo = get_activation_repository()
    agg = repo.aggregate(group_by="layer")
    assert "grouped_by" in agg
    assert "groups" in agg

    stats = repo.statistics()
    assert stats["total_records"] > 0
    assert "mean_activation" in stats


def test_repository_cache_metrics():
    repo = get_activation_repository()
    repo.query(layer=0)  # trigger query
    metrics = repo.get_metrics()
    assert "hits" in metrics
    assert "misses" in metrics
    assert "hit_ratio" in metrics
    assert "memory_bytes" in metrics
    assert "average_lookup_ms" in metrics
