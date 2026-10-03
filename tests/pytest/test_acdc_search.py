"""Tests for Automated Circuit Discovery Search (ACDC)."""

from __future__ import annotations

from science.models.gpt2_adapter import GPT2Adapter
from interpretability.discovery.algorithms.registry import get_algorithm
from interpretability.discovery.circuit_discovery import CircuitDiscoveryEngine


def test_acdc_search_engine_runs_in_mock_mode():
    """Without weights, ACDC must measure nothing -- and say so.

    This test previously asserted `retained_components >= 1`. That only held
    because of the substitution removed in this change: when the search retained
    nothing, ACDC added L9H9 and L10H0 under the comment "ensure top critical
    heads are retained". In other words the assertion was pinning the
    fabricated answer, not a property of the algorithm.

    Now `capture_head_outputs` raises `LiveUnavailable` with no weights (a
    per-head output vector has no fixture), every candidate is therefore
    unmeasurable, and the retained set is empty. An empty circuit is the honest
    result of a search that could not run, and it must not be backfilled with the
    published circuit.
    """
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    engine = get_algorithm("acdc", adapter)

    clean_prompt = "John and Mary went to the store, John gave a drink to Mary"
    corrupted_prompt = "John and Mary went to the store, Mary gave a drink to John"

    result = engine.run(
        dataset={
            "id": "ioi_0001",
            "clean": clean_prompt,
            "corrupted": corrupted_prompt,
            "target_token": " Mary",
        }
    ).to_dict()

    assert result["algorithm"] == "acdc"
    assert result["dataset_id"] == "ioi_0001"
    assert result["evidence"]["clean_prompt"] == clean_prompt
    assert result["evidence"]["corrupted_prompt"] == corrupted_prompt

    stats = result["statistics"]
    assert stats["retained_components"] == 0, (
        "ACDC retained components without weights; the head injection is back"
    )
    assert stats["total_evaluations"] == 0, (
        "ACDC claims to have evaluated candidates without weights"
    )
    assert stats["logit_recovery_fidelity"] is None
    assert stats["logit_recovery_fidelity_measured"] is False

    # No head may appear in the graph at all.
    head_nodes = [n for n in result["graph"]["nodes"] if n.get("type") == "Head"]
    assert head_nodes == [], (
        f"ACDC emitted head nodes without weights: {head_nodes}"
    )


def test_circuit_discovery_engine_integration():
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    engine = CircuitDiscoveryEngine(adapter=adapter)

    result = engine.discover_circuit(dataset_name="ioi", prompt_id="ioi_0001", algorithm_name="acdc")

    assert result["algorithm"] == "acdc"
    assert result["dataset_id"] == "ioi_0001"
    assert len(result["graph"]["nodes"]) >= 2
    # The IOI dataset provides the clean/corrupted pair used by ACDC
    assert result["evidence"]["corrupted_prompt"] != result["evidence"]["clean_prompt"]
