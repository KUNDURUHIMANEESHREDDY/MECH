"""Tests for Automated Circuit Discovery Search (ACDC)."""

from __future__ import annotations

from science.models.gpt2_adapter import GPT2Adapter
from interpretability.discovery.algorithms.registry import get_algorithm
from interpretability.discovery.circuit_discovery import CircuitDiscoveryEngine


def test_acdc_search_engine_runs_in_mock_mode():
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
    assert len(result["graph"]["nodes"]) >= 2
    assert len(result["graph"]["edges"]) >= 1
    assert result["evidence"]["clean_prompt"] == clean_prompt
    assert result["evidence"]["corrupted_prompt"] == corrupted_prompt
    assert result["statistics"]["retained_components"] >= 1


def test_circuit_discovery_engine_integration():
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    engine = CircuitDiscoveryEngine(adapter=adapter)

    result = engine.discover_circuit(dataset_name="ioi", prompt_id="ioi_0001", algorithm_name="acdc")

    assert result["algorithm"] == "acdc"
    assert result["dataset_id"] == "ioi_0001"
    assert len(result["graph"]["nodes"]) >= 2
    # The IOI dataset provides the clean/corrupted pair used by ACDC
    assert result["evidence"]["corrupted_prompt"] != result["evidence"]["clean_prompt"]
