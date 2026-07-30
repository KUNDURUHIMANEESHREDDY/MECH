"""Tests for Automated Circuit Discovery Search (ACDC)."""

from __future__ import annotations

from science.models.gpt2_adapter import GPT2Adapter
from interpretability.discovery.acdc_search import ACDCSearchEngine
from interpretability.discovery.circuit_discovery import CircuitDiscoveryEngine


def test_acdc_search_engine_runs_in_mock_mode():
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    engine = ACDCSearchEngine(adapter)
    
    clean_prompt = "John and Mary went to the store, John gave a drink to Mary"
    corrupted_prompt = "John and Mary went to the store, Mary gave a drink to John"
    
    result = engine.search(
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        target_token=" Mary"
    )
    
    assert "circuit_id" in result
    assert result["prompt"] == clean_prompt
    assert len(result["nodes"]) >= 2
    assert len(result["edges"]) >= 1
    assert "circuit_score" in result

def test_circuit_discovery_engine_integration():
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    engine = CircuitDiscoveryEngine(adapter=adapter)
    
    result = engine.discover_circuit(prompt="The capital of France is")
    
    assert "circuit_id" in result
    assert len(result["nodes"]) >= 2
    # Check that it generated a corrupted prompt
    assert result["corrupted_prompt"] == "Corrupted The capital of France is"
