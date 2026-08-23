"""Test Epistemic Integrity and Fail-Closed Scientific Execution.

Ensures:
1. Zero silent fallbacks (no sha256 pseudo-math, no hardcoded keywords, no mock graphs).
2. Engines explicitly raise RuntimeError or return abstention when live model weights cannot be loaded.
3. Live execution returns genuine tensor-derived results with authentic component attributions.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from backend.interpretability.causal.attribution_patching import AttributionPatchingEngine
from backend.interpretability.causal.causal_tracing import CausalTracingEngine
from backend.interpretability.causal.path_patching import EdgePathPatchingEngine
from backend.interpretability.discovery.circuit_discovery import CircuitDiscoveryEngine
from backend.runtime.logits import IntermediateLogitsEngine


def test_circuit_discovery_fails_closed_without_adapter():
    """Circuit discovery must explicitly abstain rather than returning mock graphs."""
    engine = CircuitDiscoveryEngine(adapter=None)
    res = engine.discover_circuit(dataset_name="ioi", prompt_id="ioi_0001")
    assert res.get("error") == "EXECUTION_REQUIRED"
    assert res.get("status") == "abstained"
    assert "graph" not in res  # No fake graph allowed!


def test_attribution_patching_fails_closed_when_uninitialized():
    """Attribution patching must raise RuntimeError if model cannot be loaded."""
    with patch("backend.services.gpt2_engine.load", return_value={"status": "error"}), \
         patch("backend.services.gpt2_engine.is_available", return_value=False):
        engine = AttributionPatchingEngine()
        with pytest.raises(RuntimeError, match="Synthetic fallback generation is prohibited"):
            engine.compute_attribution(
                clean_prompt="When Mary and John went to the store, John gave a drink to",
                corrupted_prompt="When Mary and John went to the store, Mary gave a drink to",
            )


def test_causal_tracing_fails_closed_when_uninitialized():
    """Causal tracing must raise RuntimeError if model cannot be loaded."""
    with patch("backend.services.gpt2_engine.load", return_value={"status": "error"}), \
         patch("backend.services.gpt2_engine.is_available", return_value=False):
        engine = CausalTracingEngine()
        with pytest.raises(RuntimeError, match="Synthetic fallback generation is prohibited"):
            engine.trace_causal_effect(
                clean_prompt="The Eiffel Tower is in",
                corrupted_prompt="The Colosseum is in",
            )


def test_path_patching_fails_closed_when_uninitialized():
    """Path patching must raise RuntimeError if model cannot be loaded."""
    with patch("backend.services.gpt2_engine.load", return_value={"status": "error"}), \
         patch("backend.services.gpt2_engine.is_available", return_value=False):
        engine = EdgePathPatchingEngine()
        with pytest.raises(RuntimeError, match="Synthetic fallback generation is prohibited"):
            engine.test_edge_mediation(
                sender="L5_H2",
                receiver="L8_H5",
                clean_prompt="The Eiffel Tower is in",
                corrupted_prompt="The Colosseum is in",
            )


def test_logits_engine_fails_closed_when_uninitialized():
    """Intermediate logits projection must raise RuntimeError if model cannot be loaded."""
    with patch("backend.services.gpt2_engine.load", return_value={"status": "error"}), \
         patch("backend.services.gpt2_engine.is_available", return_value=False):
        engine = IntermediateLogitsEngine()
        with pytest.raises(RuntimeError, match="Synthetic fallback generation is prohibited"):
            engine.extract_logits(prompt="The capital of France is")


def test_live_attribution_patching_returns_genuine_tensors():
    """When live model is loaded, attribution patching returns genuine components."""
    engine = AttributionPatchingEngine()
    res = engine.compute_attribution(
        clean_prompt="When Mary and John went to the store, John gave a drink to",
        corrupted_prompt="When Mary and John went to the store, Mary gave a drink to",
    )
    assert "top_attributed_nodes" in res
    assert len(res["top_attributed_nodes"]) > 0
    assert "actual_logit_delta" in res
    assert res["actual_logit_delta"] != 3.42  # 3.42 was the old hardcoded fake delta!


def test_live_causal_tracing_returns_genuine_layer_effects():
    """When live model is loaded, causal tracing returns genuine hooked AIE."""
    engine = CausalTracingEngine()
    res = engine.trace_causal_effect(
        clean_prompt="The capital of France is",
        corrupted_prompt="The capital of Italy is",
    )
    assert "max_causal_layer" in res
    assert "max_indirect_effect" in res
    assert len(res["layer_effects"]) == 12
