"""MECH Scientific Trust Gate — Phase 2 Acceptance Test Battery.

Verifies:
1. P0: Scientific Engines Fail Closed — No silent mock fallbacks in circuit, path verification, controlled causal, or Logit Lens engines.
2. P0: Module Hygiene — Unrelated job automation subsystem (backend.jobs) is completely excised.
3. P1: Model Adapters — Live mode executes real hook patching and strictly fails closed when offline.
4. P1: Unified Pipeline Integration — End-to-end chain (Logit Lens -> SAE Features -> Attention Routing -> Pathway Composition) executes against live PyTorch weights.
5. Golden Rule #2: Zero Hardcoded Results — All pipeline outputs vary dynamically with inputs.
"""

from __future__ import annotations

import importlib.util
import pytest
import torch
import numpy as np

from backend.science.logit_lens_engine import LogitLensEngine
from backend.science.scientific_circuit_engine import ScientificCircuitEngine
from backend.science.controlled_causal_engine import ControlledCausalEngine
from backend.science.path_verification_engine import PathVerificationEngine
from backend.science.models.gpt2_adapter import GPT2Adapter
from backend.science.models.adapter_base import ModelSpec





# =========================================================================
# 1. Scientific Engines Fail-Closed Verification (No Silent Deception)
# =========================================================================

def test_circuit_engine_fails_closed_when_uninitialized(monkeypatch):
    """Verify that ScientificCircuitEngine raises RuntimeError on model failure rather than returning fake metrics."""
    engine = ScientificCircuitEngine(model_id="nonexistent_model")
    
    # Mock gpt2_engine._model = None
    import backend.services.gpt2_engine as gpt2_engine
    monkeypatch.setattr(gpt2_engine, "_model", None)
    monkeypatch.setattr(gpt2_engine, "_tokenizer", None)
    monkeypatch.setattr(gpt2_engine, "load", lambda: None)
    
    with pytest.raises(RuntimeError, match="uninitialized|failed"):
        engine.get_layer_features(layer=0)


def test_path_verification_fails_closed_when_uninitialized(monkeypatch):
    """Verify that PathVerificationEngine raises RuntimeError rather than returning fake 87.5% rescue fraction."""
    engine = PathVerificationEngine(model_id="nonexistent_model")
    
    import backend.services.gpt2_engine as gpt2_engine
    monkeypatch.setattr(gpt2_engine, "_model", None)
    monkeypatch.setattr(gpt2_engine, "_tokenizer", None)
    monkeypatch.setattr(gpt2_engine, "load", lambda: None)
    
    with pytest.raises(RuntimeError, match="uninitialized|failed"):
        engine.verify_pathway(
            pathway_id="test_path",
            clean_prompt="Test prompt",
            target_token=" target",
            node_chain=["L0_N1", "L4_N2", "L8_N3"],
            edge_chain=["L0_N1->L4_N2", "L4_N2->L8_N3"],
        )



def test_controlled_causal_engine_fails_closed_when_uninitialized(monkeypatch):
    """Verify that ControlledCausalEngine raises RuntimeError rather than issuing fake CAUSALLY_VERIFIED certificates."""
    engine = ControlledCausalEngine(model_id="nonexistent_model")
    
    import backend.services.gpt2_engine as gpt2_engine
    monkeypatch.setattr(gpt2_engine, "_model", None)
    monkeypatch.setattr(gpt2_engine, "_tokenizer", None)
    monkeypatch.setattr(gpt2_engine, "load", lambda: None)
    
    with pytest.raises(RuntimeError, match="uninitialized|failed"):
        engine.evaluate_component_causality(
            prompt="The capital of France is",
            target_token=" Paris",
            layer=8,
            component_type="neuron",
            component_index=412,
        )


def test_logit_lens_fails_closed_when_uninitialized(monkeypatch):
    """Verify that LogitLensEngine raises RuntimeError rather than returning fake linear trajectories."""
    engine = LogitLensEngine(model_id="nonexistent_model")
    
    import backend.services.gpt2_engine as gpt2_engine
    monkeypatch.setattr(gpt2_engine, "_model", None)
    monkeypatch.setattr(gpt2_engine, "_tokenizer", None)
    monkeypatch.setattr(gpt2_engine, "load", lambda: None)
    
    with pytest.raises(RuntimeError, match="uninitialized|failed"):
        engine.compute_trajectory(
            prompt="The capital of France is",
            target_token=" Paris",
        )


# =========================================================================
# 2. Module Hygiene: backend.jobs Is Excised
# =========================================================================

def test_foreign_jobs_module_excised():
    """Verify that backend.jobs module does not exist and cannot be imported."""
    spec = importlib.util.find_spec("backend.jobs")
    assert spec is None, "Foreign backend.jobs module should not exist in MECH."


# =========================================================================
# 3. Model Adapters Live Hook Execution & Fail Closed
# =========================================================================

def test_model_adapter_fails_closed_when_offline():
    """Verify that ModelAdapter in live mode fails closed with RuntimeError when model is missing."""
    adapter = GPT2Adapter(mock_mode=False)
    adapter._model = None
    adapter._tokenizer = None
    
    with pytest.raises(RuntimeError, match="uninitialized"):
        adapter.get_activations("Test prompt", layer=0)
        
    with pytest.raises(RuntimeError, match="uninitialized"):
        adapter.patch_activation("Test prompt", layer=0, neuron_index=0, patch_value=1.0)


# =========================================================================
# 4. Live Unified Pipeline End-to-End Execution
# =========================================================================

def test_unified_pipeline_live_execution():
    """Verify that ScientificCircuitEngine executes the unbroken live pipeline with real GPT-2 weights."""
    import backend.services.gpt2_engine as gpt2_engine
    gpt2_engine.load()
    assert gpt2_engine._model is not None, "Live GPT-2 model must be loaded for pipeline verification."
    
    engine = ScientificCircuitEngine(model_id="gpt2")
    report = engine.generate_investigation_report(
        clean_prompt="The capital of France is",
        target_token=" Paris",
    )
    
    # 1. Logit Lens Divergence Layer is an integer in [0, 11]
    assert 0 <= report.predictive_divergence_layer < 12
    
    # 2. Features are extracted dynamically
    assert len(report.candidate_features) > 0
    primary_feat = report.candidate_features[0]
    assert primary_feat.layer == report.predictive_divergence_layer
    assert primary_feat.activation_strength > 0.0
    assert len(primary_feat.linear_logit_delta) > 0
    
    # 3. Weakest Link Circuit Composition is computed
    assert report.circuit_composition is not None
    assert report.circuit_composition.weakest_link_edge_id != ""
    assert report.circuit_composition.composed_causal_tier is not None
    assert report.circuit_composition.weakest_link_evidence_tier is not None


def test_unified_pipeline_golden_rule_dynamic_variation():
    """Verify Golden Rule #2: Pipeline outputs vary dynamically with different prompts and targets."""
    import backend.services.gpt2_engine as gpt2_engine
    gpt2_engine.load()
    
    engine = ScientificCircuitEngine(model_id="gpt2")
    
    report_france = engine.generate_investigation_report(
        clean_prompt="The capital of France is",
        target_token=" Paris",
    )
    
    report_germany = engine.generate_investigation_report(
        clean_prompt="The capital of Germany is",
        target_token=" Berlin",
    )
    
    # Both reports must be non-trivial
    assert report_france.clean_prompt == "The capital of France is"
    assert report_germany.clean_prompt == "The capital of Germany is"
    assert report_france.target_token == " Paris"
    assert report_germany.target_token == " Berlin"
    
    # Trajectories must be empirically computed and non-identical (13 states: embedding + 12 layers)
    assert len(report_france.logit_lens_trajectory) == 13
    assert len(report_germany.logit_lens_trajectory) == 13
    
    france_final_prob = report_france.logit_lens_trajectory[-1].probability
    germany_final_prob = report_germany.logit_lens_trajectory[-1].probability
    
    assert france_final_prob > 0.0
    assert germany_final_prob > 0.0
    # Dynamic values will not be exactly equal to 8 decimal places
    assert not np.isclose(france_final_prob, germany_final_prob, atol=1e-6)


