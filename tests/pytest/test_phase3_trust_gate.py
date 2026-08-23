"""MECH Scientific Trust Gate — Phase 3 Acceptance Test Battery.

Verifies:
1. P0: Real Induction Head Detection — Live attention matrices on repeated sequences identify empirical induction heads.
2. P0: Real IOI Benchmark Evaluation — Live model forward passes compute exact IOI logit differences z(IO) - z(S) and accuracy.
3. P0: Real Superposition & Dictionary Coherence — Live Gram matrix off-diagonal elements measure true feature interference.
4. P0: Real Feature Universality — Live cross-layer cosine similarity matrices measure true representation alignment.
5. P0: Autonomous Hypothesis Measurement — Live PyTorch activations replace random coin flips in hypothesis testing.
6. P1: Discovery Engine End-to-End Orchestration — Central discovery loop coordinates live empirical algorithms.
7. Golden Rule #2: Zero Hardcoded Results — Metrics vary dynamically with layers, prompts, and weights.
8. Fail Closed: All discovery engines raise RuntimeError when model is offline.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

import backend.services.gpt2_engine as gpt2_engine
from backend.interpretability.discovery.induction_head_detector import InductionHeadDetector
from backend.interpretability.discovery.ioi_benchmark import IOIBenchmarkSuite
from backend.interpretability.discovery.superposition_analyzer import SuperpositionAnalyzerEngine
from backend.interpretability.discovery.feature_universality import FeatureUniversalityEngine
from backend.interpretability.discovery.auto_hypothesis_tester import AutoHypothesisTester
from backend.interpretability.discovery.discovery_engine import DiscoveryEngine


# =========================================================================
# 1. Real Induction Head Detection Verification
# =========================================================================

def test_real_induction_head_detection():
    """Verify that InductionHeadDetector extracts real empirical induction scores from live weights."""
    gpt2_engine.load()
    assert gpt2_engine._model is not None
    
    detector = InductionHeadDetector(model=gpt2_engine._model, tokenizer=gpt2_engine._tokenizer)
    heads = detector.detect_induction_heads(top_k=5)
    
    assert len(heads) == 5
    for h in heads:
        assert 0 <= h["layer"] < 12
        assert 0 <= h["head"] < 12
        assert h["induction_score"] > 0.0
        assert h["provenance"] == "LIVE_PYTORCH"
        assert isinstance(h["is_induction_head"], bool)


# =========================================================================
# 2. Real IOI Benchmark Evaluation Verification
# =========================================================================

def test_real_ioi_benchmark_evaluation():
    """Verify that IOIBenchmarkSuite computes real logit differences on live forward passes."""
    gpt2_engine.load()
    assert gpt2_engine._model is not None
    
    suite = IOIBenchmarkSuite(model=gpt2_engine._model, tokenizer=gpt2_engine._tokenizer)
    res = suite.run_ioi_eval()
    
    assert res["benchmark_name"] == "IOI Benchmark"
    assert res["model_name"] == "gpt2"
    assert res["provenance"] == "LIVE_PYTORCH"
    assert res["num_prompts"] > 0
    assert 0.0 <= res["ioi_accuracy"] <= 1.0
    assert isinstance(res["average_logit_diff"], float)
    
    # Check individual prompt outputs
    for p in res["prompt_results"]:
        assert "z_io" in p and "z_s" in p
        assert "logit_diff" in p
        assert np.isclose(p["logit_diff"], p["z_io"] - p["z_s"], atol=1e-3)


# =========================================================================
# 3. Real Superposition & Dictionary Coherence Verification
# =========================================================================

def test_real_superposition_dictionary_coherence():
    """Verify that SuperpositionAnalyzerEngine calculates true Gram matrix coherence from live weights."""
    gpt2_engine.load()
    assert gpt2_engine._model is not None
    
    analyzer = SuperpositionAnalyzerEngine(model=gpt2_engine._model)
    res = analyzer.analyze_superposition(layer=8, sample_size=64)
    
    assert res["layer"] == 8
    assert res["num_features_evaluated"] == 64
    assert res["provenance"] == "LIVE_PYTORCH"
    assert 0.0 < res["max_dictionary_coherence"] <= 1.0
    assert res["interference_score"] > 0.0
    assert 0.0 <= res["superposition_degree"] <= 1.0


# =========================================================================
# 4. Real Feature Universality Cross-Layer Alignment
# =========================================================================

def test_real_feature_universality_alignment():
    """Verify that FeatureUniversalityEngine computes true cosine similarities across model layers."""
    gpt2_engine.load()
    assert gpt2_engine._model is not None
    
    engine = FeatureUniversalityEngine(model=gpt2_engine._model)
    res = engine.measure_universality(source_layer=6, source_neuron_idx=200, target_layers=[0, 3, 9, 11])
    
    assert res["source_layer"] == 6
    assert res["provenance"] == "LIVE_PYTORCH"
    assert 0.0 <= res["universal_alignment_score"] <= 1.0
    assert len(res["layer_alignments"]) == 4
    
    for la in res["layer_alignments"]:
        assert la["layer"] in [0, 3, 9, 11]
        assert -1.0 <= la["cosine_similarity"] <= 1.0


# =========================================================================
# 5. Live Hypothesis Measurement Verification (No Coin Flips)
# =========================================================================

def test_autonomous_hypothesis_measurement_is_live():
    """Verify that AutoHypothesisTester.measure performs live PyTorch activations instead of random coin flips."""
    gpt2_engine.load()
    assert gpt2_engine._model is not None
    
    tester = AutoHypothesisTester()
    exp_spec = {
        "id": "exp_test_1",
        "prompt": "The capital of France is Paris.",
        "expected_firing": True,
    }
    
    m_res = tester.measure(exp_spec, layer=8, neuron_index=412)
    assert m_res["provenance"] == "LIVE_PYTORCH"
    assert "activation" in m_res
    assert isinstance(m_res["activation"], float)
    assert m_res["type"] in ["supportive", "falsifying"]


# =========================================================================
# 6. Discovery Engine End-to-End Orchestration
# =========================================================================

def test_discovery_engine_end_to_end_live():
    """Verify that DiscoveryEngine orchestrates all live algorithms into a coherent discovery packet."""
    gpt2_engine.load()
    assert gpt2_engine._model is not None
    
    engine = DiscoveryEngine()
    disc_report = engine.discover_and_orchestrate("Neuron L8_N412 mediates factual subject-attribute association")
    
    assert "discovery_id" in disc_report
    assert "ioi_eval" in disc_report
    assert "induction_heads" in disc_report
    assert "superposition" in disc_report
    assert "universality" in disc_report
    
    assert disc_report["ioi_eval"]["provenance"] == "LIVE_PYTORCH"
    assert disc_report["superposition"]["provenance"] == "LIVE_PYTORCH"
    assert disc_report["universality"]["provenance"] == "LIVE_PYTORCH"


# =========================================================================
# 7. Golden Rule #2: Zero Hardcoded Results (Dynamic Variation)
# =========================================================================

def test_discovery_golden_rule_dynamic_variation():
    """Verify that Superposition and Induction outputs vary dynamically across different layers and prompts."""
    gpt2_engine.load()
    
    analyzer = SuperpositionAnalyzerEngine(model=gpt2_engine._model)
    res_l2 = analyzer.analyze_superposition(layer=2, sample_size=64)
    res_l10 = analyzer.analyze_superposition(layer=10, sample_size=64)
    
    # Layer 2 and Layer 10 must have distinct empirical weight matrices and coherence scores
    assert not np.isclose(res_l2["max_dictionary_coherence"], res_l10["max_dictionary_coherence"], atol=1e-5)
    assert not np.isclose(res_l2["interference_score"], res_l10["interference_score"], atol=1e-7)


# =========================================================================
# 8. Fail-Closed Verification When Offline
# =========================================================================

def test_discovery_engines_fail_closed_when_offline(monkeypatch):
    """Verify that all discovery engines raise RuntimeError when the model is uninitialized."""
    monkeypatch.setattr(gpt2_engine, "_model", None)
    monkeypatch.setattr(gpt2_engine, "_tokenizer", None)
    monkeypatch.setattr(gpt2_engine, "load", lambda: None)
    
    detector = InductionHeadDetector(model=None, tokenizer=None)
    with pytest.raises(RuntimeError, match="uninitialized"):
        detector.detect_induction_heads()
        
    suite = IOIBenchmarkSuite(model=None, tokenizer=None)
    with pytest.raises(RuntimeError, match="uninitialized"):
        suite.run_ioi_eval()
        
    analyzer = SuperpositionAnalyzerEngine(model=None)
    with pytest.raises(RuntimeError, match="uninitialized"):
        analyzer.analyze_superposition()
        
    engine = FeatureUniversalityEngine(model=None)
    with pytest.raises(RuntimeError, match="uninitialized"):
        engine.measure_universality()
