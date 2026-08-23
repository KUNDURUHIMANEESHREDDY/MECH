"""MECH Scientific Trust Gate — Phase 4 Acceptance Test Battery.

Verifies:
1. P0: Multi-Model Adapter Factory — Instantiates real specs across Gemma, Llama, Qwen, Mistral, and DeepSeek.
2. P0: Empirical Cross-Model Circuit Alignment — Computes structural, functional, and causal alignment metrics.
3. P0: Empirical Concept Evolution & Lineage — Computes representation lineage across network depth.
4. P0: Empirical Semantic Drift Reports — Computes bipartite Jaccard drift and persistence between models.
5. Golden Rule #2: Zero Hardcoded Results — Metrics vary dynamically across model pairs and prompt suites.
6. Fail Closed: Multi-model adapters in live mode strictly raise RuntimeError when weights are offline.
"""

from __future__ import annotations

import pytest
import numpy as np

from backend.science.models.model_adapters import (
    create_adapter,
    GemmaAdapter,
    LlamaAdapter,
    QwenAdapter,
    MistralAdapter,
    DeepSeekAdapter,
)
from backend.science.models.gpt2_adapter import GPT2Adapter
from backend.interpretability.discovery.cross_model_circuits import CrossModelCircuitsEngine
from backend.interpretability.discovery.concept_evolution_engine import ConceptEvolutionEngine
from backend.interpretability.discovery.comparative_discovery_engine import ComparativeDiscoveryEngine


# =========================================================================
# 1. Multi-Model Adapter Factory & Architecture Specs
# =========================================================================

def test_model_adapter_factory_creates_all_families():
    """Verify that create_adapter instantiates correct architecture specs for all supported families."""
    models_to_check = {
        "gpt2": (12, 12, 768),
        "gemma-2b": (18, 8, 2048),
        "gemma-7b": (28, 16, 3072),
        "tinyllama": (22, 32, 2048),
        "llama-3-8b": (32, 32, 4096),
        "qwen-2-1.5b": (28, 16, 1536),
        "mistral-7b": (32, 32, 4096),
        "deepseek-r1-1.5b": (28, 16, 1536),
    }

    for model_id, (exp_layers, exp_heads, exp_d_model) in models_to_check.items():
        adapter = create_adapter(model_id, mock_mode=True)
        assert adapter is not None
        assert adapter.spec.num_layers == exp_layers, f"{model_id} layers mismatch"
        assert adapter.spec.num_heads == exp_heads, f"{model_id} heads mismatch"
        assert adapter.spec.d_model == exp_d_model, f"{model_id} d_model mismatch"


# =========================================================================
# 2. Empirical Cross-Model Circuit Alignment
# =========================================================================

def test_cross_model_circuit_alignment_calculation():
    """Verify that CrossModelCircuitsEngine computes dynamic structural, functional, and causal alignment."""
    engine = CrossModelCircuitsEngine()
    res = engine.compare_circuits(
        source_model="gpt2",
        target_model="gemma-2b",
        circuit_type="IOI",
    )

    assert "discovery_id" in res
    assert "alignment" in res
    align = res["alignment"]

    assert align["source_model"] == "gpt2"
    assert align["target_model"] == "gemma-2b"
    assert 0.0 < align["structural_similarity"] <= 1.0
    assert 0.0 <= align["functional_similarity"] <= 1.0
    assert 0.0 < align["causal_similarity"] <= 1.0
    assert 0.0 < align["overall_alignment"] <= 1.0
    assert "provenance" in align


# =========================================================================
# 3. Empirical Concept Evolution & Layer Lineage
# =========================================================================

def test_concept_evolution_layer_progression():
    """Verify that ConceptEvolutionEngine tracks representation progression across network depth."""
    engine = ConceptEvolutionEngine()
    layer_concepts = {
        0: ["subword token representation", "punctuation syntax"],
        4: ["subject entity name", "syntactic role"],
        8: ["capital city entity", "factual relation retrieval"],
        11: ["next token probability", "target logit projection"],
    }

    edges = engine.analyze_layer_progression(layer_concepts)
    assert len(edges) > 0

    for edge in edges:
        assert edge.source_model.startswith("Layer ")
        assert edge.target_model.startswith("Layer ")
        assert edge.type in ["shared", "ancestor", "novel"]
        assert 0.0 < edge.alignment_score <= 1.0


# =========================================================================
# 4. Empirical Cross-Model Semantic Drift Reports
# =========================================================================

def test_cross_model_semantic_drift_report():
    """Verify that ComparativeDiscoveryEngine computes exact bipartite Jaccard drift and persistence."""
    comp_engine = ComparativeDiscoveryEngine()

    run_gpt2 = {
        "model_id": "gpt2",
        "concepts": ["Capital City Retrieval", "Induction Heads", "Syntactic Number Agreement"],
    }
    run_gemma = {
        "model_id": "gemma-2b",
        "concepts": ["Capital City Retrieval", "Chain of Thought", "Induction Heads"],
    }

    report = comp_engine.compare_campaigns(run_gpt2, run_gemma)
    rep_dict = report.to_dict()

    assert rep_dict["model_a"] == "gpt2"
    assert rep_dict["model_b"] == "gemma-2b"
    assert "Capital City Retrieval" in rep_dict["shared_concepts"]
    assert "Induction Heads" in rep_dict["shared_concepts"]
    assert "Chain of Thought" in rep_dict["novel_concepts"]
    assert "Syntactic Number Agreement" in rep_dict["missing_concepts"]
    
    # 2 shared / 4 total unique = 0.5 persistence, 0.5 drift
    assert np.isclose(rep_dict["universality_score"], 0.5, atol=1e-3)
    assert np.isclose(rep_dict["overall_drift"], 0.5, atol=1e-3)

    figs = comp_engine.generate_publication_figures(report)
    assert "evolution_graph" in figs


# =========================================================================
# 5. Golden Rule #2: Multi-Model Dynamic Variation (No Hardcoded Numbers)
# =========================================================================

def test_multi_model_golden_rule_dynamic_variation():
    """Verify that comparing distinct model pairs yields dynamically distinct alignment metrics."""
    engine = CrossModelCircuitsEngine()

    res_gpt_gemma = engine.compare_circuits("gpt2", "gemma-2b")
    res_gpt_llama = engine.compare_circuits("gpt2", "llama-3-8b")

    align_gemma = res_gpt_gemma["alignment"]
    align_llama = res_gpt_llama["alignment"]

    # Gemma (18 layers) and Llama-3-8B (32 layers) have distinct structural depth ratios with GPT-2 (12 layers)
    assert not np.isclose(align_gemma["structural_similarity"], align_llama["structural_similarity"], atol=1e-4)


def test_live_multi_model_adapters_fail_closed_when_offline(monkeypatch):
    """Verify that Gemma, Llama, and Qwen adapters raise RuntimeError in live mode when uninitialized/offline."""
    from backend.science.models.model_manager import ModelManager
    monkeypatch.setattr(ModelManager, "get_model_and_tokenizer", lambda self, repo, **kwargs: (None, None))

    gemma = GemmaAdapter("gemma-2b", mock_mode=False)
    with pytest.raises(RuntimeError, match="uninitialized|Failed"):
        gemma.get_logits("Test prompt")

    llama = LlamaAdapter("tinyllama", mock_mode=False)
    with pytest.raises(RuntimeError, match="uninitialized|Failed"):
        llama.get_activations("Test prompt", layer=0)

    qwen = QwenAdapter("qwen-2-1.5b", mock_mode=False)
    with pytest.raises(RuntimeError, match="uninitialized|Failed"):
        qwen.patch_activation("Test prompt", layer=0, neuron_index=0, patch_value=1.0)

