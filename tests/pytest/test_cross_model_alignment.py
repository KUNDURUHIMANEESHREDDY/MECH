"""Unit and integration tests for CrossModelUniversalityEngine."""

import pytest
from backend.science.comparative.cross_model_alignment import (
    CrossModelUniversalityEngine,
    CrossModelUniversalityReport,
    ModelArchitectureSpec,
    STANDARD_MODEL_ZOO,
)
from backend.core.capability_registry import CapabilityRegistry


def test_model_architecture_spec_depth_normalization():
    spec_gpt2 = STANDARD_MODEL_ZOO["gpt2"]
    spec_llama = STANDARD_MODEL_ZOO["llama-3-8b"]

    assert spec_gpt2.normalize_layer(0) == 0.0
    assert spec_gpt2.normalize_layer(11) == 1.0
    assert 0.45 <= spec_gpt2.normalize_layer(6) <= 0.55

    assert spec_llama.normalize_layer(0) == 0.0
    assert spec_llama.normalize_layer(31) == 1.0
    assert 0.45 <= spec_llama.normalize_layer(16) <= 0.55

    # Denormalization round-trip
    d_norm = spec_gpt2.normalize_layer(6)
    assert spec_gpt2.denormalize_depth(d_norm) == 6


def test_cross_model_universality_sweep():
    engine = CrossModelUniversalityEngine(seed=42)

    dummy_nodes = [
        {"id": "node_L6_MLP", "data": {"label": "L6_MLP", "attribution": 0.52}},
        {"id": "node_L8_H5", "data": {"label": "L8_H5", "attribution": 0.44}},
    ]
    dummy_edges = [
        {"id": "e1", "source": "node_L6_MLP", "target": "node_L8_H5", "data": {"weight": 0.40}},
    ]

    report = engine.evaluate_circuit_universality(
        circuit_nodes=dummy_nodes,
        circuit_edges=dummy_edges,
        reference_model="gpt2",
        target_models=["gpt2", "gemma-2-2b", "llama-3-8b", "qwen-2.5-7b"],
        task_name="Factual Recall (Eiffel Tower)",
    )

    assert isinstance(report, CrossModelUniversalityReport)
    assert report.universality_score >= 0.70
    assert 70 <= report.universality_index <= 100
    assert report.evidence_classification in ("Universal Transformer Primitive", "Cross-Model Conserved Candidate")
    assert report.conjunctive_model_breakdown is not None
    assert len(report.conjunctive_model_breakdown) == 4
    
    # Check all model alignments
    assert len(report.model_alignments) == 4
    for m_name in ["gpt2", "gemma-2-2b", "llama-3-8b", "qwen-2.5-7b"]:
        align = report.model_alignments[m_name]
        assert align.topology_alignment_score >= 0.70
        assert align.faithfulness_reproduced >= 0.75
        assert len(align.aligned_nodes) == len(dummy_nodes)
        assert align.conservation_status in ("CONSERVED_PRIMITIVE", "PARTIAL_CONSERVATION")

    # Check depth heatmap structure
    hm = report.depth_migration_heatmap
    assert "bin_labels" in hm
    assert "models" in hm
    assert len(hm["models"]) == 4



def test_cross_model_tool_in_capability_registry():
    registry = CapabilityRegistry()
    assert registry.get_tool("run_cross_model_universality_sweep") is not None

    res = registry.execute_tool("run_cross_model_universality_sweep", {
        "circuit_nodes": [{"id": "node_L6_MLP", "data": {"label": "L6_MLP", "attribution": 0.5}}],
        "circuit_edges": [],
        "reference_model": "gpt2",
    })

    assert res["status"] == "success"
    assert "universality_report" in res
    ur = res["universality_report"]
    assert "universality_score" in ur
    assert "model_alignments" in ur
    assert "depth_migration_heatmap" in ur
