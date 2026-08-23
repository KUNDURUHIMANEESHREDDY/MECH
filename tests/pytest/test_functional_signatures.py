"""Unit and integration tests for Functional Signature & Behavioral Embedding Engine."""

import pytest
from backend.science.comparative.functional_signatures import (
    ActivationSignature,
    AttentionSignature,
    CausalSignature,
    FunctionalEmbedding,
    FunctionalComponentMatch,
    FunctionalCrossModelEngine,
)


def test_functional_signature_vectors():
    act = ActivationSignature(
        entity_recall_energy=0.88,
        in_context_copy_energy=0.21,
        control_noise_energy=0.08,
        semantic_selectivity=11.0,
    )
    assert len(act.vector) == 4
    assert act.semantic_selectivity == 11.0

    attn = AttentionSignature(
        prefix_matching_score=0.95,
        prev_token_weight=0.89,
        bos_sink_weight=0.10,
        copy_position_weight=0.92,
        dispersion_entropy=0.38,
    )
    assert len(attn.vector) == 5

    cau = CausalSignature(
        factual_ablation_delta_logit=0.84,
        in_context_ablation_delta_logit=0.15,
        relational_steering_efficiency=0.81,
        null_sensitivity=0.04,
    )
    assert len(cau.vector) == 4


def test_functional_embedding_similarity():
    engine = FunctionalCrossModelEngine(seed=42)

    # GPT-2 Induction Head
    gpt2_head = engine.extract_functional_embedding(
        component_id="L8_H5",
        model_name="gpt2",
        layer_index=8,
        component_type="attention_head",
        role="LATE_VALUE_MOVEMENT_INDUCTION",
    )

    # LLaMA-3 Induction Head
    llama_head = engine.extract_functional_embedding(
        component_id="L24_H18",
        model_name="llama-3-8b",
        layer_index=24,
        component_type="attention_head",
        role="LATE_VALUE_MOVEMENT_INDUCTION",
    )

    # GPT-2 Parametric MLP
    gpt2_mlp = engine.extract_functional_embedding(
        component_id="L6_MLP",
        model_name="gpt2",
        layer_index=6,
        component_type="mlp",
        role="MID_PARAMETRIC_MLP_LOOKUP",
    )

    # Cross-Model Induction Heads should have high cosine similarity (>0.90)
    sim_induction = gpt2_head.cosine_similarity(llama_head)
    assert sim_induction >= 0.88

    breakdown = gpt2_head.similarity_breakdown(llama_head)
    assert breakdown["attention_similarity"] >= 0.90
    assert breakdown["causal_similarity"] >= 0.85

    # Heterogeneous components (MLP vs Induction Head) should have low similarity (<0.60)
    sim_hetero = gpt2_mlp.cosine_similarity(gpt2_head)
    assert sim_hetero < 0.60


def test_cross_model_functional_matcher():
    engine = FunctionalCrossModelEngine(seed=42)

    ref_nodes = [
        {"id": "node_L6_MLP", "data": {"label": "L6_MLP"}},
        {"id": "node_L8_H5", "data": {"label": "L8_H5"}},
    ]

    matches = engine.match_circuit_across_models(
        ref_nodes=ref_nodes,
        ref_model="gpt2",
        target_model="llama-3-8b",
        target_total_layers=32,
        target_total_heads=32,
    )

    assert len(matches) == 2
    
    # 1. MLP Match Check
    mlp_match = matches[0]
    assert mlp_match.reference_component_id == "L6_MLP"
    assert "MLP" in mlp_match.matched_component_id
    assert mlp_match.matched_layer in (13, 14, 15, 16)
    assert mlp_match.overall_similarity >= 0.85
    assert len(mlp_match.parallel_cluster_members) >= 1

    # 2. Induction Head Match Check
    head_match = matches[1]
    assert head_match.reference_component_id == "L8_H5"
    assert "H" in head_match.matched_component_id
    assert head_match.matched_layer in (22, 23, 24, 25, 26)
    assert head_match.overall_similarity >= 0.85
    assert head_match.attention_similarity >= 0.90
