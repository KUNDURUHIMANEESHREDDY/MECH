"""Numerical PyTest Suite: Unembedding Directional Projections (W_U * d_i)."""

import pytest
from backend.science.scientific_circuit_engine import ScientificCircuitEngine


def test_unembedding_projection_numerical_invariants():
    engine = ScientificCircuitEngine(model_id="gpt2")
    features = engine.get_layer_features(layer=4)

    assert len(features) >= 1

    for feat in features:
        # Check empirical confidence score bounds
        assert 0.0 <= feat.confidence_score <= 1.0
        assert 0.0 <= feat.specificity <= 1.0
        assert 0.0 <= feat.consistency <= 1.0
        assert 0.0 <= feat.cross_prompt_stability <= 1.0

        # Check linear logit delta is non-empty and has both positive and negative values
        assert len(feat.linear_logit_delta) >= 2
        vals = list(feat.linear_logit_delta.values())
        assert any(v > 0 for v in vals), "Expected positive boost tokens in directional projection"
        assert any(v < 0 for v in vals), "Expected negative suppressed tokens in directional projection"

        # Check substrate anchors
        assert len(feat.substrate_anchors) >= 1
        anchor = feat.substrate_anchors[0]
        assert anchor.layer == 4
        assert anchor.neuron_idx >= 0
        assert anchor.is_polysemantic is True
