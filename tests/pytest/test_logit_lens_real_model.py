"""Numerical PyTest Suite: Real Model Logit Lens Trajectory Invariants."""

import pytest
import math
from backend.science.logit_lens_engine import LogitLensEngine


def test_logit_lens_real_model_invariants():
    engine = LogitLensEngine(model_id="gpt2")
    res = engine.compute_trajectory(
        prompt="The capital of France is",
        target_token=" Paris",
    )

    assert "trajectory" in res
    assert "maximum_predictive_gain_layer" in res
    assert "target_probability_trajectory" in res
    assert "target_rank_trajectory" in res
    assert "target_logit_trajectory" in res

    traj = res["trajectory"]
    # In GPT-2, hidden states count = 12 layers + 1 embedding = 13
    assert len(traj) >= 12

    # Check mathematical invariants across all layers
    for step in traj:
        assert 0 <= step["layer"] < len(traj)
        assert len(step["predictions"]) >= 1
        # Probability bounds
        assert 0.0 <= step["probability"] <= 1.0
        # Probabilities in top-k are descending
        probs = [p["probability"] for p in step["predictions"]]
        assert probs == sorted(probs, reverse=True)

    # Check maximum gain layer is within valid bounds
    gain_layer = res["maximum_predictive_gain_layer"]
    assert 0 <= gain_layer < len(traj)

    # Check trajectory lists match layer counts
    assert len(res["target_probability_trajectory"]) == len(traj)
    assert len(res["target_rank_trajectory"]) == len(traj)
    assert len(res["target_logit_trajectory"]) == len(traj)
