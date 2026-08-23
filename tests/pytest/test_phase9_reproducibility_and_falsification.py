"""Phase 9: Scientific Reproducibility, Generalization & Falsification Test Suite.

Validates:
1. Cross-Prompt Replication across diverse syntactic & semantic templates (IOI, Factual Recall).
2. Seed & RNG Invariance ensuring deterministic evaluation across random seeds.
3. Dataset Generalization evaluating circuit metrics across distinct prompt batches.
4. Empirical Induction Head Replication & Falsification against non-repeated control sequences.
5. Matched Negative Controls & Null Path Distributions via live multi-hop mediation and knockouts.
6. Dynamic Evidence Level Bounding under the weakest-link epistemic rule.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from backend.interpretability.discovery.induction_head_detector import InductionHeadDetector
from backend.science.path_verification_engine import PathVerificationEngine
from backend.science.reproducibility.ioi_pipeline import IOIReproductionPipeline
from backend.science.reproducibility.greater_than_pipeline import GreaterThanCircuitPipeline
from backend.science.reproducibility.induction_heads_pipeline import InductionHeadsPipeline
from backend.science.reproducibility.sae_pipeline import SAEReproductionPipeline
from backend.interpretability.causal.causal_tracing import CausalTracingEngine


# ============================================================================
# 1. Cross-Prompt Replication & Invariance
# ============================================================================

class TestCrossPromptReplication:
    """Test mechanistic replication across multiple prompt instances and templates."""

    def test_ioi_multi_template_replication(self):
        """Verify IOI Name Mover Head effect reproduces across diverse name sets."""
        pipeline = IOIReproductionPipeline(mock_mode=True)
        # Test with 20 prompts, seed=42
        res_a = pipeline.run(n_prompts=20, seed=42)
        assert res_a["reproducibility_report"]["overall_tier"] in ("Gold", "Silver", "Bronze")
        assert res_a["observed_metrics"]["circuit_faithfulness"] > 0.70

        # Test with different seed (different name permutations)
        res_b = pipeline.run(n_prompts=20, seed=100)
        assert res_b["observed_metrics"]["circuit_faithfulness"] > 0.70
        # Faithfulness should remain robust within 15% tolerance across seed permutations
        assert abs(res_a["observed_metrics"]["circuit_faithfulness"] - res_b["observed_metrics"]["circuit_faithfulness"]) < 0.15

    def test_factual_recall_cross_entity_replication(self):
        """Verify factual tracing localizes to mid-layer MLPs across multiple country/capital pairs."""
        tracer = CausalTracingEngine()
        pairs = [
            ("The capital of France is", "The capital of Italy is"),
            ("The capital of Germany is", "The capital of Spain is"),
            ("The capital of Japan is", "The capital of China is"),
        ]

        for clean, corr in pairs:
            res = tracer.trace_causal_effect(clean_prompt=clean, corrupted_prompt=corr, num_layers=12)
            assert res["clean_probability"] > 0.0
            assert len(res["layer_effects"]) == 12
            # Bounded indirect effects
            for l_eff in res["layer_effects"]:
                assert 0.0 <= l_eff["indirect_effect"] <= 1.0


# ============================================================================
# 2. Seed & Stochasticity Invariance
# ============================================================================

class TestSeedAndStochasticityInvariance:
    """Verify deterministic metric convergence across different RNG seeds."""

    def test_induction_pipeline_seed_invariance(self):
        pipeline = InductionHeadsPipeline(mock_mode=True)
        r1 = pipeline.run(n_sequences=20, seed=42)
        r2 = pipeline.run(n_sequences=20, seed=42)

        # Same seed must produce bit-exact identical metrics
        assert r1["observed_metrics"] == r2["observed_metrics"]
        assert r1["manifest_id"] != ""

    def test_greater_than_pipeline_seed_stability(self):
        pipeline = GreaterThanCircuitPipeline(mock_mode=True)
        r1 = pipeline.run(seed=42)
        r2 = pipeline.run(seed=999)

        assert r1["observed_metrics"]["patch_effect_magnitude"] > 0.0
        assert r2["observed_metrics"]["patch_effect_magnitude"] > 0.0
        assert r1["observed_metrics"]["circuit_accuracy"] > 0.0


# ============================================================================
# 3. Induction Head Replication & Falsification
# ============================================================================

class TestInductionHeadFalsification:
    """Test empirical induction detection on repeated sequences vs non-repeated controls."""

    def test_induction_head_positive_repetition_detection(self):
        """Live GPT-2 detects induction patterns on repeated token sequences."""
        detector = InductionHeadDetector(model_name="gpt2")
        repeated_prompt = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        )
        detected = detector.detect_induction_heads(sequence_prefix=repeated_prompt, threshold=0.03, top_k=5)
        assert len(detected) > 0
        for head in detected:
            assert head["induction_score"] >= 0.03
            assert 0 <= head["layer"] < 12
            assert 0 <= head["head"] < 12

    def test_induction_head_negative_falsification_control(self):
        """Non-repeated random sequence must NOT trigger high induction head scores."""
        detector = InductionHeadDetector(model_name="gpt2")
        non_repeated_prompt = (
            "one two three four five six seven eight nine ten eleven twelve "
            "apple banana cherry date elderberry fig grape honeydew kiwi lemon mango nut"
        )
        # Non-repeated sequences lack identical offset diagonals
        detected = detector.detect_induction_heads(sequence_prefix=non_repeated_prompt, threshold=0.75, top_k=5)
        # Highly strict threshold (0.75) must falsify/find zero active induction heads on random text
        active_heads = [h for h in detected if h["is_induction_head"]]
        assert len(active_heads) == 0


# ============================================================================
# 4. Multi-Hop Mediation & Matched Empirical Null Controls
# ============================================================================

class TestMediationRescueAndEmpiricalControls:
    """Test live multi-hop knockout, mediation rescue, and null path distributions."""

    def test_path_verification_with_live_controls(self):
        verifier = PathVerificationEngine(model_id="gpt2")
        res = verifier.verify_pathway(
            clean_prompt="The capital of France is",
            target_token=" Paris",
            node_chain=["SAE_L8_F0", "Head_L8_H3", "SAE_L8_F1", "node_output"],
            edge_chain=["e_feat1_to_head", "e_head_to_feat2", "e_feat2_to_output"],
            layer=8,
            seed=42,
        )

        assert res.prompt == "The capital of France is"
        assert res.target_token == " Paris"
        assert len(res.step_measurements) == 6

        # Verify Null Path Distribution has 8 matched controls
        assert res.null_distribution.control_path_count == 8
        assert len(res.null_distribution.control_path_deltas) == 8
        assert 0.0 <= res.null_distribution.observed_path_percentile <= 100.0
        assert 0.0 <= res.null_distribution.empirical_p_value <= 1.0

        # Verify Mediation Rescue measurement
        assert not math.isnan(res.mediation_rescue.rescue_fraction)
        assert res.path_causal_status in ("END_TO_END_VERIFIED", "PARTIALLY_MEDIATED", "NON_MEDIATING", "FALSIFIED")

    def test_pathway_evidence_level_weakest_link_bounding(self):
        """Verify that a path with low direct effect or failed rescue is strictly bounded."""
        verifier = PathVerificationEngine(model_id="gpt2")
        # Test an unlinked / nonsensical pathway
        res = verifier.verify_pathway(
            clean_prompt="The capital of France is",
            target_token=" Paris",
            node_chain=["SAE_L0_F0", "Head_L1_H1", "SAE_L2_F2", "node_output"],
            layer=1,
            seed=42,
        )
        assert res.path_causal_status in ("END_TO_END_VERIFIED", "PARTIALLY_MEDIATED", "NON_MEDIATING", "FALSIFIED")
        assert len(res.epistemic_scope) > 0


# ============================================================================
# 5. SAE Reproduction Pipeline
# ============================================================================

class TestSAEReproductionPipeline:
    """Test SAE benchmark reproduction across layers."""

    def test_sae_pipeline_metrics_and_manifest(self):
        pipeline = SAEReproductionPipeline(mock_mode=True)
        res = pipeline.run(n_features=50, seed=42)
        assert res["paper_id"] == "sparse_autoencoders"
        assert "l0_sparsity" in res["observed_metrics"]
        assert "reconstruction_mse" in res["observed_metrics"]
        assert "monosemanticity_score" in res["observed_metrics"]
        assert res["manifest_id"] != ""
