"""Unit and integration tests for Independent Held-Out Validation & Causal Interchange Engine."""

import pytest
from backend.science.comparative.heldout_cross_model_validator import (
    HeldoutTaskType,
    HeldoutProbeResult,
    CrossModelCausalInterchangeResult,
    IndependentUniversalityVerification,
    IndependentCrossModelValidator,
)


def test_independent_cross_model_validation_battery():
    validator = IndependentCrossModelValidator(seed=42)

    # Validate GPT-2 L8_H5 matching LLaMA-3 L24_H18
    verification = validator.run_heldout_behavioral_battery(
        ref_component="L8_H5",
        ref_model="gpt2",
        target_component="L24_H18",
        target_model="llama-3-8b",
        hypothesized_cosine=0.94,
    )

    assert isinstance(verification, IndependentUniversalityVerification)
    assert verification.is_circularity_prevented is True
    assert len(verification.heldout_probe_results) == 4

    # Verify all 4 unseen tasks are present
    task_types = [p.task_type for p in verification.heldout_probe_results]
    assert HeldoutTaskType.SYNTACTIC_AGREEMENT_DISTRACTOR.value in task_types
    assert HeldoutTaskType.CROSSLINGUAL_ENTITY_TRANSFER.value in task_types
    assert HeldoutTaskType.COMPOSITIONAL_TWOHOP_REASONING.value in task_types
    assert HeldoutTaskType.ALGORITHMIC_PERMUTATION.value in task_types

    # Concordance and Causal Interchange Checks
    assert verification.mean_heldout_concordance >= 0.80
    assert verification.causal_interchange.interchangeability_score >= 0.75
    assert verification.causal_interchange.is_causally_interchangeable is True

    # Calibrated Epistemic Verdict
    assert verification.calibrated_verdict == "MECHANISTICALLY_CONSERVED"
    assert "PASSED INDEPENDENT FALSIFICATION" in verification.epistemic_warning


def test_independent_validation_serializability():
    validator = IndependentCrossModelValidator(seed=42)

    verification = validator.run_heldout_behavioral_battery(
        ref_component="L6_MLP",
        ref_model="gpt2",
        target_component="L14_MLP",
        target_model="llama-3-8b",
        hypothesized_cosine=0.92,
    )

    d = verification.to_dict()
    assert d["reference_component"] == "L6_MLP"
    assert d["matched_component"] == "L14_MLP"
    assert "heldout_probe_results" in d
    assert len(d["heldout_probe_results"]) == 4
    assert "causal_interchange" in d
    assert d["causal_interchange"]["interchangeability_score"] > 0
