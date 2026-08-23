"""Unit and integration tests for Phase 16: Automated Feature Interpretation & Causal Falsification."""

import pytest
import torch

from backend.discovery.automated_interpretation import (
    AutomatedInterpretationEngine,
    AutomatedInterpretationHypothesis,
    EmpiricalInterpretationReport,
    EpistemicInterpretationStatus,
    FalsificationCategory,
    FalsificationTestPrompt,
    InterpretationTargetType,
)
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PrecisionProfile
from backend.runtime.out_of_core_runtime import OutOfCoreRuntime


def test_falsification_suite_generation():
    """Verifies that the engine constructs a structured 6-category falsification test suite."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    engine = AutomatedInterpretationEngine(runtime=runtime, model_id="gpt2", device="cpu")

    hypo = engine.generate_hypothesis(layer=8, component_index=412)

    assert hypo.hypothesis_id.startswith("hypo_L8_N412_")
    assert hypo.layer == 8
    assert hypo.component_index == 412
    assert len(hypo.top_projected_tokens) >= 1
    assert len(hypo.falsification_suite) == 6

    # Verify that all 6 categories are represented
    categories = {t.category for t in hypo.falsification_suite}
    assert FalsificationCategory.POSITIVE_TEST in categories
    assert FalsificationCategory.RELATED_REPHRASE in categories
    assert FalsificationCategory.NEGATIVE_DISTRACTOR in categories
    assert FalsificationCategory.COUNTEREXAMPLE in categories
    assert FalsificationCategory.LEXICAL_CONTROL in categories
    assert FalsificationCategory.SEMANTIC_CONTROL in categories


def test_real_neuron_automated_interpretation_and_scoring():
    """Verifies that live forward passes and interventions produce a quantified empirical interpretation report."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    engine = AutomatedInterpretationEngine(runtime=runtime, model_id="gpt2", device="cpu")

    report = engine.evaluate_interpretation(layer=8, component_index=412)

    assert isinstance(report, EmpiricalInterpretationReport)
    assert report.report_id.startswith("report_interp_")
    assert report.model_id == "gpt2"
    assert len(report.test_records) == 6

    # Ensure measurements are dynamic floats
    assert isinstance(report.activation_contrast_ratio, float)
    assert isinstance(report.causal_alignment_score, float)
    assert isinstance(report.control_specificity_ratio, float)
    assert isinstance(report.overall_interpretation_score, float)

    # Epistemic status must be valid
    assert report.epistemic_status in (
        EpistemicInterpretationStatus.VERIFIED_INTERPRETATION,
        EpistemicInterpretationStatus.SUPPORTED_INTERPRETATION,
        EpistemicInterpretationStatus.CANDIDATE_INTERPRETATION,
        EpistemicInterpretationStatus.FALSIFIED_INTERPRETATION,
    )
    assert len(report.falsification_summary) > 10


def test_deliberate_falsification_of_spurious_hypothesis():
    """Verifies that an incorrect/spurious semantic label is falsified by empirical testing."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    engine = AutomatedInterpretationEngine(runtime=runtime, model_id="gpt2", device="cpu")

    # Construct a bogus hypothesis with inverted/mismatched tests
    bogus_suite = [
        FalsificationTestPrompt(
            prompt_id="bogus_01",
            category=FalsificationCategory.POSITIVE_TEST,
            prompt_text="Quantum supersymmetry explains dark matter through",
            target_token=" gravitons",
            expected_activation_level="HIGH",
            expected_causal_delta_sign="NEGATIVE",
        ),
        FalsificationTestPrompt(
            prompt_id="bogus_02",
            category=FalsificationCategory.POSITIVE_TEST,
            prompt_text="Non-abelian gauge theories require symmetry breaking via",
            target_token=" tachyons",
            expected_activation_level="HIGH",
            expected_causal_delta_sign="NEGATIVE",
        ),
        FalsificationTestPrompt(
            prompt_id="bogus_neg_01",
            category=FalsificationCategory.NEGATIVE_DISTRACTOR,
            prompt_text="The capital of France is",
            target_token=" Paris",
            expected_activation_level="LOW",
            expected_causal_delta_sign="NEUTRAL",
        ),
    ]

    bogus_hypo = AutomatedInterpretationHypothesis(
        hypothesis_id="hypo_bogus_quantum",
        target_type=InterpretationTargetType.NEURON,
        layer=8,
        component_index=412,
        candidate_label="Theoretical High-Energy Particle Physics",
        detailed_explanation="Neuron L8_N412 represents advanced non-abelian gauge quantum physics.",
        top_projected_tokens=["graviton", "tachyon"],
        falsification_suite=bogus_suite,
    )

    report = engine.evaluate_interpretation(layer=8, component_index=412, hypothesis=bogus_hypo)

    # Bogus physics hypothesis on a standard geography/factual neuron should yield low score and be falsified or low candidate
    assert report.overall_interpretation_score < 0.70
    assert report.epistemic_status in (
        EpistemicInterpretationStatus.CANDIDATE_INTERPRETATION,
        EpistemicInterpretationStatus.FALSIFIED_INTERPRETATION,
    )



def test_out_of_core_automated_interpretation():
    """Verifies that automated interpretation runs seamlessly under Out-of-Core memory constraints."""
    runtime = OutOfCoreRuntime(
        model_id="gpt2",
        device="cpu",
        max_active_layers=1,
        precision=PrecisionProfile(weight_dtype="float16", activation_dtype="float16"),
    )
    engine = AutomatedInterpretationEngine(runtime=runtime, model_id="gpt2", device="cpu")

    report = engine.evaluate_interpretation(layer=6, component_index=184)

    assert isinstance(report, EmpiricalInterpretationReport)
    assert report.model_id == "gpt2"
    assert len(report.test_records) == 6
    assert report.overall_interpretation_score >= 0.0
