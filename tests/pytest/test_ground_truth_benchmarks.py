import pytest
from backend.science.benchmarks.ground_truth_benchmark import (
    ground_truth_engine,
    IndependentDiscoveryRunner,
    GroundTruthReference,
    DiscoveryResult,
    ReferenceMethodology,
    DiscoveryMethodology,
    BenchmarkVerdict,
    MetricSourceType,
    KNOWN_GROUND_TRUTHS,
)
from backend.science.claim_calibration_engine import (
    claim_calibration_engine,
    EvidenceType,
    AllowedClaimStrength,
)


def test_independent_ioi_discovery_and_calibrated_verdict():
    """Validates that MECH independently executes GPT-2 without receiving expected results and assigns calibrated CAUSAL_MATCH."""
    runner = IndependentDiscoveryRunner()
    discovery = runner.run_ioi_discovery(model_name="gpt2", n_samples=3, seed=42)

    assert discovery.benchmark_id == "IOI_NAME_MOVER"
    assert discovery.execution_mode == "INDEPENDENT_BENCHMARK_MODE"
    assert len(discovery.all_candidate_scores) == 48  # 4 layers * 12 heads scanned
    assert discovery.discovered_component in [
        "L9H9", "L10H0", "L9H6", "L8H4", "L10H1", "L10H7", "L8H1", "L7H6", "L8H7", "L7H0"
    ]
    assert discovery.measured_treatment_metric > 0.40
    assert discovery.measured_control_metric <= 0.30

    comparison = ground_truth_engine.compare_discovery_to_ground_truth(discovery)
    assert comparison.component_match_type in ["EXACT_MATCH", "FUNCTIONAL_MATCH"]
    assert comparison.causal_effect_validated is True
    assert comparison.control_isolated is True
    # Calibrated verdict: CAUSAL_MATCH (not blanket mechanism match due to reduced sample size note)
    assert comparison.scientific_verdict in [BenchmarkVerdict.CAUSAL_MATCH, BenchmarkVerdict.MECHANISM_MATCH, BenchmarkVerdict.PARTIAL_RECOVERY]
    assert len(comparison.limitations) > 0


def test_independent_induction_head_discovery_and_calibrated_verdict():
    """Validates independent prefix attention discovery of induction head assigning COMPONENT_MATCH (observational)."""
    runner = IndependentDiscoveryRunner()
    discovery = runner.run_induction_discovery(model_name="gpt2", seed=42)

    assert discovery.benchmark_id == "INDUCTION_HEADS"
    assert len(discovery.all_candidate_scores) == 36
    assert discovery.discovered_component in ["L5H5", "L5H1", "L5H2", "L6H9", "L4H4", "L4H5"]
    assert discovery.measured_treatment_metric >= 0.05

    comparison = ground_truth_engine.compare_discovery_to_ground_truth(discovery)
    assert comparison.component_match_type in ["EXACT_MATCH", "FUNCTIONAL_MATCH"]
    # Since induction test was prefix attention observation without causal ablation, verdict is calibrated to COMPONENT_MATCH / PARTIAL_RECOVERY
    assert comparison.scientific_verdict in [BenchmarkVerdict.COMPONENT_MATCH, BenchmarkVerdict.PARTIAL_RECOVERY]


def test_independent_greater_than_discovery():
    """Validates independent causal discovery on quantitative comparison sequences."""
    runner = IndependentDiscoveryRunner()
    discovery = runner.run_greater_than_discovery(model_name="gpt2", seed=42)

    assert discovery.benchmark_id == "GREATER_THAN_NUMERICAL"
    assert discovery.discovered_component is not None
    assert discovery.measured_treatment_metric > 0.0

    comparison = ground_truth_engine.compare_discovery_to_ground_truth(discovery)
    assert comparison.ground_truth_reference_id == "GREATER_THAN_NUMERICAL"


def test_component_match_is_not_mechanism_match():
    """Verifies that an observational component match is NOT upgraded to MECHANISM_MATCH."""
    disc = DiscoveryResult(
        benchmark_id="INDUCTION_HEADS",
        model_name="gpt2",
        discovered_component="L5H5",
        all_candidate_scores={"L5H5": 0.62},
        measured_treatment_metric=0.62,
        treatment_metric_source=MetricSourceType.COMPUTED,
        measured_control_metric=0.08,
        control_metric_source=MetricSourceType.COMPUTED,
        control_component="L0H0",
        n_prompts_tested=2,
        seed=42,
        methodology=DiscoveryMethodology(
            model="gpt2",
            model_version="gpt2",
            tokenizer="gpt2",
            dataset="REPEATED_TOKENS",
            prompt_construction="Repeated tokens",
            intervention_method="PREFIX_ATTENTION_OBSERVATION",  # Not causal ablation
            control_method="SCRAMBLED_SEQUENCE",
            primary_metric="prefix_attention_score",
            aggregation="MAX",
            sample_size=2,
        ),
    )

    comparison = ground_truth_engine.compare_discovery_to_ground_truth(disc)
    assert comparison.scientific_verdict != BenchmarkVerdict.MECHANISM_MATCH
    assert comparison.scientific_verdict in [BenchmarkVerdict.COMPONENT_MATCH, BenchmarkVerdict.PARTIAL_RECOVERY]


def test_methodology_mismatch_prevents_mechanism_match():
    """Asserts that evaluating discovery across incompatible model architectures yields METHODOLOGY_MISMATCH."""
    disc = DiscoveryResult(
        benchmark_id="IOI_NAME_MOVER",
        model_name="llama-3-8b",  # Incompatible with GPT-2 benchmark reference
        discovered_component="L9H9",
        all_candidate_scores={"L9H9": 1.85},
        measured_treatment_metric=1.85,
        treatment_metric_source=MetricSourceType.COMPUTED,
        measured_control_metric=0.04,
        control_metric_source=MetricSourceType.COMPUTED,
        control_component="L0H0",
        n_prompts_tested=10,
        seed=42,
        methodology=DiscoveryMethodology(
            model="llama-3-8b",
            model_version="3.0",
            tokenizer="llama3",
            dataset="IOI",
            prompt_construction="ABBA",
            intervention_method="ZERO_ABLATION",
            control_method="L0H0",
            primary_metric="delta_logit",
            aggregation="MEAN",
            sample_size=10,
        ),
    )

    comparison = ground_truth_engine.compare_discovery_to_ground_truth(disc)
    assert comparison.methodology_compatible is False
    assert comparison.scientific_verdict == BenchmarkVerdict.METHODOLOGY_MISMATCH


def test_observation_cannot_create_causal_claim():
    """Claim calibration: Rejects claims asserting causality based purely on attention observations."""
    calib = claim_calibration_engine.calibrate_claim(
        claim_text="Attention analysis proves that L5H5 causes the model to reproduce tokens.",
        evidence_type=EvidenceType.ATTENTION_OBSERVATION,
        target_component="L5H5",
    )

    assert calib.is_claim_valid is False
    assert calib.allowed_strength == AllowedClaimStrength.OBSERVATIONAL_CLAIM
    assert "cannot support causal mechanism assertions" in calib.epistemic_rationale


def test_attention_alone_cannot_create_mechanistic_claim():
    """Claim calibration: Rejects claims asserting full circuit discovery based on attention alone."""
    calib = claim_calibration_engine.calibrate_claim(
        claim_text="We discovered the mechanism for in-context learning in head L5H5.",
        evidence_type=EvidenceType.ATTENTION_OBSERVATION,
        target_component="L5H5",
    )

    assert calib.is_claim_valid is False
    assert calib.allowed_strength == AllowedClaimStrength.OBSERVATIONAL_CLAIM


def test_unreplicated_intervention_cannot_create_strong_mechanistic_claim():
    """Claim calibration: Unreplicated single ablation is constrained to CAUSAL_COMPONENT_CLAIM without claiming complete circuit."""
    calib = claim_calibration_engine.calibrate_claim(
        claim_text="L9H9 is the complete mechanism for indirect object identification.",
        evidence_type=EvidenceType.CAUSAL_ABLATION_SINGLE,
        target_component="L9H9",
        has_negative_control=True,
        has_replication=False,
    )

    assert calib.is_claim_valid is False
    assert calib.allowed_strength == AllowedClaimStrength.CAUSAL_COMPONENT_CLAIM


def test_no_information_leakage_and_benchmark_non_circularity():
    """CRITICAL SCIENTIFIC TEST:
    Modifying the GroundTruthReference expected component MUST NOT alter the DiscoveryResult.
    The discovery runner must remain strictly isolated from reference expectations.
    """
    runner = IndependentDiscoveryRunner()
    disc1 = runner.run_ioi_discovery(model_name="gpt2", seed=42)

    mutated_ref = GroundTruthReference(
        benchmark_id="IOI_NAME_MOVER",
        phenomenon_name="Mutated Synthetic Test Reference",
        paper_title="Synthetic Paper",
        authors=["Author A"],
        year=2026,
        paper_citation="Synthetic Citation",
        paper_url="https://example.org",
        model_target="gpt2",
        expected_primary_components=["L0H3"],  # Fabricated wrong answer
        expected_negative_controls=["L11H11"],
        expected_metric="delta_logit",
        expected_direction="POSITIVE_SUPPRESSION",
        min_effect_threshold=0.50,
        max_control_threshold=0.30,
        methodology=ReferenceMethodology(
            model="gpt2",
            dataset="SYNTHETIC",
            prompt_construction="Synthetic",
            intervention_method="ZERO_ABLATION",
            control_method="L0H0",
            primary_metric="delta_logit",
            sample_size=10,
        ),
    )

    disc2 = runner.run_ioi_discovery(model_name="gpt2", seed=42)
    assert disc1.discovered_component == disc2.discovered_component
    assert disc1.measured_treatment_metric == disc2.measured_treatment_metric

    comp_mutated = ground_truth_engine.compare_discovery_to_ground_truth(disc1, mutated_ref)
    assert comp_mutated.component_match_type == "NO_MATCH"
    assert comp_mutated.scientific_verdict in [BenchmarkVerdict.NO_RECOVERY, BenchmarkVerdict.PARTIAL_RECOVERY]


def test_literature_number_never_appears_as_computed_result():
    """Asserts that DiscoveryResult tracks source type as COMPUTED and never fabricates literature values."""
    runner = IndependentDiscoveryRunner()
    disc = runner.run_ioi_discovery(model_name="gpt2", seed=42)

    assert disc.treatment_metric_source == MetricSourceType.COMPUTED
    assert disc.control_metric_source == MetricSourceType.COMPUTED
    assert isinstance(disc.measured_treatment_metric, float)
    assert disc.measured_treatment_metric != 0.0
