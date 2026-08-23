"""Unit and integration tests for Phase 59: Independent Adversarial Prospective Replication & Scientific Boundary Engine."""

import pytest

from backend.discovery.adversarial_claim_policy import AdversarialClaimPolicy, AdversarialEpistemicStatus
from backend.discovery.adversarial_generalization_calibrator import (
    AdversarialGeneralizationCalibrator,
    AdversarialReplicationCertificate,
)
from backend.discovery.adversarial_holdout_selector import AdversarialHoldoutSelector, SealedAdversarialPackage
from backend.discovery.adversarial_replication_oracle import (
    AdversarialChallengeCase,
    AdversarialReplicationOracle,
    AdversarialStressDimension,
)
from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.failure_budget_controller import (
    FailureBudgetController,
    FailureEvaluationRecord,
    FailureRegime,
)
from backend.discovery.mechanism_boundary_mapper import ClaimBoundaryRecord, MechanismBoundaryMapper


def test_adversarial_oracle_stress_generation():
    """Verifies that the independent adversary synthesizes challenges maximizing stress dimensions."""
    oracle = AdversarialReplicationOracle()
    cases = oracle.generate_adversarial_battery()

    assert len(cases) == 4
    for c in cases:
        assert isinstance(c, AdversarialChallengeCase)
        assert 0.0 <= c.adversarial_difficulty_d_adv <= 1.0
        assert len(c.expected_failure_mode) > 0

    dimensions = [c.stress_dimension for c in cases]
    assert AdversarialStressDimension.POLYSEMANTIC_SUPERPOSITION in dimensions
    assert AdversarialStressDimension.BOUNDARY_SSM_HYBRID in dimensions

    # Verify blind sealing
    selector = AdversarialHoldoutSelector()
    sealed_pkg = selector.seal_adversarial_battery(cases)
    assert isinstance(sealed_pkg, SealedAdversarialPackage)
    assert len(sealed_pkg.blind_challenges) == 4

    for ch in sealed_pkg.blind_challenges:
        assert "expected_failure_mode" not in ch
        assert "is_out_of_domain" not in ch


def test_failure_budget_and_critical_fcr_bound():
    """Verifies the Failure Budget Controller bounds False-Confidence Rate (FCR_adv <= 2.0%)."""
    controller = FailureBudgetController(max_fcr=0.02)

    # 1. Robust generalization case
    rec_robust = controller.classify_outcome(
        case_id="CASE_1",
        predicted_rescue=0.810,
        empirical_rescue=0.815,
        abstention_probability=0.05,
        is_out_of_domain=False,
    )
    assert rec_robust.regime == FailureRegime.ROBUST_GENERALIZATION
    assert rec_robust.is_scientifically_acceptable is True

    # 2. Correct boundary detection (SSM)
    rec_boundary = controller.classify_outcome(
        case_id="CASE_2",
        predicted_rescue=0.000,
        empirical_rescue=0.000,
        abstention_probability=0.98,
        is_out_of_domain=True,
    )
    assert rec_boundary.regime == FailureRegime.CORRECT_BOUNDARY_DETECTION
    assert rec_boundary.is_scientifically_acceptable is True

    # 3. Calibration failure (uncertain, not critical)
    rec_calib = controller.classify_outcome(
        case_id="CASE_3",
        predicted_rescue=0.550,
        empirical_rescue=0.420,
        abstention_probability=0.35,
        is_out_of_domain=False,
    )
    assert rec_calib.regime == FailureRegime.CALIBRATION_FAILURE
    assert rec_calib.is_scientifically_acceptable is True

    fcr, is_ok = controller.evaluate_failure_budget([rec_robust, rec_boundary, rec_calib])
    assert fcr == 0.0
    assert is_ok is True


def test_mechanism_boundary_mapping_on_abstention():
    """Verifies that abstentions and failures generate structured boundaries with actionable research agendas."""
    oracle = AdversarialReplicationOracle()
    cases = oracle.generate_adversarial_battery()
    ssm_case = [c for c in cases if c.is_out_of_domain][0]

    controller = FailureBudgetController()
    eval_rec = controller.classify_outcome(
        case_id=ssm_case.case_id,
        predicted_rescue=0.0,
        empirical_rescue=0.0,
        abstention_probability=0.95,
        is_out_of_domain=True,
    )

    mapper = MechanismBoundaryMapper()
    boundary = mapper.map_boundary_case(ssm_case, eval_rec)

    assert isinstance(boundary, ClaimBoundaryRecord)
    assert boundary.architecture_family == "Transformer-to-SSM"
    assert "recurrent state kernel" in boundary.recommended_calibration_experiment
    assert boundary.predicted_uncertainty == 0.95


def test_adversarial_calibration_and_claim_dag_policy():
    """Verifies that the AdversarialGeneralizationCalibrator issues certificates and updates the Claim DAG."""
    oracle = AdversarialReplicationOracle()
    cases = oracle.generate_adversarial_battery()

    controller = FailureBudgetController()
    eval_1 = controller.classify_outcome("C1", 0.810, 0.805, 0.05, False)
    eval_2 = controller.classify_outcome("C2", 0.790, 0.782, 0.08, False)
    eval_3 = controller.classify_outcome("C3", 0.000, 0.000, 0.98, True)
    eval_4 = controller.classify_outcome("C4", 0.620, 0.615, 0.12, False)

    mapper = MechanismBoundaryMapper()
    boundaries = [mapper.map_boundary_case(c, e) for c, e in zip(cases, [eval_1, eval_2, eval_3, eval_4])]

    calibrator = AdversarialGeneralizationCalibrator()
    cert = calibrator.evaluate_adversarial_battery(cases, [eval_1, eval_2, eval_3, eval_4], boundaries)

    assert isinstance(cert, AdversarialReplicationCertificate)
    assert cert.is_adversarially_certified is True
    assert cert.fcr_adv <= 0.02
    assert cert.bdr_score >= 0.90
    assert cert.ac_adv_score >= 0.90
    assert cert.adversarial_robustness_ar >= 0.85

    claim_graph = ClaimDependencyGraphEngine()
    policy = AdversarialClaimPolicy(claim_graph=claim_graph)

    claim_rec = policy.register_adversarial_claim(
        claim_id="CLAIM_ADV_REPLICATION_TRIAL",
        parent_generalization_claim_id="CLAIM_GEN_CROSSED_ARITHMETIC",
        fcr_adv=cert.fcr_adv,
        bdr_score=cert.bdr_score,
        ar_score=cert.adversarial_robustness_ar,
    )

    assert claim_rec.status == AdversarialEpistemicStatus.ADVERSARIAL_SUPPORTED
    assert claim_graph.claims["CLAIM_ADV_REPLICATION_TRIAL"].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
