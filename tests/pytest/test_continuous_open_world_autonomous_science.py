"""Unit and integration tests for Phase 62: Continuous Open-World Autonomous Epistemic Engine & Self-Expanding Science Loop."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.epistemic_state_manager import EpistemicDecisionRecord, EpistemicOperationalState, EpistemicStateManager
from backend.discovery.open_question_generator import OpenQuestionGenerator, QuestionTriggerType, ScientificQuestion
from backend.discovery.open_world_science_engine import OpenWorldScienceEngine, Phase62AutonomousScienceCertificate
from backend.discovery.primitive_gap_detector import DiscoveredPrimitiveRecord, PrimitiveGapDetector
from backend.discovery.scientific_memory_index import MemoryRecord, ScientificMemoryIndex
from backend.discovery.scientific_priority_engine import PrioritizedQuestionRecord, ScientificPriorityEngine


def test_scientific_question_generation_and_priority():
    """Verifies autonomous formulation and multi-factor expected utility prioritization of scientific questions."""
    q_gen = OpenQuestionGenerator()
    questions = q_gen.generate_candidate_questions(
        unexplained_residuals=[{"target": "Mixtral", "residual": -0.30}],
        theory_disagreements=[{"topic": "polysemanticity", "variance": 0.85}],
        unresolved_boundaries=[{"substrate": "Mamba-2", "status": "unmapped"}],
    )

    assert len(questions) == 3
    for q in questions:
        assert isinstance(q, ScientificQuestion)
        assert q.expected_information_gain > 0.70
        assert q.estimated_experiment_cost > 0.0

    p_engine = ScientificPriorityEngine()
    prioritized = p_engine.prioritize_questions(questions)

    assert len(prioritized) == 3
    assert prioritized[0].rank == 1
    assert prioritized[0].utility_score >= prioritized[1].utility_score


def test_primitive_gap_detection_and_synthesis():
    """Verifies that persistent residuals trigger novel primitive synthesis and causal verification."""
    gap_detector = PrimitiveGapDetector()

    # Small residual (< 0.20) should not synthesize new primitive
    prim_small = gap_detector.detect_and_synthesize_primitive(
        residual_magnitude=0.08,
        target_substrate="Dense Transformer",
        unexplained_phenomenon="Minor linear offset",
    )
    assert prim_small is None

    # Persistent large residual (>= 0.20) triggers synthesis
    prim_large = gap_detector.detect_and_synthesize_primitive(
        residual_magnitude=-0.30,
        target_substrate="Sparse MoE Transformers",
        unexplained_phenomenon="Dynamic routing dispersion attenuation",
    )

    assert isinstance(prim_large, DiscoveredPrimitiveRecord)
    assert prim_large.is_causally_verified is True
    assert prim_large.causal_explanatory_gain >= 0.80
    assert prim_large.verification_p_value < 0.001


def test_epistemic_state_manager_and_memory_index():
    """Verifies global decision routing and associative deduplication memory."""
    state_mgr = EpistemicStateManager()

    # 1. Budget exhausted -> HALT
    dec_halt = state_mgr.determine_next_action(0.10, 0.02, False, remaining_budget=0.05, max_eig=0.50)
    assert dec_halt.state == EpistemicOperationalState.HALT_BUDGET_EXHAUSTION

    # 2. Out-of-domain -> ABSTAIN
    dec_abstain = state_mgr.determine_next_action(0.10, 0.02, is_out_of_domain=True, remaining_budget=5.0, max_eig=0.50)
    assert dec_abstain.state == EpistemicOperationalState.ABSTAIN_BOUNDARY_MAP

    # 3. Large residual -> PRIMITIVE_GAP
    dec_gap = state_mgr.determine_next_action(0.10, -0.35, False, remaining_budget=5.0, max_eig=0.80)
    assert dec_gap.state == EpistemicOperationalState.SYNTHESIZE_PRIMITIVE_GAP

    # Memory deduplication
    memory = ScientificMemoryIndex()
    memory.record_investigation(
        question_statement="Does routing modulate delta z?",
        experiment_id="EXP_1",
        observed_outcome=0.81,
        residual_recorded=0.01,
        theories_falsified=[],
    )

    assert memory.is_experiment_redundant("EXP_1") is True
    assert memory.is_experiment_redundant("EXP_2") is False


def test_full_closed_loop_autonomous_science_execution():
    """Verifies that the OpenWorldScienceEngine executes >= 3 continuous discovery cycles and updates the Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = OpenWorldScienceEngine(claim_graph=claim_graph)

    cert = engine.run_autonomous_science_loop(max_cycles=3)

    assert isinstance(cert, Phase62AutonomousScienceCertificate)
    assert cert.is_fully_certified is True
    assert cert.autonomous_cycles_executed == 3
    assert cert.autonomous_question_quality >= 0.90
    assert cert.theory_improvement_rate >= 0.90
    assert cert.primitive_gap_precision >= 0.90
    assert cert.redundant_experiment_rate <= 0.10
    assert cert.autonomous_abstention_quality >= 0.90
    assert len(cert.discovered_primitives) >= 1

    claim_id = "CLAIM_AUTONOMOUS_OPEN_WORLD_SCIENCE"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
