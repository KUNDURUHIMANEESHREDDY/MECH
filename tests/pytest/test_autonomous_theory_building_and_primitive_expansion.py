"""Unit and integration tests for Phase 35: Autonomous Theory-Building & Primitive Discovery."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief, DependencyType
from backend.discovery.predictive_synthesis_engine import PredictiveSynthesisEngine
from backend.discovery.theory_building_engine import (
    NovelPrimitiveDiscovery,
    TheoryBuildingCycleResult,
    TheoryBuildingEngine,
    TheoryLoopStage,
)


def test_residual_causal_isolation_and_gap_trigger():
    """Verifies that unexplainable behaviors trigger high residual causal mediation (R_residual >= 0.60)."""
    engine = TheoryBuildingEngine()
    residual = engine.isolate_causal_residual("heldout_recursive_syntax_parsing")

    assert residual >= 0.60
    assert residual == pytest.approx(0.88, abs=0.05)


def test_novel_primitive_synthesis_and_causal_verification():
    """Verifies that novel primitives are typed, isolated, and causally verified (Rescue >= 0.85)."""
    engine = TheoryBuildingEngine()
    novel_prim = engine.synthesize_and_verify_novel_primitive(
        primitive_id="PRIM_RECURSIVE_TREE_PARSER",
        primitive_type_name="RECURSIVE_TREE_PARSER",
        behavior_name="heldout_recursive_syntax_parsing",
    )

    assert isinstance(novel_prim, NovelPrimitiveDiscovery)
    assert novel_prim.mediation_rescue_score >= 0.85
    assert novel_prim.logit_delta >= 3.0
    assert "SyntacticHierarchyTree" in novel_prim.output_signature


def test_theory_expansion_and_retrospective_prediction():
    """Verifies that expanding the primitive library yields >= +0.30 Theory Prediction Gain on held-out tasks."""
    engine = TheoryBuildingEngine()
    result = engine.run_theory_building_cycle(
        trigger_behavior="heldout_recursive_syntax_parsing",
        novel_primitive_id="PRIM_RECURSIVE_TREE_PARSER",
    )

    assert isinstance(result, TheoryBuildingCycleResult)
    assert result.theory_prediction_gain >= 0.30
    assert result.post_expansion_mpa == 1.00
    assert len(result.retrospective_behaviors_verified) == 3
    assert "heldout_nested_clause_agreement" in result.retrospective_behaviors_verified


def test_theory_building_dag_lifecycle_transitions():
    """Verifies that Knowledge Gaps are resolved and Novel Primitive claims are registered in the DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    pred_engine = PredictiveSynthesisEngine(claim_graph=claim_graph)

    # 1. Run predictive battery to establish the knowledge gap in the DAG
    report = pred_engine.run_predictive_evaluation_battery()
    gap_claim_id = "CLAIM_KNOWLEDGE_GAP_HELDOUT_RECURSIVE_SYNTAX_PARSING"
    assert gap_claim_id in claim_graph.claims

    # 2. Run theory building cycle on the same DAG
    theory_engine = TheoryBuildingEngine(claim_graph=claim_graph)
    result = theory_engine.run_theory_building_cycle(
        trigger_behavior="heldout_recursive_syntax_parsing",
        novel_primitive_id="PRIM_RECURSIVE_TREE_PARSER",
    )

    # 3. Verify Knowledge Gap statement updated with resolution
    assert "RESOLVED BY NOVEL PRIMITIVE" in claim_graph.claims[gap_claim_id].claim_statement

    # 4. Verify Novel Primitive claim registration
    novel_claim_id = "CLAIM_NOVEL_PRIMITIVE_PRIM_RECURSIVE_TREE_PARSER"
    assert novel_claim_id in claim_graph.claims
    assert claim_graph.claims[novel_claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
