"""Unit and integration tests for Phase 21: Claim Dependency Graph & Belief Revision Engine."""

import pytest
import torch

from backend.discovery.claim_dependency_graph import (
    ClaimDependencyGraphEngine,
    ClaimEpistemicBelief,
    DependencyType,
    DynamicClaimNode,
)
from backend.discovery.epistemic_revision_orchestrator import (
    EpistemicRevisionOrchestrator,
    EpistemicRevisionSummary,
)
from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.interfaces import PrecisionProfile


def test_claim_dependency_dag_registration():
    """Verifies that scientific claims and experimental dependencies are registered in the living DAG."""
    engine = ClaimDependencyGraphEngine()

    node = engine.register_claim(
        claim_id="CLAIM_L8_N412_CAPITAL",
        certificate_id="cert_claim_a4b19c8f02",
        circuit_or_component_id="L8_N412",
        behavior_name="french_capital_retrieval",
        claim_statement="Neuron L8_N412 mediates country-capital extraction.",
        dependency_experiment_ids=[
            ("EXP_NODE_ABLATION_E17", DependencyType.NODE_ABLATION),
            ("EXP_4CTRL_BATTERY_E18", DependencyType.CONTROL_BATTERY),
            ("EXP_MEDIATION_E21", DependencyType.MEDIATION_RESCUE),
        ],
    )

    assert isinstance(node, DynamicClaimNode)
    assert node.belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert len(node.dependencies) == 3
    assert "EXP_4CTRL_BATTERY_E18" in engine.experiment_to_claims_index


def test_cascade_invalidation_and_status_transitions():
    """Verifies that invalidating an underlying experiment cascades recursively down dependent claims."""
    engine = ClaimDependencyGraphEngine()

    # 1. Register child claim
    child_node = engine.register_claim(
        claim_id="CLAIM_L8_N412",
        certificate_id="cert_child_01",
        circuit_or_component_id="L8_N412",
        behavior_name="french_capital",
        claim_statement="L8_N412 is a causal country-capital node.",
        dependency_experiment_ids=[("EXP_4CTRL_E18", DependencyType.CONTROL_BATTERY)],
    )

    # 2. Register parent composite circuit claim depending on child
    parent_node = engine.register_claim(
        claim_id="CLAIM_COMPOSITE_CIRCUIT",
        certificate_id="cert_parent_01",
        circuit_or_component_id="L8_N412->H3->L9_N680",
        behavior_name="french_capital",
        claim_statement="Full pathway mediates country-capital transmission.",
        dependency_experiment_ids=[("EXP_PATH_E25", DependencyType.EDGE_PATH_PATCHING)],
    )

    engine.link_claim_dependency("CLAIM_COMPOSITE_CIRCUIT", "CLAIM_L8_N412")

    # 3. Invalidate underlying control battery experiment E18
    affected = engine.invalidate_experiment(
        experiment_id="EXP_4CTRL_E18",
        refutation_rationale="Replication with noise perturbation showed specificity ratio dropped to 1.1x.",
    )

    # Both child and parent claims must be updated in the cascade
    assert "CLAIM_L8_N412" in affected
    assert "CLAIM_COMPOSITE_CIRCUIT" in affected

    assert child_node.belief_status == ClaimEpistemicBelief.EVIDENCE_WEAKENED
    assert parent_node.belief_status == ClaimEpistemicBelief.EVIDENCE_WEAKENED
    assert len(child_node.requested_experiments) > 0
    assert len(parent_node.requested_experiments) > 0

    # Render dependency tree
    tree_md = engine.render_claim_dependency_tree("CLAIM_L8_N412")
    assert "EVIDENCE_WEAKENED" in tree_md
    assert "✗ INVALID" in tree_md
    assert "REQUESTED EXPERIMENTAL AGENDA" in tree_md


def test_immutable_audit_history_preservation():
    """Verifies that belief revisions append immutable audit events without deleting history."""
    engine = ClaimDependencyGraphEngine()

    node = engine.register_claim(
        claim_id="CLAIM_HIST_01",
        certificate_id="cert_hist_01",
        circuit_or_component_id="L7_N100",
        behavior_name="syntax_test",
        claim_statement="L7_N100 tests syntax.",
        dependency_experiment_ids=[("EXP_SYNTAX_01", DependencyType.NODE_ABLATION)],
    )

    # Initial revision
    assert len(node.revision_history) == 1

    # Invalidate
    engine.invalidate_experiment("EXP_SYNTAX_01", "Counterexample failed.")
    assert len(node.revision_history) == 2
    assert node.revision_history[0].new_belief == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert node.revision_history[1].new_belief == ClaimEpistemicBelief.EVIDENCE_WEAKENED

    # Supersede
    engine.supersede_claim("CLAIM_HIST_01", "CLAIM_HIST_02", "Refined by larger multi-layer circuit.")
    assert len(node.revision_history) == 3
    assert node.revision_history[2].new_belief == ClaimEpistemicBelief.SUPERSEDED


def test_epistemic_revision_orchestrator_end_to_end():
    """Verifies end-to-end integration: discovery run -> claim graph registration -> invalidation -> summary."""
    runtime = InMemoryRuntime(
        model_id="gpt2",
        device="cpu",
        precision=PrecisionProfile(weight_dtype="float32", activation_dtype="float32"),
    )
    orchestrator = EpistemicRevisionOrchestrator(runtime=runtime, model_id="gpt2", device="cpu")

    # Run autonomous investigation
    run_res = orchestrator.science_orchestrator.execute_autonomous_investigation(
        behavior_name="country_capital_revision_test",
        clean_prompt="The capital of France is",
        target_token=" Paris",
        corrupted_prompt="The capital of Italy is",
        target_layers=[6, 8, 10],
        pruning_threshold_tau=0.010,
    )

    # Register run to living claim graph
    claim_node = orchestrator.register_scientific_run_to_graph(run_res)
    assert claim_node.belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED

    summary_before = orchestrator.get_epistemic_ledger_summary()
    assert summary_before.active_supported_claims_count == 1
    assert summary_before.evidence_weakened_claims_count == 0

    # Invalidate one dependency
    dep_to_inval = claim_node.dependencies[1].target_dependency_id
    orchestrator.trigger_experiment_invalidation(dep_to_inval, "Replication test failed on out-of-distribution prompts.")

    summary_after = orchestrator.get_epistemic_ledger_summary()
    assert summary_after.active_supported_claims_count == 0
    assert summary_after.evidence_weakened_claims_count == 1
    assert summary_after.total_revision_events >= 1
