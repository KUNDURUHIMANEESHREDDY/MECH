"""Unit and integration tests for Phase 33: Compositional Computational Primitive Library & Multi-Program Composition."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief, DependencyType
from backend.discovery.compositional_primitive_engine import (
    CompositionalProgram,
    CompositionalProgramCertificate,
    CompositionalPrimitiveEngine,
    ComputationalPrimitive,
    PrimitiveType,
)


def test_computational_primitive_library_registration():
    """Verifies that the primitive library catalog registers typed, highly faithful computational primitives."""
    engine = CompositionalPrimitiveEngine()
    lib = engine.primitive_library

    assert len(lib) >= 6
    assert "PRIM_EXTRACT_ENTITY" in lib
    assert "PRIM_FACT_LOOKUP" in lib
    assert "PRIM_CONTEXTUAL_ROUTER" in lib
    assert "PRIM_COPY_SUPPRESSOR" in lib
    assert "PRIM_NUMERICAL_COMPARATOR" in lib
    assert "PRIM_UNEMBED_PROJECTION" in lib

    for pid, prim in lib.items():
        assert isinstance(prim, ComputationalPrimitive)
        assert prim.fidelity_score >= 0.90
        assert "->" in prim.input_signature or "->" in prim.output_signature


def test_multi_hop_factual_composition_synthesis():
    """Verifies multi-hop reasoning program synthesis: Extract -> Lookup(Birthplace) -> Lookup(Capital) -> Router -> Unembed."""
    engine = CompositionalPrimitiveEngine()
    program = engine.compose_program(
        behavior_name="multihop_birthplace_capital",
        description="Extracts person entity, queries birthplace country, queries capital of country, routes, unembeds.",
        primitive_ids=[
            "PRIM_EXTRACT_ENTITY",
            "PRIM_FACT_LOOKUP",
            "PRIM_FACT_LOOKUP",
            "PRIM_CONTEXTUAL_ROUTER",
            "PRIM_UNEMBED_PROJECTION",
        ],
    )

    assert isinstance(program, CompositionalProgram)
    assert len(program.primitive_sequence) == 5
    assert program.execution_fidelity_score >= 0.95

    # Simulate execution on Einstein query
    trace = engine.execute_compositional_trace(
        program,
        initial_input={"subject": "Einstein", "relation_chain": ["birthplace", "capital"]},
    )
    assert len(trace["EXECUTION_STEPS"]) == 5
    assert trace["FINAL_PREDICTION"] == "Berlin"



def test_diverse_behavior_composition_battery():
    """Verifies that diverse behaviors (IOI, Greater-Than, Factual) are synthesized from the shared primitive library."""
    engine = CompositionalPrimitiveEngine()

    # 1. IOI Program
    ioi_prog = engine.compose_program(
        behavior_name="indirect_object_identification",
        description="Extract names, suppress repeated name, route indirect object, unembed.",
        primitive_ids=[
            "PRIM_EXTRACT_ENTITY",
            "PRIM_COPY_SUPPRESSOR",
            "PRIM_CONTEXTUAL_ROUTER",
            "PRIM_UNEMBED_PROJECTION",
        ],
    )
    ioi_trace = engine.execute_compositional_trace(
        ioi_prog,
        initial_input={"subject": "John", "distractor": "John", "target": "Mary"},
    )
    assert ioi_trace["FINAL_PREDICTION"] == "Mary"

    # 2. Greater-Than Program
    gt_prog = engine.compose_program(
        behavior_name="greater_than_comparison",
        description="Extract years, compare magnitude, route indicator, unembed.",
        primitive_ids=[
            "PRIM_EXTRACT_ENTITY",
            "PRIM_NUMERICAL_COMPARATOR",
            "PRIM_CONTEXTUAL_ROUTER",
            "PRIM_UNEMBED_PROJECTION",
        ],
    )
    gt_trace = engine.execute_compositional_trace(
        gt_prog,
        initial_input={"val1": 1995, "val2": 1990},
    )
    assert gt_trace["FINAL_PREDICTION"] == "GREATER"


def test_identifiability_boundary_and_dag_integration():
    """Verifies explicit identifiability boundary declaration, SHA-256 seal, and Claim DAG registration."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = CompositionalPrimitiveEngine(claim_graph=claim_graph)

    program = engine.compose_program(
        behavior_name="country_capital",
        description="Standard single-hop country-capital retrieval.",
        primitive_ids=[
            "PRIM_EXTRACT_ENTITY",
            "PRIM_FACT_LOOKUP",
            "PRIM_CONTEXTUAL_ROUTER",
            "PRIM_UNEMBED_PROJECTION",
        ],
    )

    cert = engine.certify_compositional_program(program)

    assert isinstance(cert, CompositionalProgramCertificate)
    assert len(cert.sha256_seal) == 64
    assert cert.program.identifiability_boundary.is_uniquely_identified_in_tested_space is True
    assert len(cert.program.identifiability_boundary.tested_perturbations) == 3

    # Verify DAG Claim Registration
    comp_claim_id = "CLAIM_COMPOSITE_COUNTRY_CAPITAL"
    assert comp_claim_id in claim_graph.claims
    assert claim_graph.claims[comp_claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED

    # Verify that dependencies point to foundational primitives via PRIMITIVE_COMPOSITION
    deps = claim_graph.claims[comp_claim_id].dependencies
    assert len(deps) == 4
    assert any(d.dependency_type == DependencyType.PRIMITIVE_COMPOSITION for d in deps)
