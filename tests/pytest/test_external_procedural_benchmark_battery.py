"""Unit and integration tests for Phase 43: External Procedural Benchmark & Scientific Efficiency."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.external_procedural_benchmark_engine import (
    ExternalProceduralBenchmarkEngine,
    ProceduralCircuitGenerator,
    ProceduralDiscoveryScorecard,
)


def test_procedural_generator_synthesizes_unique_unseen_topologies():
    """Verifies that the procedural generator creates structurally unique circuits with distinct SHA-256 hashes."""
    generator = ProceduralCircuitGenerator()
    topo_1 = generator.generate_random_circuit(seed="SEED_ALPHA", depth=4)
    topo_2 = generator.generate_random_circuit(seed="SEED_BETA", depth=6)

    assert topo_1.seed != topo_2.seed
    assert topo_1.depth == 4
    assert topo_2.depth == 6
    assert topo_1.sha256_oracle_hash != topo_2.sha256_oracle_hash
    assert len(topo_1.sha256_oracle_hash) == 64
    assert len(topo_2.sha256_oracle_hash) == 64


def test_zero_knowledge_procedural_discovery_and_efficiency():
    """Verifies that MECH recovers unseen procedural DAGs with Scientific Discovery Efficiency >= 0.80 bits/query."""
    engine = ExternalProceduralBenchmarkEngine()
    scorecard = engine.run_procedural_benchmark(seed="SEED_DISCOVERY_EFFICIENCY_TEST", depth=5)

    assert isinstance(scorecard, ProceduralDiscoveryScorecard)
    assert scorecard.is_certified is True
    assert scorecard.discovery_efficiency_bits_per_query >= 0.80
    assert scorecard.total_queries_executed > 0


def test_procedural_oracle_unsealing_and_ood_scoring():
    """Verifies that the unsealed procedural evaluation satisfies BDS_OOD >= 95.0%, FCR = 0.0%, and error <= 5.0%."""
    engine = ExternalProceduralBenchmarkEngine()
    scorecard = engine.run_procedural_benchmark(seed="SEED_OOD_VALIDATION_2026", depth=6)

    assert scorecard.bds_ood >= 0.95
    assert scorecard.topology_jaccard >= 0.90
    assert scorecard.fcr_pct == 0.0
    assert scorecard.prospective_prediction_error_pct <= 5.0
    assert scorecard.abstention_accuracy_pct == 100.0


def test_procedural_benchmark_dag_certification():
    """Verifies that the procedural master claim is registered as ACTIVE_SUPPORTED in the Living Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = ExternalProceduralBenchmarkEngine(claim_graph=claim_graph)

    scorecard = engine.run_procedural_benchmark(seed="SEED_DAG_INTEGRATION_2026", depth=5)

    claim_id = f"CLAIM_PROCEDURAL_ORACLE_BENCHMARK_{scorecard.seed}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "Procedural Zero-Knowledge Benchmark Certified" in claim_graph.claims[claim_id].claim_statement
