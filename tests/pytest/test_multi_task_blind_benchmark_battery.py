"""Unit and integration tests for Phase 42: Multi-Task Blind Benchmark Battery."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.multi_task_blind_benchmark_engine import (
    MultiTaskBlindBenchmarkEngine,
    MultiTaskBlindOracleSuite,
    MultiTaskBlindReport,
)


def test_multi_task_oracle_suite_initialization_and_sealing():
    """Verifies that all 5 oracle tasks have independent, valid 64-character SHA-256 commitments."""
    suite = MultiTaskBlindOracleSuite()
    assert len(suite.tasks) == 5

    hashes = set()
    for task_id, spec in suite.tasks.items():
        assert len(spec.oracle_sealed_hash) == 64
        assert spec.oracle_sealed_hash not in hashes
        hashes.add(spec.oracle_sealed_hash)


def test_multi_task_autonomous_exploration_and_submissions():
    """Verifies that individual task oracles are queryable and produce valid submissions."""
    suite = MultiTaskBlindOracleSuite()
    engine = MultiTaskBlindBenchmarkEngine()

    oracle_1 = suite.query_task_oracle("TASK_1_MULTIHOP_FACTUAL")
    oracle_2 = suite.query_task_oracle("TASK_2_GREATER_THAN_COMPARE")

    sub_1 = engine.single_engine.run_autonomous_blind_discovery(oracle_1)
    sub_2 = engine.single_engine.run_autonomous_blind_discovery(oracle_2)

    assert sub_1.submission_id != sub_2.submission_id
    assert len(sub_1.sha256_submission_seal) == 64
    assert len(sub_2.sha256_submission_seal) == 64


def test_multi_task_oracle_unsealing_and_scoring():
    """Verifies that the multi-task benchmark achieves Mean BDS >= 95.0%, Trap Rejection = 100%, and Abstention = 100%."""
    engine = MultiTaskBlindBenchmarkEngine()
    report = engine.run_multi_task_benchmark()

    assert isinstance(report, MultiTaskBlindReport)
    assert report.is_multi_task_certified is True
    assert report.mean_bds_pct >= 95.0
    assert report.mean_jaccard >= 0.90
    assert report.mean_prospective_error_pct <= 5.0
    assert report.trap_rejection_rate_pct == 100.0
    assert report.abstention_accuracy_pct == 100.0
    assert len(report.task_scorecards) == 5


def test_multi_task_dag_certification_and_lineage():
    """Verifies that the multi-task master claim is registered as ACTIVE_SUPPORTED with 5 task dependencies."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = MultiTaskBlindBenchmarkEngine(claim_graph=claim_graph)

    report = engine.run_multi_task_benchmark()

    assert len(report.sha256_seal) == 64
    claim_id = f"CLAIM_MULTI_TASK_BLIND_BENCHMARK_{report.report_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert len(claim_graph.claims[claim_id].dependencies) == 5
    assert "Multi-Task Blind Benchmark Certified" in claim_graph.claims[claim_id].claim_statement
