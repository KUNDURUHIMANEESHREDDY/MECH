"""Pytest suite for AI 2 — Adaptive Runtime Deliverables."""

from __future__ import annotations

from api.dispatcher import build_dispatcher
from runtime.adaptive.adaptive_cache_engine import AdaptiveCacheEngine
from runtime.adaptive.calibration_engine import PredictionCalibrationEngine
from runtime.adaptive.energy_cost_optimizer import EnergyCostOptimizerEngine
from runtime.adaptive.execution_episodes import ExecutionEpisodesEngine
from runtime.adaptive.fault_diagnostics import AutonomousFaultDiagnosticsEngine
from runtime.adaptive.policy_engine import HardwareConstraints, PolicyEngine
from runtime.adaptive.predictive_autoscaler import PredictiveAutoscalerEngine
from runtime.adaptive.runtime_analytics import RuntimeAnalyticsSuite
from runtime.adaptive.runtime_decision_engine import RuntimeDecisionEngine
from runtime.adaptive.runtime_knowledge_base import RuntimeKnowledgeBaseEngine
from runtime.adaptive.runtime_simulator import RuntimeSimulatorEngine
from runtime.adaptive.workload_fingerprints import WorkloadFingerprintsEngine


def test_policy_engine_objectives_and_constraints():
    pe = PolicyEngine()
    pol = pe.configure_policy("Lowest Cost")
    assert pol["objective_name"] == "Lowest Cost"
    assert pol["cost_weight"] == 0.7
    assert pol["constraints"]["max_budget_usd_per_hour"] == 1.20

    custom = HardwareConstraints(max_vram_gb=40.0, max_budget_usd_per_hour=8.0)
    fast_pol = pe.configure_policy("Fastest Completion", custom_constraints=custom)
    assert fast_pol["runtime_weight"] == 0.8
    assert fast_pol["constraints"]["max_vram_gb"] == 40.0


def test_versioned_calibration_models():
    cal = PredictionCalibrationEngine()
    res = cal.record_execution_outcome(
        model_name="GPT-2 Small",
        target_backend="Ray",
        predicted_runtime_sec=48.0,
        observed_runtime_sec=52.8,
        predicted_vram_gb=8.4,
        observed_vram_gb=8.8,
    )
    assert res["latest_model_version"]["version_number"] == "2.0.0"
    versions = cal.list_model_versions("GPT-2 Small", "Ray")
    assert len(versions) == 2


def test_execution_episodes_vector_search():
    ee = ExecutionEpisodesEngine()
    ee.record_episode(
        campaign_id="camp_vector_ep",
        model_name="Llama-3",
        plan_summary={"goal": "Circuit Search"},
        simulation_output={"simulated_runtime_sec": 42.0},
        decision_record={"target_backend": "Ray"},
        execution_telemetry={"throughput_tok_per_sec": 1350.0},
        outcome_status="Success",
    )
    similar = ee.find_similar_episodes("Llama-3", top_k=2)
    assert len(similar) >= 1
    assert similar[0]["similarity_score"] > 0.0


def test_knowledge_base_validation_and_archival():
    kb = RuntimeKnowledgeBaseEngine()
    kb.record_successful_strategy(model_name="OutdatedModel", target_backend="Ray", throughput_tok_per_sec=400.0)
    val_res = kb.verify_and_prune_knowledge_base(min_throughput_threshold=800.0)
    assert val_res["archived_profiles_count"] >= 1


def test_pareto_frontier_and_decision_graph():
    dec_engine = RuntimeDecisionEngine()
    dec = dec_engine.make_execution_decision(model_name="GPT-2 Small", user_objective="Lowest Cost")
    assert len(dec["pareto_frontier_options"]) == 3
    assert len(dec["counterfactual_evaluations"]) >= 1
    assert len(dec["decision_graph_provenance"]) == 4


def test_adaptive_runtime_dispatcher_routes():
    dispatcher = build_dispatcher()

    ep_search = dispatcher["api/v2/runtime/adaptive/search_episodes"]({"query_model": "GPT-2 Small"})
    assert len(ep_search["results"]) >= 1

    prune_res = dispatcher["api/v2/runtime/adaptive/prune_knowledge_base"]({"min_throughput_threshold": 800.0})
    assert prune_res["status"] == "KnowledgeBaseValidated"

    ver_res = dispatcher["api/v2/runtime/adaptive/calibration_versions"]({"model_name": "GPT-2 Small", "target_backend": "Ray"})
    assert len(ver_res["versions"]) >= 1
