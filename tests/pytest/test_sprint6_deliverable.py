"""Pytest suite for Sprint 6 — Self-Improving AI Scientist Deliverables."""

from __future__ import annotations

from api.dispatcher import build_dispatcher
from research_platform.meta.campaign_embeddings import CampaignEmbeddingsEngine
from research_platform.meta.experience_replay import ResearchExperienceReplay
from research_platform.meta.literature_learning_pipeline import LiteratureLearningPipeline
from research_platform.meta.meta_research_engine import MetaResearchEngine
from research_platform.meta.multi_agent_evolution import MultiAgentEvolutionEngine
from research_platform.meta.policy_repository import PolicyRepository
from research_platform.meta.research_curriculum import AutonomousResearchCurriculum
from research_platform.meta.research_strategy_optimizer import ResearchStrategyOptimizer
from research_platform.meta.scientific_skill_library import ScientificSkillLibrary
from research_platform.meta.self_reflection_engine import SelfReflectionEngine


def test_epic1_meta_research_engine_and_closed_loop():
    engine = MetaResearchEngine()
    engine.record_campaign_performance(
        campaign_id="camp_s6_1",
        topic="IOI Circuit Analysis",
        hypotheses_tested=10,
        discoveries_count=8,
        compute_used_vram_gb=16.0,
        failures_count=2,
    )
    analysis = engine.analyze_meta_performance()
    assert analysis["status"] == "Analyzed"
    assert "campaign_summary" in analysis
    assert analysis["campaign_summary"]["best_workflow"] == "SAE → Circuit Discovery → Causal Tracing"

    loop_res = engine.run_closed_feedback_loop("camp_s6_1")
    assert loop_res["status"] == "ClosedFeedbackLoopCompleted"
    assert loop_res["policy_id"].startswith("pol_circuit_discovery_v")


def test_epic2_self_reflection_engine():
    engine = SelfReflectionEngine()
    report = engine.generate_reflection_report(
        campaign_id="camp_s6_2",
        successful_hypotheses=["IOI Induction Head"],
        failed_hypotheses=["Random Layer 2 Ablation"],
        compute_used_gb_hours=20.0,
    )
    assert report["campaign_id"] == "camp_s6_2"
    assert report["compute_waste"] > 0
    assert "successes" in report
    assert "good_decisions" in report
    assert len(engine.list_reflections()) == 1


def test_epic3_policy_repository_and_versioning():
    repo = PolicyRepository()
    saved = repo.save_policy(
        domain="circuit_discovery",
        workflow=["Run SAE Inspection", "Run Causal Tracing"],
        success_rate=0.95,
        average_cost_usd=3.50,
        average_runtime_min=8.0,
    )
    assert saved["version"] == "2.0.0"

    comp = repo.compare_policies("pol_cd_v1", "pol_circuit_discovery_v2")
    assert comp["improvement_detected"] is True


def test_epic4_campaign_embeddings_vector_search():
    embeddings = CampaignEmbeddingsEngine()
    embeddings.index_campaign(
        campaign_id="camp_vector_1",
        topic="Superposition in MLP layers",
        summary_text="Deconstructed polysemantic MLP activations using 16k SAE features.",
    )
    res = embeddings.find_similar_campaigns("Superposition MLP", top_k=2)
    assert len(res) >= 1
    assert res[0]["similarity_score"] > 0.0


def test_epic5_literature_learning_pipeline():
    pipeline = LiteratureLearningPipeline()
    paper = pipeline.ingest_paper(paper_title="Interpretability in the Wild: IOI")
    assert paper["title"] == "Interpretability in the Wild: IOI"
    assert len(paper["extracted_discoveries"]) >= 2
    assert len(pipeline.get_knowledge_graph_updates()) > 0


def test_epic6_multi_agent_evolution():
    evolution = MultiAgentEvolutionEngine()
    agents = evolution.list_evolving_agents()
    assert len(agents) >= 3

    agent = evolution.record_agent_interaction(
        agent_id="research_critic",
        task_success=True,
        disagreed_with_peer=False,
        contribution_delta=0.05,
    )
    assert agent["agent_id"] == "research_critic"
    assert agent["success_rate"] >= 0.80


def test_epic7_scientific_skill_library_quality_scores():
    """Catalogue entries must not masquerade as measured skills.

    `execute_skill` previously reported output_state="Success", six extracted
    circuit nodes, and reproducibility_verified=True without running anything,
    so an agent could read a fabricated circuit discovery out of a lookup.
    """
    library = ScientificSkillLibrary()
    skills = library.list_skills()
    assert len(skills) >= 1
    # The metric values are catalogue defaults; what matters is that they say so.
    assert skills[0]["measured"] is False
    assert skills[0]["provenance"] == "reference"
    assert skills[0]["publication_eligible"] is False

    res = library.execute_skill(skill_id="skill_circuit_discovery",
                                input_params={"target_layer": 8})
    assert res["status"] == "unavailable"
    assert res["executed"] is False
    assert res["execution_result"] is None
    assert res["publication_eligible"] is False
    assert "not an executable pipeline" in res["reason"]
    assert res["quality_metrics"]["metrics_measured"] is False
    assert res["quality_metrics"]["provenance"] == "reference"


def test_epic8_research_experience_replay_why_provenance():
    """The seeded trajectory is illustrative, not recorded history.

    It previously "remembered" that an SAE checkpoint was available, that the
    IOI benchmark scored 0.94, and that a logit boost was confirmed, and then
    recommended a policy on that basis. None of those are measurable here.
    """
    replay = ResearchExperienceReplay()
    rep = replay.replay_campaign("camp_s6_ioi")
    assert rep["status"] == "ReplayCompleted"
    assert len(rep["divergence_points"]) >= 1
    assert rep["divergence_points"][0]["why_provenance"] is not None

    assert rep["recorded"] is False
    assert rep["provenance"] == "reference"
    assert rep["validation_eligible"] is False
    # No policy advice is drawn from a trajectory that did not happen.
    assert rep["policy_insight"] is None
    assert "Illustrative trajectory" in rep["reason"]
    for s in rep["trajectory"]:
        assert s["recorded"] is False
        assert s["publication_eligible"] is False

    step = replay.record_trajectory_step(
        campaign_id="camp_custom",
        action_type="Run SAE",
        decision_reasoning="Inspect features",
        why=["Low sparsity score"],
        confidence=0.92,
        alternatives=["Dense Search"],
        compute_cost_sec=10.0,
        confidence_delta=0.2,
    )
    assert step["why"] == ["Low sparsity score"]


def test_sprint6_dispatcher_endpoints():
    dispatcher = build_dispatcher()

    rec_res = dispatcher["api/v2/meta/record_campaign"]({"campaign_id": "camp_disp_1", "hypotheses_tested": 5, "discoveries_count": 4})
    assert rec_res["campaign_id"] == "camp_disp_1"

    loop_res = dispatcher["api/v2/meta/run_closed_loop"]({"campaign_id": "camp_s6_ioi"})
    assert loop_res["status"] == "ClosedFeedbackLoopCompleted"

    pols_res = dispatcher["api/v2/meta/policies"]({})
    assert len(pols_res["policies"]) >= 1

    search_res = dispatcher["api/v2/meta/search_campaigns"]({"query_topic": "IOI Circuit"})
    assert len(search_res["results"]) >= 1
