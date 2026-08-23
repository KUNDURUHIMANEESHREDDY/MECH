"""Unit and integration tests for Dedicated Hallucination Competition Pipeline."""

import pytest
from backend.science.reproducibility.hallucination_pipeline import (
    HallucinationCompetitionPipeline,
    HallucinationEvidenceReport,
    CrossoverPoint,
)
from backend.core.capability_registry import CapabilityRegistry
from backend.research_platform.autonomous.ai_research_assistant import AIResearchAssistant


def test_hallucination_pipeline_four_pillars():
    pipeline = HallucinationCompetitionPipeline()
    report = pipeline.run_experiment(
        clean_factual_prompt="The capital of France is",
        fabricated_prompt="The capital of Atlantis is",
        factual_target=" Paris",
        fabricated_target=" Eldrin",
        candidate_mlp="L6_MLP",
        candidate_head="L9_H5",
        grid_resolution=5,
    )

    assert isinstance(report, HallucinationEvidenceReport)
    assert report.verdict == "Causally Verified Hallucination Circuit"

    # Pillar 1: Correlation Evidence
    corr = report.correlation_evidence
    assert corr["status"] == "verified"
    assert "parametric_mlp" in corr
    assert "induction_head" in corr
    assert corr["induction_head"]["matches_induction_criteria"] is True

    # Pillar 2: Causal Suppression Evidence (Necessity)
    supp = report.suppression_evidence
    assert supp["status"] == "verified"
    assert supp["mlp_suppression"]["necessity_confirmed"] is True
    assert supp["head_suppression"]["necessity_confirmed"] is True

    # Pillar 3: Causal Restoration Evidence (Sufficiency)
    rest = report.restoration_evidence
    assert rest["status"] == "verified"
    assert rest["mlp_restoration"]["sufficiency_confirmed"] is True
    assert rest["head_induction"]["sufficiency_confirmed"] is True

    # Pillar 4: Competition & Crossover Curve
    assert len(report.competition_grid) == 25  # 5x5 grid
    crossover = report.critical_crossover
    assert isinstance(crossover, CrossoverPoint)
    assert crossover.crossover_ratio > 0.0
    assert crossover.alpha_parametric_weight > 0.0
    assert crossover.beta_induction_weight > 0.0


def test_hallucination_tool_in_capability_registry():
    registry = CapabilityRegistry()
    assert registry.get_tool("run_hallucination_experiment") is not None

    res = registry.execute_tool("run_hallucination_experiment", {
        "clean_factual_prompt": "The primary author of Attention Is All You Need is",
        "fabricated_prompt": "The author of the fictional chronicle is",
    })

    assert res["status"] == "success"
    assert "report" in res
    rep = res["report"]
    assert "correlation_evidence" in rep
    assert "suppression_evidence" in rep
    assert "restoration_evidence" in rep
    assert "critical_crossover" in rep


def test_ai_research_assistant_with_hallucination_goal():
    assistant = AIResearchAssistant()
    res = assistant.investigate(
        goal="Automatically discover and causally verify an induction-head vs MLP hallucination circuit",
        model_name="gpt2",
    )

    assert res["status"] == "completed"
    assert "hallucination_competition" in res
    assert res["hallucination_competition"] is not None

    # Verify reflection findings mention competition and crossover
    findings = " ".join(res["reflection"]["findings"])
    assert "Competition" in findings or "Induction" in findings
    assert "Necessity" in findings or "Sufficiency" in findings
