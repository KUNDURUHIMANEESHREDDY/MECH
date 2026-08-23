"""
MECH Reasoning Agent

Scientific Research Orchestrator that sits between Agent 1 (scientific engine)
and Agent 2 (UI). Manages hypothesis lifecycle, experiment planning,
falsification testing, and evidence reasoning.

IMPORTANT: This module NEVER fabricates experimental evidence.
All measurements come from Agent 1's scientific engine.
"""

from backend.reasoning.research_orchestrator import (
    ScientificResearchOrchestrator,
    Hypothesis,
    ExperimentSpecification,
    EvidenceChain,
    CandidatePriority,
    ResearchLoopState,
    HypothesisState,
)

__all__ = [
    "ScientificResearchOrchestrator",
    "Hypothesis",
    "ExperimentSpecification",
    "EvidenceChain",
    "CandidatePriority",
    "ResearchLoopState",
    "HypothesisState",
]
