"""Scientific Knowledge Graph Formal Ontology.

Defines standard entity Node types and multi-relational Edge types for 
mechanistic interpretability knowledge representation.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Set


class NodeType(str, Enum):
    """Formal Node entity types in the scientific knowledge graph."""
    PAPER = "Paper"
    EXPERIMENT = "Experiment"
    CAMPAIGN = "Campaign"
    DATASET = "Dataset"
    PROMPT = "Prompt"
    CIRCUIT = "Circuit"
    ATTENTION_HEAD = "AttentionHead"
    MLP_NEURON = "MLPNeuron"
    SAE_FEATURE = "SAEFeature"
    MECHANISM_CLAIM = "MechanismClaim"
    EVIDENCE = "Evidence"
    PUBLICATION = "Publication"
    RESEARCH_GOAL = "ResearchGoal"
    BENCHMARK_RUN = "BenchmarkRun"
    BENCHMARK_RESULT = "BenchmarkResult"
    FEATURE = "Feature" # General term for Model/Layer/Head/Neuron/SAE
    TOKEN_EXAMPLE = "TokenExample"


class EdgeType(str, Enum):
    """Formal Edge relation types in the scientific knowledge graph."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REPRODUCES = "reproduces"
    DERIVES_FROM = "derives_from"
    ACTIVATES = "activates"
    PATCHES = "patches"
    DISCOVERED_BY = "discovered_by"
    VALIDATED_BY = "validated_by"
    SUPERSEDES = "supersedes"
    CITES = "cites"
    CONTAINS = "contains"
    REFERENCES = "references"
    EVALUATED = "evaluated"
    VALIDATED_AGAINST = "validated_against"
    EXPLAINS = "explains"
    ILLUSTRATED_BY = "illustrated_by"
