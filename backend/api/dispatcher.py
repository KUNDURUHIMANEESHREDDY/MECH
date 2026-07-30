from fastapi import APIRouter
from typing import Dict, Any, List

router = APIRouter()


@router.get("/status")
def api_status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "platform": "MECH Research Platform",
        "version": "2.0",
        "modules": [
            "api", "core", "interpretability", "discovery",
            "benchmarking", "reproductions", "science",
            "validation", "runtime", "agents", "knowledge_graph",
            "platform", "sdk", "services", "research",
            "research_platform", "datasets", "ui", "storage",
            "plugins", "analysis", "experiments"
        ]
    }


@router.get("/models")
def list_models() -> Dict[str, Any]:
    return {
        "models": [
            "gpt2-small", "gpt2-medium", "gemma-2b",
            "llama-3-8b", "qwen-7b", "pythia-1b",
            "mistral-7b", "distilgpt2"
        ]
    }


@router.get("/benchmarks")
def list_benchmarks() -> Dict[str, Any]:
    return {
        "benchmarks": ["IOI", "Induction", "SAE", "ACDC", "PathPatching"]
    }


@router.get("/discoveries")
def list_discoveries() -> Dict[str, Any]:
    return {
        "discoveries": [
            "InductionCircuitDiscovery",
            "IOISubcircuitDiscovery",
            "SAEFeatureDiscovery",
            "CrossModelUniversality",
            "ConceptEvolution",
            "PolysemanticityDiscovery",
            "AutomaticHypothesisGenerator",
            "PolysemanticityScaleCampaign",
            "CrossFamilySAEAlignment",
            "SuppressionCircuitDiscovery",
            "TrainingDynamicsDiscovery"
        ]
    }


@router.get("/portal/summary")
def portal_summary() -> Dict[str, Any]:
    return {
        "status": "active",
        "projects_count": 5
    }


@router.get("/interpretability/inspectors")
def list_inspectors() -> Dict[str, Any]:
    return {
        "inspectors": [
            "neuron", "attention", "residual", "layer",
            "token", "logit", "prediction", "feature"
        ]
    }


@router.get("/runtime/engines")
def list_runtime_engines() -> Dict[str, Any]:
    return {
        "engines": [
            "local", "distributed", "kubernetes", "slurm", "ray"
        ]
    }


@router.get("/agents")
def list_agents() -> Dict[str, Any]:
    return {
        "agents": [
            "ResearchSociety", "AIScientist", "PeerReviewPanel",
            "ResearchAgent", "ResearchCritic"
        ],
        "llm_backends": ["openai", "ollama"]
    }


@router.get("/knowledge-graph")
def knowledge_graph_info() -> Dict[str, Any]:
    return {
        "status": "active",
        "stores": ["provenance", "ontology", "query"]
    }
