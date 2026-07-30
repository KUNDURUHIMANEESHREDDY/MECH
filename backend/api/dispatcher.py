from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()


@router.get("/status")
def api_status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "platform": "MECH Research Platform",
        "version": "2.0"
    }


@router.get("/models")
def list_models() -> Dict[str, Any]:
    return {
        "models": ["gpt2-small", "gpt2-medium", "gemma-2b", "llama-3-8b", "qwen-7b"]
    }


@router.get("/benchmarks")
def list_benchmarks() -> Dict[str, Any]:
    return {
        "benchmarks": ["IOI", "Induction", "SAE", "ACDC"]
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
