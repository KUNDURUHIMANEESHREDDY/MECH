"""Agent 1 -> Agent 2 Scientific REST API Router for MECH Platform.

Exposes clean, programmatic endpoints for:
- Experiment Management: POST /experiments, POST /experiments/{id}/run, GET /experiments/{id}, GET /experiments/{id}/runs, GET /experiments/{id}/evidence
- Model Introspection: GET /models, GET /models/{id}/architecture
- Hypotheses: POST /hypotheses, GET /hypotheses/{id}
- Ad-Hoc Causal Interventions: POST /interventions
- Automated Candidate Discovery: POST /discovery/scan
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.core.discovery_engine import AutomatedDiscoveryEngine
from backend.core.experiment_engine import (
    ComponentTarget,
    InterventionSpec,
    MechanisticExperimentEngine,
)
from backend.core.model_adapter import get_model_adapter
from backend.core.model_registry import get_model_registry
from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import (
    EvidenceRecord,
    Experiment,
    ExperimentExecutionStatus,
    ExperimentRun,
    Hypothesis,
    HypothesisStatus,
    InterventionType,
)

logger = logging.getLogger("MECH.api.experiments_router")

router = APIRouter(tags=["experiments"])
storage = DesktopStorage()
storage.initialize()


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class CreateExperimentRequest(BaseModel):
    name: str
    clean_prompt: str
    target_token: str
    source_component: str = "L9H9"
    corrupted_prompt: Optional[str] = None
    distractor_token: Optional[str] = None
    model_id: str = "gpt2"
    dataset_id: str = "custom"
    description: str = ""
    intervention_type: str = "ABLATION_ZERO"
    repeats: int = 3
    investigation_id: Optional[str] = None
    hypothesis_id: Optional[str] = None


class RunExperimentRequest(BaseModel):
    repeats: Optional[int] = None
    seed: int = 42


class CreateHypothesisRequest(BaseModel):
    title: str
    statement: str
    target_component: str
    investigation_id: Optional[str] = None
    prediction: str = "Intervention reduces target logit"
    falsification_condition: str = "Delta logit <= 0.0 or specificity < 1.5"


class AdHocInterventionRequest(BaseModel):
    prompt: str
    target: str
    component: str
    model_id: str = "gpt2"
    corrupted: Optional[str] = None
    distractor: Optional[str] = None
    type: str = "ablation_zero"
    coeff: float = 0.0
    repeats: int = 3
    seed: int = 42


class DiscoveryScanRequest(BaseModel):
    clean_prompt: str
    target_token: str
    corrupted_prompt: Optional[str] = None
    distractor_token: Optional[str] = None
    model_id: str = "gpt2"
    max_candidates: int = 4
    repeats: int = 3


# ---------------------------------------------------------------------------
# Dataset Schemas
# ---------------------------------------------------------------------------
class DatasetDefinition(BaseModel):
    id: str
    name: str
    description: str
    category: str = "custom"  # "ioi" | "nlp" | "custom" | "synthetic"
    prompt_pairs: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Built-in dataset definitions for mechanistic interpretability
_BUILTIN_DATASETS: List[Dict[str, Any]] = [
    {
        "id": "ioi_basic",
        "name": "IOI Basic (Indirect Object Identification)",
        "description": "Standard IOI probe prompts with clean/corrupted pairs for testing name mover heads.",
        "category": "ioi",
        "prompt_pairs": [
            {"clean": "When Mary and John went to the store, John gave a drink to", "corrupted": "When Mary and John went to the store, Mary gave a drink to", "target_token": " Mary", "distractor_token": " John"},
            {"clean": "The cat chased the mouse. The mouse ran away from", "corrupted": "The cat chased the mouse. The cat ran away from", "target_token": " cat", "distractor_token": " mouse"},
            {"clean": "Alice told Bob that she would help him with", "corrupted": "Alice told Bob that he would help her with", "target_token": " him", "distractor_token": " her"},
            {"clean": "After the teacher praised the student, she felt proud of", "corrupted": "After the teacher praised the student, he felt proud of", "target_token": " her", "distractor_token": " him"},
            {"clean": "Sarah handed the book to Tom because", "corrupted": "Sarah handed the book to Tom after", "target_token": " he", "distractor_token": " she"},
        ],
        "metadata": {"source": "Wang et al. 2022", "task_type": "indirect_object_identification"},
    },
    {
        "id": "ioi_counterfactual",
        "name": "IOI Counterfactual Suite",
        "description": "IOI prompts with counterfactual name swaps to test robustness of name mover circuits.",
        "category": "ioi",
        "prompt_pairs": [
            {"clean": "Then, Mary and John went to the store. John gave a drink to", "corrupted": "Then, Mary and John went to the store. Mary gave a drink to", "target_token": " Mary", "distractor_token": " John"},
            {"clean": "The dog followed the cat. The cat climbed the tree to escape from", "corrupted": "The dog followed the cat. The dog climbed the tree to escape from", "target_token": " dog", "distractor_token": " cat"},
        ],
        "metadata": {"source": "Counterfactual IOI", "task_type": "indirect_object_identification"},
    },
    {
        "id": "factual_recall",
        "name": "Factual Recall Probes",
        "description": "Fact completion prompts for testing knowledge recall circuits in early and late layers.",
        "category": "nlp",
        "prompt_pairs": [
            {"clean": "The capital of France is", "corrupted": "The capital of Germany is", "target_token": " Paris", "distractor_token": " Berlin"},
            {"clean": "Water boils at", "corrupted": "Water freezes at", "target_token": " 100", "distractor_token": " 0"},
            {"clean": "The sun rises in the", "corrupted": "The sun sets in the", "target_token": " east", "distractor_token": " west"},
        ],
        "metadata": {"source": "Synthetic factual probes", "task_type": "factual_recall"},
    },
    {
        "id": "syntactic_dependency",
        "name": "Syntactic Dependency Probes",
        "description": "Prompts testing syntactic dependency resolution and subject-verb agreement.",
        "category": "nlp",
        "prompt_pairs": [
            {"clean": "The cat that chased the mouse was", "corrupted": "The cat that chased the mouse were", "target_token": " was", "distractor_token": " were"},
            {"clean": "The dogs that chased the cat were", "corrupted": "The dogs that chased the cat was", "target_token": " were", "distractor_token": " was"},
        ],
        "metadata": {"source": "Syntactic probes", "task_type": "syntactic_dependency"},
    },
    {
        "id": "sae_feature_discovery",
        "name": "SAE Feature Discovery Suite",
        "description": "Prompts designed to activate sparse autoencoder features for mechanism discovery.",
        "category": "custom",
        "prompt_pairs": [
            {"clean": "The brilliant scientist discovered a", "corrupted": "The clumsy chef burnt the", "target_token": " new", "distractor_token": " old"},
            {"clean": "In the year 2050, humanity will", "corrupted": "In the year 1900, humanity did", "target_token": " have", "distractor_token": " had"},
        ],
        "metadata": {"source": "SAE exploration", "task_type": "feature_discovery"},
    },
]


# ---------------------------------------------------------------------------
# Model Registry Endpoints
# ---------------------------------------------------------------------------
@router.get("/models")
def list_models() -> List[Dict[str, Any]]:
    """List all registered models with capability descriptors."""
    reg = get_model_registry(storage)
    return reg.list_models()


@router.get("/models/{model_id}/architecture")
def get_model_architecture(model_id: str) -> Dict[str, Any]:
    """Returns introspected model architecture from live ModelAdapter."""
    try:
        adapter = get_model_adapter(model_id=model_id)
        return adapter.get_architecture_info()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Failed to inspect model '{model_id}': {exc}")


# ---------------------------------------------------------------------------
# Dataset Endpoints (Agent 1)
# ---------------------------------------------------------------------------
@router.get("/datasets")
def list_datasets(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists all available datasets (built-in + custom) for mechanistic experiments."""
    datasets = list(_BUILTIN_DATASETS)
    if category:
        datasets = [d for d in datasets if d["category"] == category]
    return datasets


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> Dict[str, Any]:
    """Retrieves a specific dataset with its prompt pairs."""
    for ds in _BUILTIN_DATASETS:
        if ds["id"] == dataset_id:
            return ds
    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")


@router.post("/datasets", status_code=201)
def create_dataset(req: DatasetDefinition) -> Dict[str, Any]:
    """Registers a custom dataset for use in experiments."""
    ds = {
        "id": req.id,
        "name": req.name,
        "description": req.description,
        "category": req.category,
        "prompt_pairs": req.prompt_pairs,
        "metadata": req.metadata,
    }
    _BUILTIN_DATASETS.append(ds)
    return ds


@router.get("/datasets/{dataset_id}/prompts")
def get_dataset_prompts(dataset_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Returns prompt pairs from a dataset, ready for experiment execution."""
    for ds in _BUILTIN_DATASETS:
        if ds["id"] == dataset_id:
            return ds["prompt_pairs"][:limit]
    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")


@router.get("/datasets/{dataset_id}/random-pair")
def get_random_prompt_pair(dataset_id: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Returns a single random prompt pair from a dataset for experiment seeding."""
    import random as _random
    for ds in _BUILTIN_DATASETS:
        if ds["id"] == dataset_id:
            pairs = ds["prompt_pairs"]
            if not pairs:
                raise HTTPException(status_code=404, detail="Dataset has no prompt pairs.")
            if seed is not None:
                _random.seed(seed)
            pair = _random.choice(pairs)
            return {
                "dataset_id": dataset_id,
                "clean_prompt": pair.get("clean", ""),
                "corrupted_prompt": pair.get("corrupted"),
                "target_token": pair.get("target_token", ""),
                "distractor_token": pair.get("distractor_token"),
            }
    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")


# ---------------------------------------------------------------------------
# Experiment Management Endpoints
# ---------------------------------------------------------------------------
@router.post("/experiments", status_code=201)
def create_experiment(req: CreateExperimentRequest) -> Dict[str, Any]:
    """Creates an experiment specification in SQLite storage."""
    itype_map = {
        "ablation_zero": InterventionType.ABLATION_ZERO,
        "ablation_mean": InterventionType.ABLATION_MEAN,
        "ablation_noise": InterventionType.ABLATION_NOISE,
        "activation_patching": InterventionType.ACTIVATION_PATCHING,
        "patching": InterventionType.ACTIVATION_PATCHING,
        "steering": InterventionType.STEERING,
        "scaling": InterventionType.SCALING,
        "clamping": InterventionType.CLAMPING,
    }
    itype = itype_map.get(req.intervention_type.lower(), InterventionType.ABLATION_ZERO)
    inv_id = req.investigation_id or f"inv_{uuid.uuid4().hex[:6]}"

    exp = Experiment(
        name=req.name,
        description=req.description,
        investigation_id=inv_id,
        hypothesis_id=req.hypothesis_id,
        model_id=req.model_id,
        dataset_id=req.dataset_id,
        clean_prompt=req.clean_prompt,
        corrupted_prompt=req.corrupted_prompt,
        target_token=req.target_token,
        distractor_token=req.distractor_token,
        intervention_type=itype,
        source_component=req.source_component,
        repeats=req.repeats,
    )
    storage.save_experiment(exp.model_dump())
    return exp.model_dump()


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> Dict[str, Any]:
    """Retrieves an experiment by ID."""
    exp = storage.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return exp


@router.post("/experiments/{experiment_id}/run")
def run_experiment(experiment_id: str, req: Optional[RunExperimentRequest] = None) -> Dict[str, Any]:
    """Executes a full multi-trial causal experiment with 5-tier controls against real weights."""
    exp_dict = storage.get_experiment(experiment_id)
    if not exp_dict:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    repeats = (req.repeats if req and req.repeats else exp_dict.get("repeats", 3))
    seed = req.seed if req else 42

    itype_val = exp_dict.get("intervention_type", "ABLATION_ZERO")
    itype = InterventionType(itype_val) if isinstance(itype_val, str) else itype_val

    comp = ComponentTarget.from_string(exp_dict.get("source_component", "L9H9"))
    spec = InterventionSpec(
        clean_prompt=exp_dict["clean_prompt"],
        corrupted_prompt=exp_dict.get("corrupted_prompt"),
        target_token=exp_dict["target_token"],
        distractor_token=exp_dict.get("distractor_token"),
        target_components=[comp],
        intervention_type=itype,
        random_seed=seed,
        repeats=repeats,
        investigation_id=exp_dict.get("investigation_id"),
        hypothesis_id=exp_dict.get("hypothesis_id"),
        experiment_id=experiment_id,
    )

    engine = MechanisticExperimentEngine(model_id=exp_dict.get("model_id", "gpt2"), storage=storage)
    result = engine.run_intervention_experiment(spec)

    # Update experiment execution status
    exp_dict["execution_status"] = ExperimentExecutionStatus.COMPLETED.value
    storage.save_experiment(exp_dict)

    return result.to_dict()


@router.get("/experiments/{experiment_id}/runs")
def list_experiment_runs(experiment_id: str) -> List[Dict[str, Any]]:
    """Lists all execution runs recorded for an experiment."""
    runs = storage.list_experiment_runs(experiment_id=experiment_id)
    return runs


@router.get("/experiments/{experiment_id}/evidence")
def list_experiment_evidence(experiment_id: str) -> List[Dict[str, Any]]:
    """Lists all evidence records associated with an experiment."""
    evidence = storage.list_evidence_records(experiment_run_id=experiment_id)
    if not evidence:
        evidence = storage.list_evidence_records(investigation_id=experiment_id)
    return evidence


# ---------------------------------------------------------------------------
# Hypotheses Endpoints
# ---------------------------------------------------------------------------
@router.post("/hypotheses", status_code=201)
def create_hypothesis(req: CreateHypothesisRequest) -> Dict[str, Any]:
    """Registers a scientific hypothesis for testing."""
    inv_id = req.investigation_id or f"inv_{uuid.uuid4().hex[:6]}"
    hyp = Hypothesis(
        investigation_id=inv_id,
        title=req.title,
        statement=req.statement,
        target_component=req.target_component,
        prediction=req.prediction,
        falsification_condition=req.falsification_condition,
        status=HypothesisStatus.UNTESTED,
    )
    storage.save_hypothesis(hyp.model_dump())
    return hyp.model_dump()


@router.get("/hypotheses/{hypothesis_id}")
def get_hypothesis(hypothesis_id: str) -> Dict[str, Any]:
    """Retrieves a hypothesis record with live evidence evaluation."""
    hyp = storage.get_hypothesis(hypothesis_id)
    if not hyp:
        raise HTTPException(status_code=404, detail=f"Hypothesis '{hypothesis_id}' not found.")
    return hyp


# ---------------------------------------------------------------------------
# Ad-Hoc Interventions Endpoint
# ---------------------------------------------------------------------------
@router.post("/interventions")
def execute_intervention(req: AdHocInterventionRequest) -> Dict[str, Any]:
    """Executes a standalone causal intervention with multi-trial analysis and 5-tier controls."""
    itype_map = {
        "ablation_zero": InterventionType.ABLATION_ZERO,
        "ablation_mean": InterventionType.ABLATION_MEAN,
        "ablation_noise": InterventionType.ABLATION_NOISE,
        "activation_patching": InterventionType.ACTIVATION_PATCHING,
        "patching": InterventionType.ACTIVATION_PATCHING,
        "steering": InterventionType.STEERING,
        "scaling": InterventionType.SCALING,
        "clamping": InterventionType.CLAMPING,
    }
    itype = itype_map.get(req.type.lower(), InterventionType.ABLATION_ZERO)
    comp = ComponentTarget.from_string(req.component)

    spec = InterventionSpec(
        clean_prompt=req.prompt,
        corrupted_prompt=req.corrupted,
        target_token=req.target,
        distractor_token=req.distractor,
        target_components=[comp],
        intervention_type=itype,
        scale_coefficient=req.coeff,
        random_seed=req.seed,
        repeats=req.repeats,
    )

    engine = MechanisticExperimentEngine(model_id=req.model_id, storage=storage)
    result = engine.run_intervention_experiment(spec)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Automated Candidate Discovery Endpoint
# ---------------------------------------------------------------------------
@router.post("/discovery/scan")
def run_discovery_scan(req: DiscoveryScanRequest) -> Dict[str, Any]:
    """Executes automated candidate discovery: layer scan -> head scan -> neuron scan -> causal validation."""
    engine = AutomatedDiscoveryEngine(model_id=req.model_id, storage=storage)
    report = engine.discover_and_validate_circuit(
        clean_prompt=req.clean_prompt,
        corrupted_prompt=req.corrupted_prompt,
        target_token=req.target_token,
        distractor_token=req.distractor_token,
        max_candidates_to_validate=req.max_candidates,
        validation_repeats=req.repeats,
    )
    return report.to_dict()
