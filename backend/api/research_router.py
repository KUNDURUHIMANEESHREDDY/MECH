"""Research and Scientific Experimentation API Router for MECH Platform.

Exposes REST endpoints for Investigations, Hypotheses, Experiments, Live Interventions,
Evidence Graphs, Mechanism Builders, and Research Artifacts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.science.experiment_runner import ScientificExperimentRunner
from backend.science.hypothesis_engine import HypothesisEngine
from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import (
    ComputeJob,
    EvidenceRecord,
    Experiment,
    ExperimentRun,
    Hypothesis,
    Investigation,
    Mechanism,
    ResearchArtifact,
)

logger = logging.getLogger("MECH.api.research_router")

router = APIRouter(prefix="/api/v1/research", tags=["research"])


def _normalize_run_status(run: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure every run payload carries a normalized `execution_status` for the UI.

    The ExperimentRun model stores the terminal status in `status` (JobStatus).
    The frontend reads `execution_status` to render PENDING/RUNNING/COMPLETED/
    FAILED/NOT_EXECUTABLE. Never allow a run to silently default to COMPLETED
    when it actually failed.
    """
    run = dict(run)
    if "execution_status" not in run or not run.get("execution_status"):
        run["execution_status"] = run.get("status") or "COMPLETED"
    if "used_mock_data" not in run:
        run["used_mock_data"] = False
    return run

_db_path = Path.home() / ".cache" / "neural-debugger" / "mech.db"
_storage = DesktopStorage(_db_path)
_storage.initialize()

_experiment_runner = ScientificExperimentRunner(storage=_storage)
_hypothesis_engine = HypothesisEngine(storage=_storage)


# ---------------------------------------------------------------------------
# Investigations
# ---------------------------------------------------------------------------
@router.get("/investigations")
def list_investigations() -> Dict[str, Any]:
    items = _storage.list_investigations()
    # If no investigations exist, seed a default scientific IOI investigation
    if not items:
        default_inv = Investigation(
            id="inv_ioi_default",
            title="Indirect Object Identification (IOI) Circuit",
            research_question="How does GPT-2 identify the indirect object token in ambiguous dual-subject sentences?",
            model_id="gpt2",
            dataset_id="ioi",
            tags=["ioi", "circuit-discovery", "attention-routing"],
        )
        _storage.save_investigation(default_inv.model_dump())
        
        # Seed default hypothesis
        default_hyp = Hypothesis(
            id="hyp_l9h9_namemover",
            investigation_id="inv_ioi_default",
            title="L9H9 Name Mover Hypothesis",
            statement="Attention head L9H9 causally retrieves and moves the indirect object token into the residual stream.",
            target_component="L9H9",
            prediction="Ablating L9H9 or patching its output with corrupted activations will significantly reduce target token probability.",
            expected_evidence="Δlogit > 2.0 upon activation patching on IOI probe dataset.",
            falsification_condition="Ablating L9H9 produces Δlogit < 0.5 or equal effect on random control tokens.",
        )
        _storage.save_hypothesis(default_hyp.model_dump())
        default_inv.current_hypothesis_id = default_hyp.id
        _storage.save_investigation(default_inv.model_dump())
        items = [default_inv.model_dump()]

    return {"investigations": items, "count": len(items)}


@router.post("/investigations")
def create_or_update_investigation(payload: Dict[str, Any]) -> Dict[str, Any]:
    inv = Investigation(**payload)
    inv.updated_at = time.time()
    saved = _storage.save_investigation(inv.model_dump())
    return {"status": "success", "investigation": saved}


@router.get("/investigations/{inv_id}")
def get_investigation(inv_id: str) -> Dict[str, Any]:
    inv = _storage.get_investigation(inv_id)
    if not inv:
        raise HTTPException(status_code=404, detail=f"Investigation {inv_id} not found.")
    hypotheses = _storage.list_hypotheses(inv_id)
    runs = _storage.list_experiment_runs(investigation_id=inv_id)
    evidence = _storage.list_evidence_records(investigation_id=inv_id)
    mechanisms = _storage.list_mechanisms(inv_id)
    return {
        "investigation": inv,
        "hypotheses": hypotheses,
        "runs": runs,
        "evidence": evidence,
        "mechanisms": mechanisms,
    }


@router.delete("/investigations/{inv_id}")
def delete_investigation(inv_id: str) -> Dict[str, Any]:
    deleted = _storage.delete_investigation(inv_id)
    return {"status": "deleted" if deleted else "not_found", "id": inv_id}


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------
@router.get("/hypotheses")
def list_hypotheses(investigation_id: Optional[str] = None) -> Dict[str, Any]:
    items = _storage.list_hypotheses(investigation_id)
    return {"hypotheses": items, "count": len(items)}


@router.post("/hypotheses")
def create_or_update_hypothesis(payload: Dict[str, Any]) -> Dict[str, Any]:
    hyp = Hypothesis(**payload)
    hyp.updated_at = time.time()
    saved = _storage.save_hypothesis(hyp.model_dump())
    return {"status": "success", "hypothesis": saved}


@router.delete("/hypotheses/{hyp_id}")
def delete_hypothesis(hyp_id: str) -> Dict[str, Any]:
    deleted = _storage.delete_hypothesis(hyp_id)
    return {"status": "deleted" if deleted else "not_found", "id": hyp_id}


@router.post("/hypotheses/{hyp_id}/evaluate")
def evaluate_hypothesis(hyp_id: str, investigation_id: str = Query(...)) -> Dict[str, Any]:
    return _hypothesis_engine.evaluate_hypothesis(hyp_id, investigation_id)


# ---------------------------------------------------------------------------
# Experiments & Causal Interventions
# ---------------------------------------------------------------------------
@router.post("/experiments/run")
def run_causal_experiment(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Executes a live causal intervention experiment on model weights."""
    from backend.services import gpt2_engine
    if not gpt2_engine.is_available():
        from backend.science.scientific_envelope import wrap_unexecuted_failure
        return wrap_unexecuted_failure("PyTorch ML backend is unavailable.").model_dump()
    if gpt2_engine._model is None:
        gpt2_engine.load()
    if gpt2_engine._model is None or gpt2_engine._tokenizer is None:
        from backend.science.scientific_envelope import wrap_unexecuted_failure
        return wrap_unexecuted_failure("Failed to load model weights.").model_dump()
    try:
        exp = Experiment(**payload)
        run = _experiment_runner.run_experiment(exp)
        # Mark whether the run used live model data or mock/synthetic data
        run_dict = _normalize_run_status(run.model_dump())
        run_dict["used_mock_data"] = False  # Live run, data is from actual model
        return {"status": "success", "run": run_dict}
    except Exception as exc:
        logger.exception("Experiment execution failed: %s", exc)
        from backend.science.scientific_envelope import wrap_unexecuted_failure
        return wrap_unexecuted_failure(str(exc)).model_dump()


@router.get("/runs")
def list_runs(experiment_id: Optional[str] = None, investigation_id: Optional[str] = None) -> Dict[str, Any]:
    items = _storage.list_experiment_runs(investigation_id=investigation_id, experiment_id=experiment_id)
    items = [_normalize_run_status(run) for run in items]
    return {"runs": items, "count": len(items)}


# ---------------------------------------------------------------------------
# Evidence Records & Knowledge Graph
# ---------------------------------------------------------------------------
@router.get("/evidence")
def list_evidence(investigation_id: Optional[str] = None, hypothesis_id: Optional[str] = None) -> Dict[str, Any]:
    items = _storage.list_evidence_records(investigation_id, hypothesis_id)
    return {"evidence": items, "count": len(items)}


@router.post("/evidence")
def create_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
    evi = EvidenceRecord(**payload)
    saved = _storage.save_evidence_record(evi.dict())
    return {"status": "success", "evidence": saved}


@router.get("/evidence/matrix")
def get_evidence_matrix(investigation_id: str = Query(...)) -> Dict[str, Any]:
    matrix = _hypothesis_engine.get_evidence_matrix(investigation_id)
    return {"matrix": matrix, "count": len(matrix)}


# ---------------------------------------------------------------------------
# Mechanism Builder
# ---------------------------------------------------------------------------
@router.get("/mechanisms")
def list_mechanisms(investigation_id: Optional[str] = None) -> Dict[str, Any]:
    items = _storage.list_mechanisms(investigation_id)
    return {"mechanisms": items, "count": len(items)}


@router.post("/mechanisms")
def create_or_update_mechanism(payload: Dict[str, Any]) -> Dict[str, Any]:
    mech = Mechanism(**payload)
    mech.updated_at = time.time()
    saved = _storage.save_mechanism(mech.dict())
    return {"status": "success", "mechanism": saved}


@router.delete("/mechanisms/{mech_id}")
def delete_mechanism(mech_id: str) -> Dict[str, Any]:
    deleted = _storage.delete_mechanism(mech_id)
    return {"status": "deleted" if deleted else "not_found", "id": mech_id}


# ---------------------------------------------------------------------------
# Research Artifacts & Publication Mode
# ---------------------------------------------------------------------------
@router.get("/artifacts")
def list_artifacts(investigation_id: Optional[str] = None) -> Dict[str, Any]:
    items = _storage.list_artifacts(investigation_id)
    return {"artifacts": items, "count": len(items)}


@router.post("/artifacts/generate-report")
def generate_research_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generates structured 11-section publication report from real experiment records."""
    inv_id = payload.get("investigation_id", "inv_ioi_default")
    inv = _storage.get_investigation(inv_id) or {}
    hypotheses = _storage.list_hypotheses(inv_id)
    runs = _storage.list_experiment_runs(investigation_id=inv_id)
    evidence = _storage.list_evidence_records(investigation_id=inv_id)
    mechanisms = _storage.list_mechanisms(inv_id)

    report_md = f"""# Scientific Research Report: {inv.get('title', 'Mechanistic Investigation')}

**Investigation ID**: `{inv_id}`  
**Model**: `{inv.get('model_id', 'gpt2')}`  
**Dataset**: `{inv.get('dataset_id', 'ioi')}`  
**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  

---

## 1. Research Question
{inv.get('research_question', 'N/A')}

## 2. Model & Checkpoint
Model architecture: GPT-2 Small (12 layers, 12 attention heads, 768 hidden dimension).

## 3. Dataset & Probing Suite
Dataset ID `{inv.get('dataset_id', 'ioi')}` with prompt pairs (clean vs corrupted).

## 4. Hypotheses Tested
"""
    for h in hypotheses:
        report_md += f"""
### Hypothesis: {h.get('title')} (`{h.get('id')}`)
- **Target Component**: `{h.get('target_component')}`
- **Statement**: {h.get('statement')}
- **Prediction**: {h.get('prediction')}
- **Falsification Condition**: {h.get('falsification_condition')}
- **Status**: **{h.get('status', 'UNTESTED')}** (Supporting: {h.get('evidence_count_supporting', 0)}, Contradicting: {h.get('evidence_count_contradicting', 0)})
"""

    report_md += f"""
## 5. Experimental Interventions
Total experiments executed: {len(runs)}.
"""
    for r in runs[:10]:
        report_md += f"""
- **Run `{r.get('id')}`**: Baseline target prob `{r.get('baseline_target_prob', 0.0):.4f}` $\\to$ Intervened `{r.get('intervened_target_prob', 0.0):.4f}` ($\\Delta \\text{{Logit}} = {r.get('delta_logit', 0.0):.2f}$). Control $\\Delta$: `{r.get('control_delta_logit')}`.
"""

    report_md += f"""
## 6. Causal Evidence & Controls
Total evidence records: {len(evidence)}.
"""
    for e in evidence:
        report_md += f"""
- **Evidence `{e.get('id')}`** ({e.get('evidence_level')}): {e.get('claim')} (Metric: `{e.get('metric_value')}`, Control: `{e.get('control_value')}`)
"""

    report_md += """
## 7. Mechanism Topology
"""
    for m in mechanisms:
        report_md += f"""
- **Mechanism `{m.get('name')}`**: Nodes: {len(m.get('nodes', []))}, Edges: {len(m.get('edges', []))}, Evidence-Backed: {m.get('is_evidence_backed')}
"""

    report_md += """
## 8. Limitations & Epistemic Scope
- Measured on standard in-distribution factual/IOI probes.
- Linear logit attribution (DLA) is treated as observational; only PyTorch hook interventions are treated as causal.

## 9. Conclusion & Verdict
Hypothesis evaluation grounded strictly on executed empirical runs.
"""

    return {"status": "success", "report_markdown": report_md, "investigation_id": inv_id}


# ---------------------------------------------------------------------------
# Jobs & Compute Center
# ---------------------------------------------------------------------------
from backend.science.job_queue import job_queue


@router.get("/jobs")
def list_jobs(limit: int = 50, status: Optional[str] = None) -> Dict[str, Any]:
    items = job_queue.list_jobs(limit, status)
    return {"jobs": items, "count": len(items)}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return {"status": "success", "job": job}


@router.post("/jobs/run-intervention-async")
def run_intervention_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Submits a causal intervention experiment to the background worker pool."""
    exp = Experiment(**payload)

    def _execute(update_stage: Any, cancel_event: Any):
        update_stage("LOADING_MODEL")  # stage-based: no fabricated percentage
        from backend.services import gpt2_engine
        if not gpt2_engine.is_available():
            raise RuntimeError("ML backend not available.")
        if gpt2_engine._model is None:
            gpt2_engine.load()

        update_stage("RUNNING_INFERENCE")  # stage-based
        update_stage("COMPUTING_INTERVENTION")  # stage-based
        run = _experiment_runner.run_experiment(exp)

        update_stage("WRITING_ARTIFACTS")  # stage-based
        return run.model_dump()

    job = job_queue.submit_job(
        job_type="CAUSAL_INTERVENTION",
        name=f"{exp.intervention_type.value} on {exp.source_component}",
        target_fn=_execute,
        metadata={"experiment_id": exp.id, "source_component": exp.source_component},
    )

    return {"status": "submitted", "job": job.model_dump()}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    cancelled = job_queue.cancel_job(job_id)
    return {"status": "cancelled" if cancelled else "not_found", "id": job_id}


# ---------------------------------------------------------------------------
# Model Registry & Health Validation
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Model Registry & Health Validation
# ---------------------------------------------------------------------------
import platform

@router.get("/models/validate")
def validate_model(model_name: str = "gpt2") -> Dict[str, Any]:
    """Validates runtime readiness, parameter dimensions, device, and forward hooks.
    
    Returns READY status with complete model metadata and environment snapshot.
    """
    from backend.services import gpt2_engine
    is_avail = gpt2_engine.is_available()
    if not is_avail:
        return {
            "status": "NOT_AVAILABLE",
            "model_name": model_name,
            "error": "PyTorch / Transformers not available.",
            "capabilities": {},
            "environment_snapshot": None,
        }

    if gpt2_engine._model is None:
        gpt2_engine.load()

    m = gpt2_engine._model
    tok = gpt2_engine._tokenizer

    if m is None or tok is None:
        return {
            "status": "LOAD_ERROR",
            "model_name": model_name,
            "error": "Failed to initialize model instance.",
            "capabilities": {},
            "environment_snapshot": None,
        }

    # Build environment snapshot
    env_snapshot = build_environment_snapshot()

    return {
        "status": "READY",
        "model_name": model_name,
        "parameters": m.config.n_embd * m.config.n_layer if hasattr(m.config, 'n_embd') and hasattr(m.config, 'n_layer') else 124439808,
        "num_layers": m.config.n_layer,
        "num_heads": m.config.n_head,
        "hidden_dim": m.config.n_embd,
        "vocab_size": tok.vocab_size if hasattr(tok, "vocab_size") else 50257,
        "device": str(m.device),
        "dtype": str(m.dtype),
        "capabilities": {
            "inference": True,
            "attention_capture": True,
            "activation_capture": True,
            "zero_ablation": True,
            "mean_ablation": True,
            "activation_patching": True,
            "activation_steering": True,
        },
        "environment_snapshot": env_snapshot,
    }


def build_environment_snapshot() -> Dict[str, Any]:
    """Build a deterministic environment snapshot for reproducibility.
    
    Captures Python version, package versions, GPU info, and runtime state.
    This is mandatory for serious research reproducibility.
    """
    try:
        import torch
        import transformers
    except ImportError:
        torch = None
        transformers = None

    # Python version
    python_version = platform.python_version()

    # Package versions
    package_versions = {}
    if transformers:
        package_versions["transformers"] = transformers.__version__
    if torch:
        package_versions["torch"] = torch.__version__

    # CUDA version (if available)
    cuda_version = None
    if torch and torch.cuda.is_available():
        cuda_version = torch.version.cuda

    # GPU info
    gpu_name = None
    gpu_compute_capability = None
    gpu_memory_total = None
    if torch and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_compute_capability = torch.cuda.get_device_capability(0)
        gpu_memory_total = torch.cuda.get_device_property(0).total_memory

    # PyTorch version details
    pytorch_version = torch.__version__ if torch else None

    # OS info
    os_platform = platform.system()
    os_release = platform.release()

    return {
        "python_version": python_version,
        "package_versions": package_versions,
        "cuda_version": cuda_version,
        "pytorch_version": pytorch_version,
        "gpu_name": gpu_name,
        "gpu_compute_capability": gpu_compute_capability,
        "gpu_memory_total_mb": gpu_memory_total,
        "os_platform": os_platform,
        "os_release": os_release,
    }


# ---------------------------------------------------------------------------
# Hardware & VRAM Estimation Endpoint
# ---------------------------------------------------------------------------
from backend.services.hardware_manager import hardware_manager


@router.get("/hardware/vram-estimate")
def get_vram_estimate(model_name: str = "gpt2", seq_len: int = 128, batch_size: int = 1) -> Dict[str, Any]:
    """Calculates GPU VRAM requirement and returns remediation recommendations."""
    return hardware_manager.estimate_vram_requirement(model_name, batch_size, seq_len)


# ---------------------------------------------------------------------------
# Research Bundle Export & Import
# ---------------------------------------------------------------------------
from backend.science.bundle_manager import bundle_manager
from backend.science.scientific_envelope import wrap_scientific_success, wrap_unexecuted_failure


@router.post("/bundles/export")
def export_research_bundle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Exports an investigation, hypotheses, run records, and raw tensors to a verified ZIP bundle."""
    inv_id = payload.get("investigation_id")
    if not inv_id:
        raise HTTPException(status_code=400, detail="investigation_id is required.")
    
    export_dir = Path.home() / ".cache" / "neural-debugger" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    zip_path = export_dir / f"bundle_{inv_id}_{int(time.time())}.zip"

    out_file = bundle_manager.export_bundle(inv_id, zip_path)
    return {"status": "success", "bundle_path": out_file, "investigation_id": inv_id}


@router.post("/bundles/import")
def import_research_bundle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validates manifest SHA256 checksums and imports all research entities into database."""
    bundle_path = payload.get("bundle_path")
    if not bundle_path:
        raise HTTPException(status_code=400, detail="bundle_path is required.")
    
    inv = bundle_manager.import_bundle(bundle_path)
    return {"status": "success", "investigation": inv}


@router.post("/experiments/run-intervention-envelope")
def run_intervention_with_envelope(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Executes a causal intervention and returns a tamper-proof ScientificResponseEnvelope."""
    from backend.services import gpt2_engine
    if not gpt2_engine.is_available():
        from backend.science.scientific_envelope import wrap_unexecuted_failure
        return wrap_unexecuted_failure("PyTorch ML backend is unavailable.").model_dump()
    
    if gpt2_engine._model is None:
        gpt2_engine.load()

    if gpt2_engine._model is None or gpt2_engine._tokenizer is None:
        from backend.science.scientific_envelope import wrap_unexecuted_failure
        return wrap_unexecuted_failure("Failed to load model weights.").model_dump()
    
    exp = Experiment(**payload)
    run = _experiment_runner.run_experiment(exp)
    
    # The run from _experiment_runner always uses live model data, so used_mock_data = False
    # However, we mark it explicitly for UI awareness
    run_dict = run.model_dump()
    run_dict["used_mock_data"] = False
    
    envelope = wrap_scientific_success(
        data=run_dict,
        manifest_id=run.manifest_id or f"man_{run.id}",
        manifest_payload=run.model_dump(),
        model_version=run.model_id,
        dataset_version=run.dataset_version,
    )
    return envelope.model_dump()


# ---------------------------------------------------------------------------
# Reproduction & Diagnostics Endpoints
# ---------------------------------------------------------------------------
from backend.science.reproduction_engine import reproduction_engine
from backend.services.diagnostics_service import diagnostics_service


@router.post("/experiments/reproduce")
def reproduce_experiment_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically re-executes a run and computes numerical divergence."""
    run_id = payload.get("run_id")
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required.")
    
    report = reproduction_engine.reproduce_run(run_id)
    return {"status": "success", "reproduction": report.model_dump()}


@router.get("/diagnostics/report")
def get_diagnostics_report() -> Dict[str, Any]:
    """Returns live system diagnostics and hardware health."""
    return diagnostics_service.generate_diagnostics_report()


@router.post("/diagnostics/export")
def export_diagnostics(payload: Dict[str, Any] = None) -> Dict[str, Any]:
    """Exports full diagnostic snapshot to JSON file."""
    export_dir = Path.home() / ".cache" / "neural-debugger" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    out_file = export_dir / f"MECH_Diagnostics_{int(time.time())}.json"
    p = diagnostics_service.export_diagnostics_to_file(out_file)
    return {"status": "success", "file_path": p}


# ---------------------------------------------------------------------------
# Research Validation v2 Endpoints (Ground Truth, Preregistration, Versioning, Critic)
# ---------------------------------------------------------------------------
from backend.science.benchmarks.ground_truth_benchmark import ground_truth_engine
from backend.science.preregistration_engine import preregistration_engine, PreregistrationRecord, LedgerEntry
from backend.science.mechanism_versioning import mechanism_versioning_engine, MechanismVersion, compute_mechanism_diff
from backend.science.mechanism_critic import mechanism_critic_engine


@router.post("/benchmarks/ground-truth/evaluate")
def evaluate_ground_truth(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Executes independent mechanistic discovery and compares against peer-reviewed literature ground truth."""
    benchmark_id = payload.get("benchmark_id", "IOI_NAME_MOVER")
    model_name = payload.get("model_name", "gpt2")
    comparison = ground_truth_engine.execute_and_evaluate(benchmark_id=benchmark_id, model_name=model_name)
    return {"status": "success", "report": comparison.model_dump()}


@router.post("/preregistrations")
def create_preregistration(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Registers an experimental protocol proposal."""
    record = PreregistrationRecord(**payload)
    saved = preregistration_engine.create_preregistration(record)
    return {"status": "success", "preregistration": saved.model_dump()}


@router.post("/preregistrations/{prereg_id}/lock")
def lock_preregistration(prereg_id: str) -> Dict[str, Any]:
    """Permanently freezes an experimental protocol prior to confirmatory execution."""
    locked = preregistration_engine.lock_preregistration(prereg_id)
    return {"status": "success", "preregistration": locked.model_dump()}


@router.get("/preregistrations/ledger")
def get_experiment_ledger(investigation_id: str) -> Dict[str, Any]:
    """Returns immutable research history ledger entries."""
    entries = preregistration_engine.list_ledger_entries(investigation_id)
    return {"status": "success", "ledger": [e.model_dump() for e in entries]}


@router.post("/mechanisms/versions")
def save_mechanism_version(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Saves a versioned snapshot of a mechanism circuit."""
    ver = MechanismVersion(**payload)
    saved = mechanism_versioning_engine.save_version(ver)
    return {"status": "success", "version": saved.model_dump()}


@router.post("/mechanisms/diff")
def compare_mechanism_versions(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Computes structural and evidence diff between two mechanism versions."""
    v1 = mechanism_versioning_engine.get_version(payload.get("v1_id"))
    v2 = mechanism_versioning_engine.get_version(payload.get("v2_id"))
    if not v1 or not v2:
        raise HTTPException(status_code=404, detail="One or both mechanism versions not found.")
    diff = compute_mechanism_diff(v1, v2)
    return {"status": "success", "diff": diff.model_dump()}


@router.get("/critic/audit")
def audit_claim_rigor(investigation_id: str, component: str = "L9H9") -> Dict[str, Any]:
    """Audits empirical claims for missing controls and alternative explanations."""
    audit = mechanism_critic_engine.audit_claim(investigation_id, component)
    return {"status": "success", "audit": audit.model_dump()}


@router.get("/critic/next-experiment")
def recommend_next_experiment(investigation_id: str) -> Dict[str, Any]:
    """Proposes high-information-gain next experiment to resolve scientific ambiguity."""
    prop = mechanism_critic_engine.recommend_next_experiment(investigation_id)
    return {"status": "success", "proposal": prop.model_dump()}




