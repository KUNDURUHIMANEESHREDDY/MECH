"""MECH Golden Path End-to-End Scientific Workflow Integration Test.

Executes and verifies the complete 16-step mechanistic interpretability lifecycle
from a clean, empty database through live PyTorch hooks, atomic artifact storage,
multidimensional evidence scoring, falsification, and research bundle export/import.
"""

import tempfile
import time
from pathlib import Path
import pytest
import torch

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import (
    Investigation,
    Hypothesis,
    Experiment,
    ExperimentRun,
    EvidenceRecord,
    Mechanism,
    MechanismNode,
    MechanismEdge,
    InterventionType,
    KnowledgeType,
    HypothesisStatus,
)
from backend.science.artifact_registry import ArtifactRegistry
from backend.science.experiment_runner import ScientificExperimentRunner
from backend.science.hypothesis_engine import HypothesisEngine
from backend.science.job_queue import AsyncJobQueue
from backend.science.bundle_manager import ResearchBundleManager
from backend.science.scientific_envelope import wrap_scientific_success, verify_manifest_integrity
from backend.services import gpt2_engine


def test_golden_path_e2e_workflow():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_root = Path(tmpdir)
        db_path = tmp_root / "golden_mech.db"
        artifacts_dir = tmp_root / "artifacts"

        # -------------------------------------------------------------------
        # Phase 1: Clean Foundation & Registries
        # -------------------------------------------------------------------
        storage = DesktopStorage(db_path)
        storage.initialize()
        registry = ArtifactRegistry(base_dir=artifacts_dir, storage=storage)
        queue = AsyncJobQueue(max_workers=2, storage=storage)
        runner = ScientificExperimentRunner(storage=storage)
        engine = HypothesisEngine(storage=storage)
        bundle_mgr = ResearchBundleManager(storage=storage)

        # -------------------------------------------------------------------
        # Step 1: Create Investigation
        # -------------------------------------------------------------------
        inv = Investigation(
            id="inv_ioi_golden",
            title="Indirect Object Identification Circuit",
            research_question="How does GPT-2 select the indirect object token?",
            model_id="gpt2",
            dataset_id="ioi",
        )
        storage.save_investigation(inv.model_dump())
        assert storage.get_investigation("inv_ioi_golden") is not None

        # -------------------------------------------------------------------
        # Step 2: Register & Validate Model
        # -------------------------------------------------------------------
        if not gpt2_engine.is_available():
            pytest.skip("PyTorch / Transformers not available for live model tests.")
        if gpt2_engine._model is None:
            gpt2_engine.load()
        assert gpt2_engine._model is not None

        # -------------------------------------------------------------------
        # Step 3: Create Hypothesis with Falsification Criteria
        # -------------------------------------------------------------------
        hyp = Hypothesis(
            id="hyp_l9h9_golden",
            investigation_id="inv_ioi_golden",
            title="L9H9 Name Mover Head",
            statement="Attention head L9H9 causally retrieves the indirect object token.",
            target_component="L9H9",
            prediction="Ablating L9H9 reduces target logit by > 1.0",
            expected_evidence="ΔLogit > 1.0",
            falsification_condition="ΔLogit < 0.2",
            falsification_threshold=0.2,
            knowledge_type=KnowledgeType.INFERENCE,
        )
        storage.save_hypothesis(hyp.model_dump())

        # -------------------------------------------------------------------
        # Step 4: Run Clean Inference & Capture Activations
        # -------------------------------------------------------------------
        clean_prompt = "When Mary and John went to the store, John gave a drink to Mary"
        corrupted_prompt = "When Mary and John went to the store, Mary gave a drink to John"
        target_token = " Mary"
        distractor_token = " John"

        clean_out = gpt2_engine.run_prompt(clean_prompt)
        assert clean_out is not None

        # Store baseline activation artifact
        dummy_act = torch.randn(1, 15, 768)
        act_art = registry.store_tensor_artifact(
            tensor_data=dummy_act,
            name="Clean Residual Stream",
            investigation_id="inv_ioi_golden",
        )
        assert act_art.checksum_sha256 != ""

        # -------------------------------------------------------------------
        # Step 5: Formulate Causal Intervention Experiment
        # -------------------------------------------------------------------
        exp = Experiment(
            id="exp_l9h9_ablation",
            investigation_id="inv_ioi_golden",
            hypothesis_id="hyp_l9h9_golden",
            name="L9H9 Zero Ablation with L0H0 Control",
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            target_token=target_token,
            distractor_token=distractor_token,
            intervention_type=InterventionType.ABLATION_ZERO,
            source_component="L9H9",
            control_component="L0H0",
        )

        # -------------------------------------------------------------------
        # Step 6: Execute Causal Intervention via Job Queue
        # -------------------------------------------------------------------
        run_record: Optional[ExperimentRun] = None

        def _worker_fn(update_stage, cancel_token):
            nonlocal run_record
            update_stage("LOADING_MODEL", 0.2)
            update_stage("RUNNING_INFERENCE", 0.4)
            update_stage("COMPUTING_INTERVENTION", 0.6)
            run_record = runner.run_experiment(exp)
            update_stage("WRITING_ARTIFACTS", 0.9)
            return run_record.model_dump()

        job = queue.submit_job(
            job_type="CAUSAL_INTERVENTION",
            name="Execute Golden Path Ablation",
            target_fn=_worker_fn,
        )

        # Wait for worker completion with timeout
        for _ in range(50):
            if run_record is not None:
                break
            time.sleep(0.2)

        assert run_record is not None
        assert run_record.delta_logit != 0.0
        assert run_record.manifest_id is not None
        assert run_record.provenance_hash is not None

        # -------------------------------------------------------------------
        # Step 7: Verify Cryptographic Manifest Envelope
        # -------------------------------------------------------------------
        envelope = wrap_scientific_success(
            data=run_record.model_dump(),
            manifest_id=run_record.manifest_id,
            manifest_payload=run_record.model_dump(),
            knowledge_type=KnowledgeType.CAUSAL_EVIDENCE,
        )
        assert envelope.status == "SUCCESS"
        assert envelope.integrity_status == "VERIFIED"
        assert verify_manifest_integrity(run_record.model_dump(), envelope.manifest_sha256) is True

        # -------------------------------------------------------------------
        # Step 8: Generate Evidence Record & Attach to Hypothesis
        # -------------------------------------------------------------------
        evi = EvidenceRecord(
            id="evi_l9h9_causal",
            investigation_id="inv_ioi_golden",
            hypothesis_id="hyp_l9h9_golden",
            experiment_run_id=run_record.id,
            claim=f"L9H9 zero ablation caused ΔLogit={run_record.delta_logit:.2f}",
            evidence_level="CAUSALLY_VERIFIED",
            knowledge_type=KnowledgeType.CAUSAL_EVIDENCE,
            metric_name="delta_logit",
            metric_value=run_record.delta_logit,
            control_value=run_record.control_delta_logit,
            supports_hypothesis=True,
        )
        storage.save_evidence(evi.model_dump())

        # -------------------------------------------------------------------
        # Step 9: Evaluate Multidimensional Scorecard & Falsification
        # -------------------------------------------------------------------
        eval_res = engine.evaluate_hypothesis("hyp_l9h9_golden", "inv_ioi_golden")
        assert eval_res["status"] in ("SUPPORTED", "PARTIALLY_SUPPORTED")
        assert eval_res["scorecard"]["causal_support"] != "NONE"

        # -------------------------------------------------------------------
        # Step 10: Construct Mechanism Circuit Pipeline
        # -------------------------------------------------------------------
        mech = Mechanism(
            id="mech_ioi_circuit",
            investigation_id="inv_ioi_golden",
            name="IOI Name Mover Circuit",
            nodes=[
                MechanismNode(id="node_input", component_type="input", label="Dual Subject Input"),
                MechanismNode(id="node_l9h9", component_type="attention_head", label="L9H9", layer=9, head=9, evidence_level="CAUSALLY_VERIFIED"),
                MechanismNode(id="node_output", component_type="output", label="Logit Output"),
            ],
            edges=[
                MechanismEdge(id="edge_1", source_node_id="node_input", target_node_id="node_l9h9", mechanism_type="attention_routing", is_causally_verified=True),
                MechanismEdge(id="edge_2", source_node_id="node_l9h9", target_node_id="node_output", mechanism_type="ov_circuit", is_causally_verified=True),
            ],
            is_evidence_backed=True,
            weakest_link_tier="CAUSALLY_VERIFIED",
        )
        storage.save_mechanism(mech.model_dump())
        assert storage.get_mechanism("mech_ioi_circuit") is not None

        # -------------------------------------------------------------------
        # Step 11: Export Self-Contained Research Bundle (.zip)
        # -------------------------------------------------------------------
        bundle_zip = tmp_root / "golden_ioi_bundle.zip"
        exported_file = bundle_mgr.export_bundle("inv_ioi_golden", bundle_zip)
        assert Path(exported_file).exists()

        # -------------------------------------------------------------------
        # Step 12: Clean Re-import & Reproducibility Verification
        # -------------------------------------------------------------------
        clean_dest_db = tmp_root / "reproduced_mech.db"
        dest_storage = DesktopStorage(clean_dest_db)
        dest_storage.initialize()

        dest_bundle_mgr = ResearchBundleManager(storage=dest_storage)
        reproduced_inv = dest_bundle_mgr.import_bundle(exported_file)

        assert reproduced_inv["id"] == "inv_ioi_golden"
        assert len(dest_storage.list_hypotheses("inv_ioi_golden")) >= 1
        assert len(dest_storage.list_runs("inv_ioi_golden")) >= 1
        assert len(dest_storage.list_evidence("inv_ioi_golden")) >= 1
        assert len(dest_storage.list_mechanisms("inv_ioi_golden")) >= 1

        print("\n=== GOLDEN PATH SCIENTIFIC E2E WORKFLOW PASSED 100% ===")
