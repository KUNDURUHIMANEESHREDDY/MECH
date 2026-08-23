import pytest
from pathlib import Path
import tempfile
import torch

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import Experiment, InterventionType, Hypothesis
from backend.science.experiment_runner import ScientificExperimentRunner
from backend.science.hypothesis_engine import HypothesisEngine


def test_scientific_experiment_runner_ablation():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_mech.db"
        storage = DesktopStorage(db_path)
        storage.initialize()

        # Seed investigation and hypothesis
        inv_data = {
            "id": "inv_test",
            "title": "Test IOI Investigation",
            "research_question": "Does L9H9 drive name moving?",
            "model_id": "gpt2",
            "dataset_id": "ioi",
            "tags": ["test"],
        }
        storage.save_investigation(inv_data)

        hyp_data = {
            "id": "hyp_test",
            "investigation_id": "inv_test",
            "title": "L9H9 Name Mover Test",
            "statement": "L9H9 moves name token.",
            "target_component": "L9H9",
            "prediction": "Ablation causes delta logit > 1.0",
            "expected_evidence": "Positive delta",
            "falsification_condition": "Delta < 0.2",
        }
        storage.save_hypothesis(hyp_data)

        runner = ScientificExperimentRunner(storage=storage)

        exp = Experiment(
            name="Test Zero Ablation L9H9",
            investigation_id="inv_test",
            hypothesis_id="hyp_test",
            clean_prompt="When Mary and John went to the store, John gave a drink to",
            target_token=" Mary",
            intervention_type=InterventionType.ABLATION_ZERO,
            source_component="L9H9",
            control_component="L0H0",
        )

        run = runner.run_experiment(exp)

        assert run.experiment_id == exp.id
        assert run.investigation_id == "inv_test"
        assert run.provenance_hash is not None
        assert len(run.provenance_hash) == 64
        assert run.baseline_logit != 0.0
        assert run.top_predicted_tokens_clean is not None
        assert run.top_predicted_tokens_intervened is not None

        # Verify evidence record was automatically created
        evidence = storage.list_evidence_records(investigation_id="inv_test", hypothesis_id="hyp_test")
        assert len(evidence) == 1
        assert evidence[0]["source_type"] == "COMPUTED"

        # Verify hypothesis evaluation engine
        hyp_engine = HypothesisEngine(storage=storage)
        eval_res = hyp_engine.evaluate_hypothesis("hyp_test", "inv_test")
        assert eval_res["status"] in ("SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED")
        assert eval_res["supporting_count"] + eval_res["contradicting_count"] == 1
