import tempfile
from pathlib import Path
import pytest

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import (
    Investigation,
    Hypothesis,
    Experiment,
    ExperimentRun,
    InterventionType,
)
from backend.science.experiment_runner import ScientificExperimentRunner
from backend.science.reproduction_engine import ReproductionEngine
from backend.services import gpt2_engine


def test_live_experiment_reproduction_and_differential():
    """Executes a live intervention run, reproduces it, and verifies numerical differential."""
    if not gpt2_engine.is_available():
        pytest.skip("PyTorch / Transformers not available.")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "rep_test.db"
        storage = DesktopStorage(db_path)
        storage.initialize()

        storage.save_investigation(
            Investigation(id="inv_rep_e2e", title="Reproduction Investigation", research_question="Testing reproduction?").model_dump()
        )

        exp = Experiment(
            id="exp_rep_source",
            investigation_id="inv_rep_e2e",
            name="Original Run for Reproduction",
            clean_prompt="When Mary and John went to the store, John gave a drink to Mary",
            corrupted_prompt="When Mary and John went to the store, Mary gave a drink to John",
            target_token=" Mary",
            distractor_token=" John",
            intervention_type=InterventionType.ABLATION_ZERO,
            source_component="L9H9",
            control_component="L0H0",
        )

        runner = ScientificExperimentRunner(storage=storage)
        orig_run = runner.run_experiment(exp)
        assert orig_run.id is not None

        # Execute Reproduction Engine
        engine = ReproductionEngine(storage=storage)
        report = engine.reproduce_run(orig_run.id)

        assert report.original_run_id == orig_run.id
        assert report.is_deterministic is True
        assert report.overall_status.value in ("BITWISE_IDENTICAL", "NUMERICALLY_REPRODUCED", "WITHIN_TOLERANCE")
        assert len(report.metric_differentials) == 2
        assert report.delta_logit_differential < 0.001
        assert report.manifest_match is True
