import gc
import tempfile
from pathlib import Path
import pytest

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import (
    Experiment,
    Investigation,
    InterventionType,
)
from backend.science.experiment_runner import ScientificExperimentRunner
from backend.services import gpt2_engine


def test_extended_25_iteration_memory_stability():
    """Runs 25 consecutive model forward pass and intervention cycles to verify memory stability."""
    if not gpt2_engine.is_available():
        pytest.skip("PyTorch / Transformers not available.")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "long_mem_test.db")
        storage.initialize()
        runner = ScientificExperimentRunner(storage=storage)

        storage.save_investigation(
            Investigation(id="inv_long_mem", title="25-Iteration Memory Test", research_question="Does RAM leak over 25 runs?").model_dump()
        )

        exp = Experiment(
            id="exp_long_mem",
            investigation_id="inv_long_mem",
            name="Extended Memory Run",
            clean_prompt="When Mary and John went to the store, John gave a drink to Mary",
            corrupted_prompt="When Mary and John went to the store, Mary gave a drink to John",
            target_token=" Mary",
            distractor_token=" John",
            intervention_type=InterventionType.ABLATION_ZERO,
            source_component="L9H9",
            control_component="L0H0",
        )

        gc.collect()
        initial_objects = len(gc.get_objects())

        # Execute 25 continuous passes
        for _ in range(25):
            _ = runner.run_experiment(exp)

        gc.collect()
        final_objects = len(gc.get_objects())

        growth = final_objects - initial_objects
        # Verify object retention is strictly bounded
        assert growth < 10000, f"Excessive object retention over 25 runs: {growth} objects."
