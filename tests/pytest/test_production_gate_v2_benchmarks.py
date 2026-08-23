import gc
import tempfile
import time
from pathlib import Path
import pytest
import torch

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import (
    Experiment,
    Investigation,
    InterventionType,
)
from backend.science.artifact_registry import ArtifactRegistry
from backend.science.experiment_runner import ScientificExperimentRunner
from backend.services import gpt2_engine


def test_performance_latency_benchmarks():
    """Measures and benchmarks latency across database, storage, and model hooks."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "bench.db")
        storage.initialize()
        registry = ArtifactRegistry(base_dir=Path(tmpdir) / "artifacts", storage=storage)

        # 1. Database write latency
        t0 = time.perf_counter()
        for i in range(20):
            storage.save_investigation(
                Investigation(id=f"inv_bench_{i}", title=f"Bench {i}", research_question="Speed?").model_dump()
            )
        db_write_latency_ms = ((time.perf_counter() - t0) / 20) * 1000.0
        assert db_write_latency_ms < 50.0, f"DB write too slow: {db_write_latency_ms:.2f} ms"

        # 2. Atomic tensor write latency
        dummy_tensor = torch.randn(10, 12, 64)
        t0 = time.perf_counter()
        art = registry.store_tensor_artifact(dummy_tensor, "Bench Tensor", "inv_bench_0")
        write_latency_ms = (time.perf_counter() - t0) * 1000.0
        assert write_latency_ms < 100.0, f"Atomic tensor write too slow: {write_latency_ms:.2f} ms"

        # 3. Checksum verification read latency
        t0 = time.perf_counter()
        _ = registry.load_tensor_artifact(art.id, verify_checksum=True)
        read_latency_ms = (time.perf_counter() - t0) * 1000.0
        assert read_latency_ms < 100.0, f"Tensor read verification too slow: {read_latency_ms:.2f} ms"


def test_memory_leak_and_gc_stability():
    """Runs consecutive intervention iterations and verifies memory stability without leak."""
    if not gpt2_engine.is_available():
        pytest.skip("PyTorch / Transformers not available.")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "mem_test.db")
        storage.initialize()
        runner = ScientificExperimentRunner(storage=storage)

        storage.save_investigation(
            Investigation(id="inv_mem", title="Memory Leak Test", research_question="Does RAM leak?").model_dump()
        )

        exp = Experiment(
            id="exp_mem",
            investigation_id="inv_mem",
            name="Leak Test Intervention",
            clean_prompt="The quick brown fox jumps over the lazy dog",
            target_token=" dog",
            intervention_type=InterventionType.ABLATION_ZERO,
            source_component="L9H9",
        )

        # Baseline GC count
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Execute 5 consecutive iterations
        for _ in range(5):
            _ = runner.run_experiment(exp)

        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count growth should be modest / bounded
        growth = final_objects - initial_objects
        assert growth < 5000, f"Excessive object growth detected: {growth} objects retained."
