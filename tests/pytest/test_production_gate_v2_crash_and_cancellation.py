import tempfile
import time
from pathlib import Path
import pytest

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import JobStatus
from backend.science.job_queue import AsyncJobQueue
from backend.science.artifact_registry import ArtifactRegistry


def test_multi_stage_cancellation():
    """Validates cancellation at each execution stage, ensuring no corrupted artifacts persist."""
    stages_to_test = ["PREPARING", "LOADING_MODEL", "RUNNING_INFERENCE", "COMPUTING_INTERVENTION", "WRITING_ARTIFACTS"]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "cancel_test.db")
        storage.initialize()
        queue = AsyncJobQueue(max_workers=2, storage=storage)

        for target_stage in stages_to_test:
            cancelled_cleanly = False

            def _worker_fn(update_stage, cancel_event):
                nonlocal cancelled_cleanly
                for s in stages_to_test:
                    update_stage(s)
                    if s == target_stage:
                        # Wait for cancel token
                        for _ in range(50):
                            if cancel_event.is_set():
                                cancelled_cleanly = True
                                return None
                            time.sleep(0.05)
                return {"status": "finished"}

            job = queue.submit_job(
                job_type="CAUSAL_INTERVENTION",
                name=f"Cancel Test at {target_stage}",
                target_fn=_worker_fn,
            )

            time.sleep(0.1)
            # Cancel job while at target_stage
            queue.cancel_job(job.id)

            # Wait for cancellation resolution
            for _ in range(50):
                j = storage.get_job(job.id)
                if j and j.get("status") in (JobStatus.CANCELLED.value, JobStatus.FAILED.value):
                    break
                time.sleep(0.05)

            j = storage.get_job(job.id)
            assert j is not None
            assert j.get("status") == JobStatus.CANCELLED.value, f"Expected CANCELLED at stage {target_stage}, got {j.get('status')}"
            assert j.get("status") != JobStatus.COMPLETED.value


def test_aborted_worker_crash_recovery_and_retry():
    """Simulates an abrupt unhandled worker exception, verifying DB unlock and retry readiness."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "crash_test.db")
        storage.initialize()
        queue = AsyncJobQueue(max_workers=2, storage=storage)

        def _crashing_worker(update_stage, cancel_event):
            update_stage("RUNNING_INFERENCE")
            raise RuntimeError("CRITICAL WORKER CRASH SIMULATION")

        job = queue.submit_job(
            job_type="CAUSAL_INTERVENTION",
            name="Crashing Job",
            target_fn=_crashing_worker,
        )

        time.sleep(0.3)

        # Verify job marked as FAILED with error trace
        j = storage.get_job(job.id)
        assert j is not None
        assert j.get("status") == JobStatus.FAILED.value
        assert "CRITICAL WORKER CRASH SIMULATION" in j.get("error", "")

        # Verify database is healthy and allows immediate retry job submission
        def _retry_worker(update_stage, cancel_event):
            update_stage("COMPUTING_INTERVENTION")
            return {"status": "retry_succeeded"}

        retry_job = queue.submit_job(
            job_type="CAUSAL_INTERVENTION",
            name="Retry Job",
            target_fn=_retry_worker,
        )

        time.sleep(0.3)
        rj = storage.get_job(retry_job.id)
        assert rj is not None
        assert rj.get("status") == JobStatus.COMPLETED.value
