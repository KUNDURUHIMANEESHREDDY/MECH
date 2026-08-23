import time
import pytest
from pathlib import Path
import tempfile

from backend.storage.database import DesktopStorage
from backend.science.job_queue import AsyncJobQueue
from backend.storage.scientific_entities import JobStatus


def test_async_job_queue_lifecycle():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_jobs.db"
        storage = DesktopStorage(db_path)
        storage.initialize()

        queue = AsyncJobQueue(max_workers=2, storage=storage)

        def mock_work(update_stage, cancel_event):
            update_stage("STEP_1", 0.3)
            time.sleep(0.05)
            update_stage("STEP_2", 0.7)
            time.sleep(0.05)
            return {"id": "res_123", "metric": 42.0}

        job = queue.submit_job("TEST_TASK", "Test Async Task", mock_work)
        assert job.id is not None
        assert job.status == JobStatus.QUEUED

        # Wait briefly for completion
        time.sleep(0.3)

        finished_job = queue.get_job(job.id)
        assert finished_job is not None
        assert finished_job["status"] == JobStatus.COMPLETED.value
        assert finished_job["progress"] == 1.0
        assert finished_job["current_stage"] == "COMPLETED"


def test_async_job_cancellation():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_cancel.db"
        storage = DesktopStorage(db_path)
        storage.initialize()

        queue = AsyncJobQueue(max_workers=1, storage=storage)

        def long_work(update_stage, cancel_event):
            for i in range(20):
                if cancel_event.is_set():
                    break
                update_stage(f"STEP_{i}", i / 20.0)
                time.sleep(0.05)
            return {"cancelled": True}

        job = queue.submit_job("LONG_TASK", "Long Cancellable Task", long_work)
        time.sleep(0.08)

        cancelled = queue.cancel_job(job.id)
        assert cancelled is True

        time.sleep(0.15)
        cancelled_job = queue.get_job(job.id)
        assert cancelled_job["status"] == JobStatus.CANCELLED.value
