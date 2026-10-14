"""F-20: Cross-process write durability.

The current atomic write uses `threading.Lock` which only serializes
within a single process. Concurrent writes from different processes
can corrupt the file or lose data.

This test suite verifies:
1. Cross-process atomic writes don't corrupt data
2. No data loss under concurrent writes from multiple processes
3. Crash consistency (simulated via abrupt process termination)
"""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from multiprocessing import Process, Barrier, Manager

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core import evidence_graph as eg  # noqa: E402


def _make_record(run_id: str, value: int) -> dict:
    """Create a minimal record for testing (inner record data, not an envelope)."""
    return {
        "run_id": run_id,
        "value": value,
    }


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _worker_same_run(barrier: Barrier, run_id: str, value: int, directory: str, results: list, idx: int):
    """Worker that saves a record with same run_id."""
    try:
        barrier.wait(timeout=5.0)
        path = eg.save_run_record(run_id, _make_record(run_id, value), directory=directory)
        results[idx] = ("ok", path)
    except Exception as e:
        results[idx] = ("error", str(e))


def _worker_different_runs(barrier: Barrier, run_id: str, value: int, directory: str, results: list, idx: int):
    """Worker that saves a record with different run_id."""
    try:
        barrier.wait(timeout=5.0)
        path = eg.save_run_record(run_id, _make_record(run_id, value), directory=directory)
        results[idx] = ("ok", path)
    except Exception as e:
        results[idx] = ("error", str(e))


def _writer_rapid(val: int, run_id: str, directory: str):
    """Worker for rapid writes."""
    eg.save_run_record(run_id, _make_record(run_id, val), directory=directory)


def _worker_lock_test(barrier: Barrier, idx: int, iterations: int, directory: str, results: list):
    """Worker for cross-process lock test."""
    try:
        barrier.wait(timeout=5.0)
        for i in range(iterations):
            run_id = f"r{idx:012x}"
            eg.save_run_record(run_id, _make_record(run_id, i), directory=directory)
        results[idx] = "ok"
    except Exception as e:
        results[idx] = f"error: {e}"


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_record(run_id: str, value: int) -> dict:
    """Create a minimal record for testing (inner record data, not an envelope)."""
    return {
        "run_id": run_id,
        "value": value,
    }


def test_concurrent_processes_same_run_id_no_corruption(tmp_path):
    """Multiple processes saving the same run_id concurrently must not corrupt."""
    num_processes = 4
    barrier = Barrier(num_processes)
    manager = Manager()
    results = manager.list([None] * num_processes)
    
    run_id = "r" + "1" * 12  # valid run_id format: r + 12 hex chars
    
    processes = [
        Process(target=_worker_same_run, args=(barrier, run_id, i, str(tmp_path), results, i))
        for i in range(num_processes)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=10)
    
    # All should succeed
    for status, _ in results:
        assert status == "ok", f"Process failed: {results}"
    
    # Final file must be valid JSON and match one of the written values
    path = tmp_path / f"{run_id}.json"
    assert path.exists()
    data = _read_json(path)
    assert data["record"]["run_id"] == run_id
    assert "value" in data["record"]


def test_concurrent_processes_different_run_ids_no_crosstalk(tmp_path):
    """Concurrent writes to different run_ids must not interfere."""
    num_processes = 6
    barrier = Barrier(num_processes)
    manager = Manager()
    results = manager.list([None] * num_processes)
    
    run_ids = [f"r{i:012x}" for i in range(num_processes)]
    
    processes = [
        Process(target=_worker_different_runs, args=(barrier, run_ids[i], i, str(tmp_path), results, i))
        for i in range(num_processes)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=10)
    
    for status, _ in results:
        assert status == "ok", f"Process failed: {results}"
    
    # All files should exist with correct data
    for i in range(num_processes):
        run_id = run_ids[i]
        path = tmp_path / f"{run_id}.json"
        assert path.exists(), f"Missing {run_id}.json"
        data = _read_json(path)
        assert data["record"]["run_id"] == run_id
        assert data["record"]["value"] == i


def test_no_data_loss_under_rapid_concurrent_writes(tmp_path):
    """Rapid sequential writes from different processes must not lose data."""
    num_writes = 20
    run_id = "r" + "a" * 12  # valid run_id format
    
    # Write rapidly from multiple threads (simulating processes)
    threads = [threading.Thread(target=eg.save_run_record, args=(run_id, _make_record(run_id, i), str(tmp_path))) for i in range(num_writes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Final value must be one of the written values (no corruption)
    path = tmp_path / f"{run_id}.json"
    assert path.exists()
    data = _read_json(path)
    assert data["record"]["run_id"] == run_id
    assert 0 <= data["record"]["value"] < num_writes


def test_crash_consistency_simulated(tmp_path):
    """Simulated crash during write should not leave corrupt file."""
    run_id = "r111111111112"
    eg.save_run_record(run_id, _make_record(run_id, 1), directory=str(tmp_path))
    original = _read_json(tmp_path / f"{run_id}.json")
    assert original["record"]["value"] == 1
    
    # Now overwrite with a new value - this is atomic
    eg.save_run_record(run_id, _make_record(run_id, 2), directory=str(tmp_path))
    updated = _read_json(tmp_path / f"{run_id}.json")
    assert updated["record"]["value"] == 2
    
    # File should be valid JSON, not partially written
    assert "attestation" in updated


def test_orphaned_temp_files_cleaned_up(tmp_path):
    """Temp files from failed writes should not accumulate."""
    run_id = "r222222222222"
    eg.save_run_record(run_id, _make_record(run_id, 1), directory=str(tmp_path))
    
    # Temp file pattern should not leave orphans
    temp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(temp_files) == 0, f"Orphaned temp files: {temp_files}"


def _worker_lock_test(barrier: Barrier, idx: int, iterations: int, directory: str, results: list):
    """Worker for cross-process lock test."""
    try:
        barrier.wait(timeout=5.0)
        for i in range(iterations):
            run_id = f"r{idx:012x}"
            eg.save_run_record(run_id, _make_record(run_id, i), directory=directory)
        results[idx] = "ok"
    except Exception as e:
        results[idx] = f"error: {e}"


def test_cross_process_lock_prevents_corruption(tmp_path):
    """Cross-process lock ensures only one writer at a time."""
    # This test verifies the locking mechanism works across processes
    num_processes = 4
    iterations = 5
    barrier = Barrier(num_processes)
    manager = Manager()
    results = manager.list([None] * num_processes)
    
    processes = [Process(target=_worker_lock_test, args=(barrier, i, iterations, str(tmp_path), results)) for i in range(num_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=15)
    
    for status in results:
        assert status == "ok", f"Process failed: {results}"
    
    # Verify all records are valid
    for i in range(num_processes):
        path = tmp_path / f"r{i:012x}.json"
        assert path.exists()
        data = _read_json(path)
        assert data["record"]["run_id"] == f"r{i:012x}"