"""Adversarial stress-testing suite for Milestone 1: Server Orchestration & Lifecycle.

Empirically challenges:
1. FastAPI lifespan startup and shutdown cycles (repeatability, background task cancellation, exception safety).
2. Concurrency stress and header fuzzing on GET /health (concurrent requests, oversized headers, origin regex bypass attempts, unexpected bodies).
3. SQLite WAL checkpointing under active load and transaction lock contention.
4. Process signal termination and tree cleanup behavior.
"""
from __future__ import annotations

import asyncio
import gc
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main
from backend.main import app, lifespan
from backend.storage.database import DesktopStorage, checkpoint_wal


# ============================================================================
# 1. LIFESPAN STARTUP & SHUTDOWN CYCLE ADVERSARIAL CHALLENGES
# ============================================================================

def test_repeated_lifespan_cycles():
    """Verify that entering and exiting the lifespan context multiple times

    does not leak tasks, lock the database, or corrupt application state.
    """
    for cycle in range(3):
        with TestClient(app) as client:
            res = client.get("/health")
            assert res.status_code == 200
            assert res.json() == {"status": "healthy"}


def test_lifespan_with_preload_models_cancellation():
    """Verify that if MECH_PRELOAD_MODELS=1 is set, shutting down the server

    promptly cancels the in-flight preload task without hanging or throwing unhandled errors.
    """
    old_env = os.environ.get("MECH_PRELOAD_MODELS")
    os.environ["MECH_PRELOAD_MODELS"] = "1"
    try:
        start_time = time.time()
        with TestClient(app) as client:
            res = client.get("/health")
            assert res.status_code == 200
        duration = time.time() - start_time
        # Shutdown should complete quickly and not hang indefinitely waiting for model download
        assert duration < 10.0, f"Lifespan shutdown took too long: {duration}s"
    finally:
        if old_env is None:
            os.environ.pop("MECH_PRELOAD_MODELS", None)
        else:
            os.environ["MECH_PRELOAD_MODELS"] = old_env


@pytest.mark.asyncio
async def test_lifespan_resilience_to_checkpoint_error(monkeypatch):
    """Verify that if SQLite checkpoint_wal raises an unexpected error during shutdown,

    the lifespan context manager catches the exception, logs a warning, and completes teardown.
    """
    def broken_checkpoint(*args, **kwargs):
        raise RuntimeError("Simulated SQLite disk I/O error during shutdown")

    monkeypatch.setattr("backend.storage.database.checkpoint_wal", broken_checkpoint)

    # Lifespan should yield and teardown without raising RuntimeError
    async with lifespan(app):
        pass


# ============================================================================
# 2. CONCURRENCY STRESS & HEADER FUZZING ON GET /health
# ============================================================================

def test_health_high_concurrency_stress():
    """Stress-test GET /health with 200 concurrent requests across a thread pool."""
    concurrency = 200
    with TestClient(app) as client:
        def do_get(i: int):
            t0 = time.perf_counter()
            r = client.get("/health")
            lat = time.perf_counter() - t0
            return r.status_code, r.json(), lat

        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(do_get, range(concurrency)))

    status_codes = [res[0] for res in results]
    latencies = [res[2] for res in results]

    assert all(code == 200 for code in status_codes), f"Non-200 responses: {set(status_codes)}"
    assert all(res[1] == {"status": "healthy"} for res in results)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    assert p95 < 0.1, f"P95 latency exceeded threshold: {p95:.4f}s"


@pytest.mark.parametrize(
    "origin,expected_allow",
    [
        ("http://localhost:5173", True),
        ("http://127.0.0.1:5173", True),
        ("http://localhost:3000", True),
        ("http://127.0.0.1:3000", True),
        ("http://localhost:8080", True),          # Dynamic localhost port matches regex
        ("http://127.0.0.1:9999", True),         # Dynamic 127.0.0.1 port matches regex
        ("null", True),                           # Electron null origin
        ("http://evil.com", False),               # External untrusted origin
        ("http://localhost.evil.com", False),     # Subdomain spoofing attempt
        ("http://127.0.0.1.attacker.org", False), # Subdomain spoofing attempt
        ("https://localhost:5173", True),         # HTTPS localhost
    ],
)
def test_health_cors_origin_adversarial_matrix(origin: str, expected_allow: bool):
    """Adversarially probe CORS headers on GET /health to ensure strict origin regex filtering."""
    with TestClient(app) as client:
        res = client.get("/health", headers={"Origin": origin})
        assert res.status_code == 200
        allow_header = res.headers.get("access-control-allow-origin")
        if expected_allow:
            assert allow_header == origin, f"Origin {origin} should be allowed but got {allow_header}"
        else:
            assert allow_header is None, f"Origin {origin} should NOT be allowed, but got {allow_header}"


def test_health_adversarial_headers_and_body():
    """Fuzz GET /health with oversized headers, valid ASCII symbols, and unexpected body."""
    with TestClient(app) as client:
        # 1. Oversized header (16KB)
        big_val = "A" * 16384
        res = client.get("/health", headers={"X-Oversized-Header": big_val})
        assert res.status_code == 200

        # 2. Complex symbols and punctuation in custom headers
        res = client.get(
            "/health",
            headers={
                "X-Symbols": "!@#$%^&*()_+{}[]|:;'<>?,./~`-=",
            },
        )
        assert res.status_code == 200

        # 3. GET /health with unexpected request body (should be safely ignored, not 500)
        res = client.request(
            "GET",
            "/health",
            data=b'{"unexpected": "payload", "garbage": 9999}',
            headers={"Content-Type": "application/json"},
        )
        assert res.status_code == 200
        assert res.json() == {"status": "healthy"}


def test_health_invalid_methods():
    """Verify that unsupported HTTP methods on /health return 405 Method Not Allowed."""
    with TestClient(app) as client:
        assert client.post("/health").status_code == 405
        assert client.put("/health").status_code == 405
        assert client.delete("/health").status_code == 405
        assert client.patch("/health").status_code == 405


# ============================================================================
# 3. SQLITE WAL CHECKPOINT BEHAVIOR UNDER ACTIVE LOAD & LOCK CONTENTION
# ============================================================================

def test_wal_truncation_under_heavy_writes():
    """Verify that a database with substantial write volume flushes and truncates

    its -wal file to 0 bytes upon checkpoint_wal().
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_wal.db"
        storage = DesktopStorage(test_db)
        storage.initialize()

        wal_file = Path(f"{test_db}-wal")
        assert test_db.exists()

        # Insert 1000 experiment records to force dirty pages into the WAL file
        for i in range(1000):
            storage.add_experiment({
                "id": f"exp-{i}",
                "name": f"Adversarial Experiment {i}",
                "data": [i * 1.5, f"payload-{i}"],
            })

        # WAL file should have grown
        assert wal_file.exists(), "WAL file was not created"
        wal_size_before = wal_file.stat().st_size
        assert wal_size_before > 0, "WAL size did not grow after 1000 inserts"

        # Execute WAL checkpoint
        busy, log_pages, checkpointed = storage.checkpoint_wal()
        assert busy == 0, f"Checkpoint reported busy={busy}"
        assert checkpointed >= 0

        # WAL file should now be truncated to 0 bytes
        wal_size_after = wal_file.stat().st_size
        assert wal_size_after == 0, f"WAL was not truncated to 0 bytes: actual {wal_size_after}"

        # Verify all 1000 records are durable in main DB file
        experiments = storage.list_experiments()
        assert len(experiments) == 1000
        assert experiments[0]["id"] == "exp-0"
        assert experiments[999]["id"] == "exp-999"

        # Explicit GC to release Python sqlite3 connection file handles on Windows
        del storage
        gc.collect()


def test_wal_checkpoint_under_lock_contention():
    """Verify behavior when checkpoint_wal is called while another connection holds

    an active uncommitted transaction. SQLite should return busy=1 without crashing.
    """
    import sqlite3

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "contention.db"
        storage = DesktopStorage(test_db)
        storage.initialize()

        storage.add_experiment({"id": "init-1", "val": 1})

        # Open raw connection and hold an exclusive/immediate write lock
        raw_conn = sqlite3.connect(test_db, timeout=0.1)
        raw_conn.execute("BEGIN IMMEDIATE")
        raw_conn.execute("INSERT INTO experiments VALUES (?, ?, ?)", ("lock-1", "{}", "2026-01-01"))

        try:
            # Checkpoint attempt while locked
            busy, log_pages, checkpointed = storage.checkpoint_wal()
            # SQLite returns busy > 0 when the checkpoint could not acquire all locks
            assert busy >= 1, f"Expected busy >= 1 under active lock, got busy={busy}"
        finally:
            raw_conn.commit()
            raw_conn.close()

        # Now that lock is released, checkpoint should cleanly succeed
        busy_after, _, _ = storage.checkpoint_wal()
        assert busy_after == 0, f"Expected busy=0 after lock release, got busy={busy_after}"

        wal_file = Path(f"{test_db}-wal")
        if wal_file.exists():
            assert wal_file.stat().st_size == 0

        del storage
        gc.collect()


def test_concurrent_multithreaded_storage_writes_and_checkpoints():
    """Verify that multiple concurrent threads writing to DesktopStorage while a separate

    thread performs periodic checkpoints completes with 0 errors and 0 data loss.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "concurrent_stress.db"
        storage = DesktopStorage(test_db)
        storage.initialize()

        num_threads = 5
        records_per_thread = 100
        errors: list[Exception] = []

        def writer(thread_id: int):
            try:
                for i in range(records_per_thread):
                    storage.add_experiment({
                        "id": f"t{thread_id}-rec{i}",
                        "thread": thread_id,
                        "seq": i,
                    })
            except Exception as e:
                errors.append(e)

        def checkpointer():
            try:
                for _ in range(10):
                    storage.checkpoint_wal()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_threads + 1) as pool:
            futures = [pool.submit(writer, t) for t in range(num_threads)]
            futures.append(pool.submit(checkpointer))
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Encountered concurrent storage errors: {errors}"
        all_records = storage.list_experiments()
        assert len(all_records) == num_threads * records_per_thread

        # Final checkpoint must truncate WAL
        busy, _, _ = storage.checkpoint_wal()
        assert busy == 0
        wal_file = Path(f"{test_db}-wal")
        if wal_file.exists():
            assert wal_file.stat().st_size == 0

        del storage
        gc.collect()


# ============================================================================
# 4. PROCESS SIGNAL HANDLING & TREE TERMINATION
# ============================================================================

def test_windows_process_tree_termination():
    """Empirically test Windows process tree termination (taskkill /T /F /PID)

    on a multi-level child process tree.
    """
    if sys.platform != "win32":
        pytest.skip("Windows-specific test")

    parent_code = (
        "import subprocess, sys, time\n"
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "with open('child_pid.tmp', 'w') as f:\n"
        "    f.write(str(grandchild.pid))\n"
        "time.sleep(60)\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path(tmpdir)
        pid_file = cwd / "child_pid.tmp"

        parent_proc = subprocess.Popen(
            [sys.executable, "-c", parent_code],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for grandchild PID to be written
        deadline = time.time() + 5.0
        grandchild_pid = None
        while time.time() < deadline:
            if pid_file.exists() and pid_file.stat().st_size > 0:
                grandchild_pid = int(pid_file.read_text().strip())
                break
            time.sleep(0.1)

        assert grandchild_pid is not None, "Failed to obtain grandchild PID"

        def is_pid_running(pid: int) -> bool:
            check = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
            )
            return str(pid) in check.stdout

        assert is_pid_running(parent_proc.pid), f"Parent {parent_proc.pid} not running"
        assert is_pid_running(grandchild_pid), f"Grandchild {grandchild_pid} not running"

        # Execute tree kill on parent PID
        kill_res = subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(parent_proc.pid)],
            capture_output=True,
            text=True,
        )
        assert kill_res.returncode == 0, f"taskkill failed: {kill_res.stderr}"

        time.sleep(0.5)

        # Both parent and grandchild MUST be dead
        assert not is_pid_running(parent_proc.pid), f"Parent PID {parent_proc.pid} survived tree kill"
        assert not is_pid_running(grandchild_pid), f"Grandchild PID {grandchild_pid} was orphaned!"


def test_backend_graceful_shutdown_on_process_signal():
    """Start an actual backend instance via subprocess on an ephemeral port (8019),

    send a graceful shutdown signal (CTRL_BREAK_EVENT), and verify lifespan teardown.
    """
    port = 8019
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT};{BACKEND_DIR}"
    env["PYTHONUNBUFFERED"] = "1"
    env["MECH_PRELOAD_MODELS"] = "0"

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        creationflags=creationflags,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        import urllib.request
        deadline = time.time() + 45.0
        ready = False
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.5)

        assert ready, "Backend failed to become healthy on ephemeral port within 30s"

        # Send graceful shutdown signal
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGTERM)

        stdout, _ = proc.communicate(timeout=10.0)

        assert "MECH Platform backend shutting down..." in stdout or "Shutting down" in stdout, (
            f"Expected shutdown logs in stdout, got:\n{stdout}"
        )
        assert "SQLite WAL checkpoint complete" in stdout, (
            f"Expected WAL checkpoint logs in stdout, got:\n{stdout}"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
