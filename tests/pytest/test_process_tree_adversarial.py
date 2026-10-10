"""Adversarial stress-test comparing Windows naive process termination vs taskkill /T /F.

Proves:
1. Naive process.kill() on Windows orphans child and grandchild processes.
2. taskkill /T /F /PID terminates the entire process tree cleanly with zero orphans.
3. Tests a multi-branch tree (Root with 2 children, each with 2 grandchildren = 7 processes).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import psutil


TESTS_DIR = Path(__file__).parent.parent
WORKER_SCRIPT = TESTS_DIR / "mock_tree_worker.py"


@pytest.fixture
def registry_dir(tmp_path):
    """Provide an isolated registry directory for each test."""
    reg_dir = tmp_path / "tree_registry_dir"
    reg_dir.mkdir(parents=True, exist_ok=True)
    yield reg_dir
    # Cleanup
    if reg_dir.exists():
        shutil.rmtree(reg_dir, ignore_errors=True)


def cleanup_pids(pids):
    for pid in pids:
        try:
            if psutil.pid_exists(pid):
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True,
                    check=False,
                )
        except Exception:
            pass


def run_tree_test(registry_dir: Path, use_taskkill: bool):
    """Run a full process tree test and return number of survivors."""
    if registry_dir.exists():
        shutil.rmtree(registry_dir, ignore_errors=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    # Root process will spawn 2 children, and each child will spawn 2 grandchildren = 1 + 2 + 4 = 7 processes!
    root_proc = subprocess.Popen([
        sys.executable,
        str(WORKER_SCRIPT),
        "R",          # worker_id
        "1",          # depth
        "3",          # max_depth
        "2",          # children_per_node
        str(registry_dir),
    ])

    # Wait for all 7 processes to register
    registered = []
    for _ in range(50):
        time.sleep(0.3)
        files = list(registry_dir.glob("*.json"))
        if len(files) >= 7:
            for f in files:
                try:
                    with open(f, "r") as fp:
                        registered.append(json.load(fp))
                except Exception:
                    pass
            if len(registered) >= 7:
                break
            registered = []

    assert len(registered) >= 7, f"Expected 7 processes in tree, got {len(registered)}"

    all_pids = [item['pid'] for item in registered]
    assert all(psutil.pid_exists(p) for p in all_pids), "Not all spawned processes are alive!"

    if use_taskkill:
        res = subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(root_proc.pid)],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"taskkill failed: {res.stderr}"
    else:
        root_proc.kill()

    time.sleep(1.5)

    # Check survival status
    survivors = [p for p in all_pids if psutil.pid_exists(p)]

    # Clean up survivors
    cleanup_pids(survivors)
    if registry_dir.exists():
        shutil.rmtree(registry_dir, ignore_errors=True)

    return len(survivors)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process tree test")
class TestProcessTreeAdversarial:
    """Empirical comparison: naive kill vs Windows taskkill /T /F"""

    def test_naive_kill_leaves_orphans(self, registry_dir):
        """Naive process.kill() on Windows must leave orphan processes."""
        survivors = run_tree_test(registry_dir, use_taskkill=False)
        assert survivors > 0, "Expected naive kill to leave orphans on Windows!"

    def test_taskkill_terminates_entire_tree(self, registry_dir):
        """taskkill /T /F must terminate the entire 7-process tree with zero orphans."""
        survivors = run_tree_test(registry_dir, use_taskkill=True)
        assert survivors == 0, f"Expected 0 survivors with taskkill /T /F, got {survivors}!"


# ---- mock_tree_worker.py logic as a reusable fixture ----


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only process test")
def test_mock_tree_node_isolation(registry_dir):
    """Test that the mock tree worker correctly registers processes."""
    import json

    # Use the actual worker script
    worker_file = WORKER_SCRIPT

    # Spawn root
    root_proc = subprocess.Popen([
        sys.executable,
        str(worker_file),
        "R", "1", "3", "2", str(registry_dir)
    ])

    # Wait for registration
    registered = []
    for _ in range(30):
        time.sleep(0.3)
        files = list(registry_dir.glob("*.json"))
        if len(files) >= 7:
            for f in files:
                try:
                    with open(f, "r") as fp:
                        registered.append(json.load(fp))
                except Exception:
                    pass
            if len(registered) >= 7:
                break
            registered = []

    assert len(registered) >= 7

    # All should be alive
    all_pids = [item['pid'] for item in registered]
    assert all(psutil.pid_exists(p) for p in all_pids)

    # Clean up
    root_proc.kill()
    time.sleep(0.5)
    cleanup_pids(all_pids)