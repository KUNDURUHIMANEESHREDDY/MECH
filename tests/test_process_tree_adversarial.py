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
import psutil

TESTS_DIR = Path(__file__).parent
WORKER_SCRIPT = TESTS_DIR / "mock_tree_worker.py"
REGISTRY_DIR = TESTS_DIR / "tree_registry_dir"


def cleanup_pids(pids):
    for pid in pids:
        try:
            if psutil.pid_exists(pid):
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        except Exception:
            pass


def run_tree_test(use_taskkill: bool):
    if REGISTRY_DIR.exists():
        shutil.rmtree(REGISTRY_DIR, ignore_errors=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    # Root process will spawn 2 children, and each child will spawn 2 grandchildren = 1 + 2 + 4 = 7 processes!
    root_proc = subprocess.Popen([
        sys.executable,
        str(WORKER_SCRIPT),
        "R",          # worker_id
        "1",          # depth
        "3",          # max_depth
        "2",          # children_per_node
        str(REGISTRY_DIR),
    ])

    print(f"Spawned Root process: PID {root_proc.pid}")

    # Wait for all 7 processes to register
    registered = []
    for _ in range(50):
        time.sleep(0.3)
        files = list(REGISTRY_DIR.glob("*.json"))
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

    print(f"Registered {len(registered)} processes in tree:")
    for item in sorted(registered, key=lambda x: x['id']):
        alive = psutil.pid_exists(item['pid'])
        print(f"  - Node {item['id']} (Depth {item['depth']}): PID {item['pid']}, alive={alive}")

    all_pids = [item['pid'] for item in registered]

    if len(all_pids) < 7:
        cleanup_pids(all_pids)
        raise RuntimeError(f"Expected 7 processes in tree, got {len(all_pids)}")

    # Ensure all are running
    assert all(psutil.pid_exists(p) for p in all_pids), "Not all spawned processes are alive!"

    if use_taskkill:
        print(f"\n[TERMINATING WITH TASKKILL /T /F on Root PID {root_proc.pid}]")
        res = subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(root_proc.pid)],
            capture_output=True,
            text=True,
        )
        print("taskkill stdout:", res.stdout.strip())
        print("taskkill returncode:", res.returncode)
    else:
        print(f"\n[TERMINATING WITH NAIVE root_proc.kill() on Root PID {root_proc.pid}]")
        root_proc.kill()

    time.sleep(1.5)

    # Check survival status
    survivors = [p for p in all_pids if psutil.pid_exists(p)]
    print(f"\nPost-termination check: {len(survivors)} / {len(all_pids)} processes still alive")
    for item in sorted(registered, key=lambda x: x['id']):
        alive = psutil.pid_exists(item['pid'])
        print(f"  - Node {item['id']} (PID {item['pid']}): {'ALIVE (ORPHAN)' if alive else 'DEAD (CLEAN)'}")

    # Clean up survivors
    cleanup_pids(survivors)
    if REGISTRY_DIR.exists():
        shutil.rmtree(REGISTRY_DIR, ignore_errors=True)

    return len(survivors)


def main():
    print("=" * 60)
    print("EMPIRICAL COMPARISON: NAIVE KILL VS WINDOWS TASKKILL /T /F")
    print("=" * 60)

    # Part A: Demonstrate naive kill failure
    print("\n>>> EXPERIMENT A: Naive Process Kill (Standard TerminateProcess) <<<")
    naive_survivors = run_tree_test(use_taskkill=False)
    print(f"Result with naive kill: {naive_survivors} orphan processes survived!")
    assert naive_survivors > 0, "Expected naive kill to leave orphans on Windows!"
    print(f"[CONFIRMED]: Naive kill failed as predicted ({naive_survivors} child processes were orphaned).")

    time.sleep(1)

    # Part B: Demonstrate taskkill /T /F success
    print("\n>>> EXPERIMENT B: Tree Termination (taskkill /T /F /PID) <<<")
    taskkill_survivors = run_tree_test(use_taskkill=True)
    print(f"Result with taskkill /T /F: {taskkill_survivors} orphan processes survived.")
    assert taskkill_survivors == 0, f"Expected 0 survivors with taskkill /T /F, got {taskkill_survivors}!"
    print("[CONFIRMED]: taskkill /T /F successfully terminated the entire 7-process tree with zero orphans!")

    print("\n" + "=" * 60)
    print("EMPIRICAL VERIFICATION COMPLETE: ALL HYPOTHESES CONFIRMED")
    print("=" * 60)


if __name__ == "__main__":
    main()
