import json
import os
import subprocess
import sys
import time
from pathlib import Path

def main():
    if len(sys.argv) < 6:
        print("Usage: mock_tree_worker.py <worker_id> <depth> <max_depth> <children_per_node> <reg_dir>")
        sys.exit(1)

    worker_id = sys.argv[1]
    depth = int(sys.argv[2])
    max_depth = int(sys.argv[3])
    children_per_node = int(sys.argv[4])
    reg_dir = Path(sys.argv[5])

    my_pid = os.getpid()

    # Write own registration file (atomic, no file contention)
    reg_file = reg_dir / f"{worker_id}_{my_pid}.json"
    with open(reg_file, 'w') as f:
        json.dump({'pid': my_pid, 'depth': depth, 'id': worker_id}, f)

    # Spawn children if not at leaf
    children = []
    if depth < max_depth:
        for i in range(children_per_node):
            child_id = f"{worker_id}.{i+1}"
            cmd = [
                sys.executable,
                __file__,
                child_id,
                str(depth + 1),
                str(max_depth),
                str(children_per_node),
                str(reg_dir),
            ]
            p = subprocess.Popen(cmd)
            children.append(p)

    # Keep alive until terminated by taskkill or parent kill
    time.sleep(120)

if __name__ == "__main__":
    main()
