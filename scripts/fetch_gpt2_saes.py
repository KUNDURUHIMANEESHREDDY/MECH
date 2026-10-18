"""Fetch and cache the real GPT-2 small SAE checkpoints (layers 8 and 10).

Source: jbloom/GPT2-Small-SAEs -- public, 24576-feature residual-stream SAEs at
`blocks.{layer}.hook_resid_pre`. Verified before downloading that the file is a
plain tensor state dict (pickle references only `_rebuild_tensor_v2`,
`FloatStorage`, `OrderedDict`), so the repository's existing
`torch.load(..., weights_only=True)` policy applies without a permissive load.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.interpretability.sae import registry


def fetch_all(dest_dir: str | None = None) -> list[str]:
    paths = []
    print(
        f"Fetching GPT-2 small SAE checkpoints from {registry.REPO_ID} "
        f"(revision: {registry.REVISION[:8]})...",
        flush=True,
    )
    for layer in registry.available_layers():
        entry = registry.resolve(layer)
        if entry is None:
            continue
        t0 = time.time()
        path = registry.fetch(layer)
        size_mb = os.path.getsize(path) / 1e6
        elapsed = time.time() - t0
        print(f"Layer {layer:2d}: {size_mb:.1f} MB in {elapsed:.1f}s -> {path}", flush=True)

        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
            target = os.path.join(dest_dir, entry.filename)
            if not os.path.exists(target) or os.path.getsize(target) != os.path.getsize(path):
                shutil.copy2(path, target)
                print(f"  Copied to: {target}", flush=True)
            paths.append(target)
        else:
            paths.append(path)

    print("All SAE checkpoints cached successfully.", flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Fetch/cache GPT-2 Small SAE checkpoints.")
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Optional local directory to copy the cached checkpoints into.",
    )
    args = parser.parse_args()
    fetch_all(dest_dir=args.dest)


if __name__ == "__main__":
    main()
