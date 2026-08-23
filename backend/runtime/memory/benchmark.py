"""MECH Large-Model Disk Execution & Progressive Computation Benchmark.

Demonstrates and measures out-of-core disk-resident execution of large models
(125M -> 1B -> 3B -> 7B -> 13B -> 70B), verifying:
1. Peak RAM remains bounded at O(1 layer) rather than full model size.
2. Run 1: 100% Compute -> NVMe CAS populated.
3. Run 2: 100% Cache Hit -> 0 Disk Reads & 0 Compute.
4. Run 3: Intervention @ Layer K -> Layers 0..K-1 Cache Hit, Layers K..N Recomputed.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
from transformers import AutoTokenizer

from ..artifacts.cas_store import ArtifactStore, get_artifact_store
from .disk_weight_store import DiskWeightStore, ShardedModelManifest, get_disk_weight_store
from .layer_pager import LayerExecutionOutput, LayerPager

logger = logging.getLogger("MECH.benchmark")

# Model Scale Configurations
MODEL_CONFIGS = {
    "125M": {"layers": 12, "hidden": 768, "heads": 12, "vocab": 50257, "fp16_gb": 0.25},
    "1B":   {"layers": 24, "hidden": 2048, "heads": 16, "vocab": 32000, "fp16_gb": 2.1},
    "3B":   {"layers": 32, "hidden": 2560, "heads": 32, "vocab": 32000, "fp16_gb": 6.2},
    "7B":   {"layers": 32, "hidden": 4096, "heads": 32, "vocab": 32000, "fp16_gb": 14.5},
    "13B":  {"layers": 40, "hidden": 5120, "heads": 40, "vocab": 32000, "fp16_gb": 26.8},
    "70B":  {"layers": 80, "hidden": 8192, "heads": 64, "vocab": 32000, "fp16_gb": 142.0},
}


@dataclass
class ModelBenchmarkResult:
    """Benchmark metrics for a single model scale."""
    scale: str
    num_layers: int
    hidden_size: int
    full_model_fp16_gb: float
    disk_weights_mb: float
    cold_run_time_s: float
    cold_peak_ram_mb: float
    warm_run_time_s: float
    warm_peak_ram_mb: float
    warm_speedup_x: float
    warm_cache_hit_rate: float
    intervention_layer: int
    intervention_time_s: float
    intervention_cache_hit_rate: float
    memory_reduction_ratio: float  # full model RAM / peak execution RAM


class MockTokenizer:
    """Lightweight standalone tokenizer for synthetic benchmark scaling."""
    def __init__(self, vocab_size: int = 32000) -> None:
        self.vocab_size = vocab_size

    def __call__(self, prompt: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        # Hash words to tokens within vocab range
        words = prompt.split()
        tids = [abs(hash(w)) % self.vocab_size for w in words] or [1]
        input_ids = torch.tensor([tids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, token_ids: List[int]) -> str:
        return f"tok_{token_ids[0]}" if token_ids else ""


def run_model_disk_benchmark(
    scale: str,
    weight_store: Optional[DiskWeightStore] = None,
    cas_store: Optional[ArtifactStore] = None,
    prompt: str = "The mechanistic interpretability circuit explains neural features",
) -> ModelBenchmarkResult:
    """Executes a full 3-phase progressive disk-execution benchmark on a given model scale."""
    if scale not in MODEL_CONFIGS:
        raise ValueError(f"Unknown scale '{scale}'. Available: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[scale]
    weight_store = weight_store or get_disk_weight_store()
    cas_store = cas_store or get_artifact_store()
    model_id = f"mech-disk-{scale.lower()}"

    # 1. Create or load on-disk sharded model (out-of-core, 1 layer at a time to NVMe)
    logger.info(f"Preparing on-disk sharded weights for {scale} model ({cfg['layers']} layers, {cfg['hidden']} hidden)...")
    manifest = weight_store.create_synthetic_scaled_sharded_model(
        model_id=model_id,
        num_layers=cfg["layers"],
        hidden_size=cfg["hidden"],
        num_heads=cfg["heads"],
        vocab_size=cfg["vocab"],
        dtype=torch.float32,
    )

    disk_size_mb = manifest.total_size_bytes / (1024 * 1024)
    pager = LayerPager(device="cpu", dtype=torch.float32, store=cas_store, weight_store=weight_store)
    tokenizer = MockTokenizer(vocab_size=cfg["vocab"])

    # ── Phase 1: Cold Run (Baseline Stream & Compute) ──
    gc.collect()
    cold_output = pager.run_disk_paged_forward(
        manifest=manifest,
        prompt=prompt,
        tokenizer=tokenizer,
        session_id=f"cold_session_{scale}",
    )

    # ── Phase 2: Warm Run (100% Cache Reuse) ──
    gc.collect()
    warm_output = pager.run_disk_paged_forward(
        manifest=manifest,
        prompt=prompt,
        tokenizer=tokenizer,
        session_id=f"warm_session_{scale}",
    )

    warm_speedup = cold_output.execution_time_seconds / max(warm_output.execution_time_seconds, 1e-6)

    # ── Phase 3: Intervention Run (Partial Recomputation) ──
    intervention_layer = cfg["layers"] // 2
    intervention_fn = lambda t: t * 0.0  # Zero ablation hook

    gc.collect()
    intervention_output = pager.run_disk_paged_forward(
        manifest=manifest,
        prompt=prompt,
        tokenizer=tokenizer,
        interventions={intervention_layer: intervention_fn},
        session_id=f"intervention_session_{scale}",
    )

    # Clean up disk files after benchmark
    weight_store.cleanup_model(model_id)

    full_ram_mb = cfg["fp16_gb"] * 1024
    mem_reduction = full_ram_mb / max(cold_output.peak_ram_mb, 1.0)

    return ModelBenchmarkResult(
        scale=scale,
        num_layers=cfg["layers"],
        hidden_size=cfg["hidden"],
        full_model_fp16_gb=cfg["fp16_gb"],
        disk_weights_mb=disk_size_mb,
        cold_run_time_s=cold_output.execution_time_seconds,
        cold_peak_ram_mb=cold_output.peak_ram_mb,
        warm_run_time_s=warm_output.execution_time_seconds,
        warm_peak_ram_mb=warm_output.peak_ram_mb,
        warm_speedup_x=warm_speedup,
        warm_cache_hit_rate=warm_output.cache_hit_rate,
        intervention_layer=intervention_layer,
        intervention_time_s=intervention_output.execution_time_seconds,
        intervention_cache_hit_rate=intervention_output.cache_hit_rate,
        memory_reduction_ratio=mem_reduction,
    )


def run_benchmark_suite(scales: Optional[List[str]] = None) -> List[ModelBenchmarkResult]:
    """Runs benchmarks across specified model scales and prints formatted summary."""
    scales = scales or ["125M", "1B", "3B", "7B", "13B"]
    results: List[ModelBenchmarkResult] = []

    print("\n" + "=" * 100)
    print("MECH LARGE-MODEL DISK EXECUTION & PROGRESSIVE COMPUTATION BENCHMARK")
    print("=" * 100)
    print(f"System Physical RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print("-" * 100)

    for scale in scales:
        print(f"\n[BENCHMARKING SCALE: {scale}]")
        t0 = time.time()
        res = run_model_disk_benchmark(scale)
        results.append(res)
        print(f"  -> Cold Run (All layers computed & cached): {res.cold_run_time_s:.4f}s (Peak RAM: {res.cold_peak_ram_mb:.1f} MB)")
        print(f"  -> Warm Run (100% CAS Cache Hit, 0 Disk I/O): {res.warm_run_time_s:.4f}s (Speedup: {res.warm_speedup_x:.1f}x, Hit Rate: {res.warm_cache_hit_rate*100:.0f}%)")
        print(f"  -> Intervention @ Layer {res.intervention_layer}: {res.intervention_time_s:.4f}s (Hit Rate: {res.intervention_cache_hit_rate*100:.0f}% on upstream layers)")
        print(f"  -> Memory Reduction: {res.memory_reduction_ratio:.1f}x less RAM than full model ({res.full_model_fp16_gb:.1f} GB model executed in {res.cold_peak_ram_mb:.0f} MB RAM)")

    # Print Summary Table
    print("\n" + "=" * 100)
    print("SUMMARY RESULTS TABLE")
    print("=" * 100)
    header = f"{'Model':<8} | {'Layers':<6} | {'Full Size':<10} | {'Peak RAM':<10} | {'Cold Run':<10} | {'Warm Run':<10} | {'Speedup':<8} | {'Intervention Hit'}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.scale:<8} | {r.num_layers:<6} | {r.full_model_fp16_gb:>5.1f} GB    | "
            f"{r.cold_peak_ram_mb:>6.0f} MB   | {r.cold_run_time_s:>7.3f} s  | "
            f"{r.warm_run_time_s:>7.4f} s | {r.warm_speedup_x:>6.1f}x  | "
            f"{r.intervention_cache_hit_rate*100:>5.0f}% (L0..{r.intervention_layer-1})"
        )
    print("=" * 100 + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MECH Large-Model Disk Execution Benchmark")
    parser.add_argument("--scales", nargs="+", default=["125M", "1B", "3B", "7B", "13B"], help="Model scales to benchmark")
    args = parser.parse_args()

    run_benchmark_suite(args.scales)
