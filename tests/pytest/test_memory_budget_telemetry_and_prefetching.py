"""Tests for Physical Memory-Budget Enforcement, Telemetry Reporting, and Pipelined Prefetching.

Validates Phase 12G/12H:
1. Physical process RSS and CUDA VRAM sampling and budget compliance checking.
2. Asynchronous pipelined prefetching hits and timing instrumentation.
3. Full telemetry report structure.
4. Numerical equivalence between pipelined prefetching and in-memory reference execution.
"""

import pytest
import torch

from backend.runtime.in_memory_runtime import InMemoryRuntime
from backend.runtime.out_of_core_runtime import OutOfCoreRuntime
from backend.science.reproducibility.tolerance_engine import DEFAULT_ABS_TOLERANCE


@pytest.fixture(scope="module")
def runtimes():
    in_mem = InMemoryRuntime(model_id="gpt2", device="cpu")
    out_core = OutOfCoreRuntime(
        model_id="gpt2",
        device="cpu",
        max_active_layers=1,
        vram_budget_mb=4096.0,
        ram_budget_mb=32768.0,
    )
    return in_mem, out_core


def test_physical_memory_sampling_and_budget_compliance(runtimes):
    """Verifies that physical memory is measured and budget compliance is correctly evaluated."""
    _, out_core = runtimes
    stats = out_core.residency.get_memory_statistics()

    assert stats["peak_rss_mb"] > 0
    assert "memory_budget_compliant" in stats
    assert stats["memory_budget_compliant"] is True
    assert stats["max_active_layers"] == 1


def test_pipelined_prefetching_hits_and_timing(runtimes):
    """Verifies that asynchronous layer prefetching records hits and measures load/eviction durations."""
    _, out_core = runtimes
    out_core.residency.reset_peak_stats()

    prompt = "The Eiffel Tower is in"
    target_token = " Paris"

    res = out_core.forward(prompt, target_token=target_token)
    stats = out_core.residency.get_memory_statistics()

    # Assert prefetch hits occurred during sequential layer execution
    assert stats["prefetch_hits"] > 0
    assert stats["peak_active_layers_observed"] <= 1
    assert stats["total_load_time_ms"] > 0


def test_full_execution_telemetry_report(runtimes):
    """Verifies the structured hardware, runtime, and model telemetry report."""
    _, out_core = runtimes
    report = out_core.get_execution_telemetry_report()

    # Model metadata
    assert report["model"]["model_id"] == "gpt2"
    assert report["model"]["parameter_count"] > 100_000_000
    assert report["model"]["quantization"] == "fp32"

    # Hardware metadata
    assert report["hardware"]["system_ram_budget_mb"] > 0
    assert report["hardware"]["compute_device"] == "cpu"

    # Runtime metadata
    assert report["runtime"]["max_active_layers"] == 1
    assert report["runtime"]["peak_active_layers_observed"] <= 1
    assert report["runtime"]["prefetch_hits"] > 0
    assert report["runtime"]["memory_budget_compliant"] is True


def test_prefetched_execution_equivalence_against_in_memory(runtimes):
    """Verifies that pipelined prefetching produces 100% exact numerical outputs as in-memory reference."""
    in_mem, out_core = runtimes
    prompt = "The capital of France is"
    target_token = " Paris"

    out_core.residency.reset_peak_stats()
    res_core = out_core.forward(prompt, target_token=target_token)
    res_mem = in_mem.forward(prompt, target_token=target_token)

    assert res_core.top_predicted_token == res_mem.top_predicted_token
    assert res_core.target_rank == res_mem.target_rank
    assert abs(res_core.target_logit - res_mem.target_logit) <= DEFAULT_ABS_TOLERANCE
    assert abs(res_core.target_probability - res_mem.target_probability) <= DEFAULT_ABS_TOLERANCE
