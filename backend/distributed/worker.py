"""Distributed Experiment Worker.

Executes standalone discovery algorithm experiments on assigned GPU/CPU workers.

Note: Workers execute experiments in isolation and emit DiscoveryReport artifacts 
to the Evidence Aggregator. Workers NEVER mutate global belief registries directly.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from ..interpretability.discovery.algorithms import get_algorithm
from ..interpretability.discovery.algorithms.base_algorithm import DiscoveryReport
from ..science.models.adapter_base import ModelAdapter
from ..science.models.gpt2_adapter import GPT2Adapter


@dataclass
class ExperimentTask:
    """Task specification dispatched to a distributed worker."""
    task_id: str
    campaign_id: str
    algorithm_name: str
    model_id: str
    dataset_shard: Dict[str, Any]
    config_override: Dict[str, Any] = field(default_factory=dict)
    assigned_worker_id: str = "gpu_worker_0"
    priority: int = 1


@dataclass
class WorkerExecutionResult:
    """Artifact returned by a distributed worker after task completion."""
    task_id: str
    worker_id: str
    status: str  # Success, Error
    discovery_report: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    compute_flops: float = 1.0e12
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")


class DistributedWorker:
    """Isolated execution worker running on an assigned cluster node."""

    def __init__(self, worker_id: str = "gpu_worker_0", adapter: Optional[ModelAdapter] = None) -> None:
        self.worker_id = worker_id
        self.adapter = adapter

    def _resolve_adapter(self, model_id: str, config_override: Optional[Dict[str, Any]] = None) -> ModelAdapter:
        """
        Dynamically resolves model adapter for the worker node.
        
        NEVER falls back to mock mode. If the model cannot be loaded,
        raises an error to prevent fabricated results from reaching
        EvidenceRecord or DiscoveryReport.
        """
        if self.adapter is not None:
            return self.adapter
        
        cfg = config_override or {}
        mock_mode = cfg.get("mock_mode", False)
        
        # CRITICAL: Never allow mock_mode in distributed execution
        if mock_mode:
            raise RuntimeError(
                f"Distributed worker rejecting mock_mode=True for model '{model_id}'. "
                "Mock data must never reach DiscoveryReport or EvidenceRecord."
            )
        
        from ..science.models.adapter_registry import ModelAdapterRegistry
        registry = ModelAdapterRegistry()
        try:
            adapter = registry.get_adapter(model_id, mock_mode=False)
            return adapter
        except (ImportError, ValueError, KeyError, OSError) as exc:
            # CRITICAL: Return explicit error instead of mock fallback
            raise RuntimeError(
                f"Cannot load model '{model_id}' on worker '{self.worker_id}'. "
                f"Model adapter resolution failed: {exc}. "
                "Refusing to fall back to mock mode - this would produce fabricated scientific results."
            ) from exc

    def execute_task(self, task: ExperimentTask) -> WorkerExecutionResult:
        """Executes experiment task in isolation on worker hardware."""
        t0 = time.time()
        try:
            adapter = self._resolve_adapter(task.model_id, task.config_override)
            alg_instance = get_algorithm(task.algorithm_name, adapter)
            report: DiscoveryReport = alg_instance.run(task.dataset_shard)

            execution_ms = (time.time() - t0) * 1000
            flops = 1.5e12 * max(0.01, execution_ms / 1000.0)

            return WorkerExecutionResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status="Success",
                discovery_report=report.to_dict(),
                execution_time_ms=round(execution_ms, 2),
                compute_flops=flops,
            )
        except Exception as e:
            execution_ms = (time.time() - t0) * 1000
            return WorkerExecutionResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status="Error",
                error_message=str(e),
                execution_time_ms=round(execution_ms, 2),
                compute_flops=0.5e12,
            )
