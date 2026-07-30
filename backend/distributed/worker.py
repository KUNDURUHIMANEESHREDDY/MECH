"""Distributed Experiment Worker.

Executes standalone discovery algorithm experiments on assigned GPU/CPU workers.

Note: Workers execute experiments in isolation and emit DiscoveryReport artifacts 
to the Evidence Aggregator. Workers NEVER mutate global belief registries directly.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    """Isolated execution worker running on a assigned cluster node."""

    def __init__(self, worker_id: str = "gpu_worker_0", adapter: Optional[ModelAdapter] = None) -> None:
        self.worker_id = worker_id
        self.adapter = adapter or GPT2Adapter(variant="small", mock_mode=True)

    def execute_task(self, task: ExperimentTask) -> WorkerExecutionResult:
        """Executes experiment task in isolation on worker hardware."""
        t0 = time.time()
        try:
            alg_instance = get_algorithm(task.algorithm_name, self.adapter)
            report: DiscoveryReport = alg_instance.run(task.dataset_shard)

            execution_ms = (time.time() - t0) * 1000
            flops = 1.5e12 * (execution_ms / 1000.0)

            return WorkerExecutionResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status="Success",
                discovery_report=report.to_dict(),
                execution_time_ms=round(execution_ms, 2),
                compute_flops=flops
            )
        except Exception as e:
            execution_ms = (time.time() - t0) * 1000
            return WorkerExecutionResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status="Error",
                error_message=str(e),
                execution_time_ms=round(execution_ms, 2),
                compute_flops=0.5e12
            )
