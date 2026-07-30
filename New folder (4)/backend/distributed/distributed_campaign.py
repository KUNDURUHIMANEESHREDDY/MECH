"""Distributed Campaign Orchestrator.

Orchestrates multi-worker distributed campaign execution across cluster workers,
collecting worker output reports and feeding them into Evidence Aggregator.

Workflow:
Research Campaign ➔ Task Sharding ➔ Distributed Scheduler ➔ Parallel Workers ➔ Evidence Aggregator ➔ Bayesian Belief Update
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .resource_manager import ResourceManager
from .scheduler import DistributedScheduler
from .worker import ExperimentTask, WorkerExecutionResult
from .evidence_aggregator import EvidenceAggregator, AggregatedEvidenceResult
from ..interpretability.discovery.discovery_planner import ResearchGoal


@dataclass
class DistributedCampaignSummary:
    """Artifact emitted by Distributed Campaign Orchestrator."""
    campaign_id: str
    goal_description: str
    total_tasks_scheduled: int
    successful_tasks: int
    failed_tasks: int
    cluster_workers_used: List[str]
    aggregated_evidence: Dict[str, Any]
    total_runtime_ms: float
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "goal_description": self.goal_description,
            "total_tasks_scheduled": self.total_tasks_scheduled,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "cluster_workers_used": self.cluster_workers_used,
            "aggregated_evidence": self.aggregated_evidence,
            "total_runtime_ms": self.total_runtime_ms,
            "timestamp": self.timestamp,
        }


class DistributedCampaignOrchestrator:
    """Orchestrates distributed campaign execution across cluster workers."""

    def __init__(
        self,
        resource_manager: Optional[ResourceManager] = None,
        scheduler: Optional[DistributedScheduler] = None,
        aggregator: Optional[EvidenceAggregator] = None
    ) -> None:
        self.resource_manager = resource_manager or ResourceManager()
        self.scheduler = scheduler or DistributedScheduler(resource_manager=self.resource_manager)
        self.aggregator = aggregator or EvidenceAggregator()

    def run_distributed_campaign(self, goal: ResearchGoal) -> DistributedCampaignSummary:
        """Executes a multi-task campaign across distributed hardware workers."""
        t0 = time.time()
        campaign_id = f"dist_camp_{hash(goal.goal_id + str(time.time())) & 0xffffffff:08x}"

        planned_algorithms = ["attribution_patching", "acdc", "causal_scrubbing", "transcoders", "feature_universality"]

        # 1. Enqueue Sharded Tasks into Scheduler
        for idx, alg in enumerate(planned_algorithms):
            task = ExperimentTask(
                task_id=f"task_{campaign_id}_{idx + 1:02d}_{alg}",
                campaign_id=campaign_id,
                algorithm_name=alg,
                model_id=goal.model_id,
                dataset_shard={"clean": "John gave a drink to Mary", "corrupted": "John gave a drink to John", "target_token": " Mary"},
                priority=len(planned_algorithms) - idx
            )
            self.scheduler.queue_task(task)

        # 2. Dispatch Tasks to Workers
        results: List[WorkerExecutionResult] = []
        while self.scheduler.task_queue:
            res = self.scheduler.dispatch_next()
            if res:
                results.append(res)

        # 3. Aggregate Worker Results via Single-Threaded Evidence Aggregator
        agg_res: AggregatedEvidenceResult = self.aggregator.aggregate_worker_results(
            campaign_id=campaign_id,
            results=results,
            claim_title=f"Distributed Claim: {goal.description}"
        )

        total_runtime = (time.time() - t0) * 1000
        workers_used = list(set(r.worker_id for r in results))

        return DistributedCampaignSummary(
            campaign_id=campaign_id,
            goal_description=goal.description,
            total_tasks_scheduled=len(results),
            successful_tasks=agg_res.successful_results,
            failed_tasks=agg_res.failed_results,
            cluster_workers_used=workers_used,
            aggregated_evidence=agg_res.__dict__,
            total_runtime_ms=round(total_runtime, 2)
        )
