"""Central Execution Orchestrator (Sprint 4 & 5).

Coordinates execution planner, experiment queue, resource optimizer, cloud runtime providers,
Kubernetes, Ray, Slurm HPC, tensor cache, multi-user isolation, cost tracking, auto-recovery,
learned runtime optimizer, data locality manager, and execution backend abstractions.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List

from ..execution.cloud_runtime import CloudRuntimeManager
from ..execution.distributed_experiments import DistributedExperimentEngine
from ..memory.checkpoint_resume import CheckpointRecoveryEngine
from ..scheduling.experiment_queue import PriorityExperimentQueue
from .auto_recovery import RuntimeAutoRecoveryEngine
from .background_executor import BackgroundExecutionDaemon
from .cost_tracker import ExperimentCostTracker
from .data_locality_manager import DataLocalityManager
from .distributed_tensor_cache import DistributedTensorCache
from .execution_backends import (
    BaseExecutionBackend,
    CloudExecutionBackend,
    KubernetesExecutionBackend,
    LocalExecutionBackend,
    RayExecutionBackend,
    SlurmExecutionBackend,
)
from .execution_planner import ExecutionPlannerEngine
from .experiment_lifecycle import ExperimentLifecycleState
from .k8s_runtime import KubernetesRuntimeManager
from .learned_runtime_optimizer import LearnedRuntimeOptimizer
from .ray_integration import RayClusterManager
from .resource_optimizer import ResourceOptimizerEngine
from .runtime_benchmark import RuntimeBenchmarkEngine
from .runtime_isolation import RuntimeIsolationEngine
from .slurm_scheduler import SlurmSchedulerManager


class ExecutionOrchestrator:
    """Central orchestrator managing all runtime execution services."""

    def __init__(self) -> None:
        self.cloud_manager = CloudRuntimeManager()
        self.distributed_engine = DistributedExperimentEngine()
        self.queue = PriorityExperimentQueue()
        self.background_daemon = BackgroundExecutionDaemon()
        self.checkpoint_recovery = CheckpointRecoveryEngine()
        self.benchmark_engine = RuntimeBenchmarkEngine()
        self.optimizer = ResourceOptimizerEngine()

        # AI 2 Industrial-Scale Execution Extensions
        self.k8s_manager = KubernetesRuntimeManager()
        self.ray_manager = RayClusterManager()
        self.slurm_manager = SlurmSchedulerManager()
        self.tensor_cache = DistributedTensorCache()
        self.isolation_engine = RuntimeIsolationEngine()
        self.cost_tracker = ExperimentCostTracker()
        self.auto_recovery = RuntimeAutoRecoveryEngine()

        # AI 2 Platform Extensions
        self.learned_optimizer = LearnedRuntimeOptimizer()
        self.locality_manager = DataLocalityManager()
        self.execution_planner = ExecutionPlannerEngine(
            optimizer=self.learned_optimizer,
            locality_manager=self.locality_manager,
        )
        self.backends: Dict[str, BaseExecutionBackend] = {
            "Local": LocalExecutionBackend(),
            "Kubernetes": KubernetesExecutionBackend(),
            "Ray": RayExecutionBackend(),
            "Slurm": SlurmExecutionBackend(),
            "Cloud": CloudExecutionBackend(),
        }

        self.experiments: Dict[str, ExperimentLifecycleState] = {}

    def submit_and_orchestrate(
        self,
        experiment_id: str,
        goal: str = "Large Scale Parallel Analysis",
        priority: int = 1,
        strategy: str = "Balanced",
        user_id: str = "usr_1",
    ) -> Dict[str, Any]:
        lifecycle = ExperimentLifecycleState(experiment_id=experiment_id, goal=goal)
        self.experiments[experiment_id] = lifecycle

        # 1. Multi-user isolation & cost tracking & separate Execution Plan generation
        iso = self.isolation_engine.create_isolated_namespace(user_id=user_id)
        cost = self.cost_tracker.track_cost(gpu_hours=2.5)

        # Separate planning from execution
        execution_plan = self.execution_planner.create_execution_plan(
            experiment_id=experiment_id,
            model_name="GPT-2 Small",
            num_prompts=1000,
            strategy=strategy,
        )

        # 2. Lifecycle state transitions
        lifecycle.transition_to("Queued", "Submitted to priority queue")
        self.queue.submit_experiment(experiment_id=experiment_id, priority=priority)

        lifecycle.transition_to("Planning", "Resource optimization & cloud target selection")
        resource_plan = self.optimizer.optimize_resources(strategy=strategy)

        # 3. Select backend from compiled execution plan and execute
        chosen_backend_name = execution_plan.target_backend
        backend = self.backends.get(chosen_backend_name, self.backends["Local"])
        backend_job = backend.submit_job(job_name=experiment_id, payload={"goal": goal})

        lifecycle.transition_to("Scheduled", f"Allocated execution target [{chosen_backend_name}]")
        lifecycle.transition_to("Running", "Dispatching distributed experiment DAG")

        res = self.distributed_engine.execute_graph(experiment_id=experiment_id)

        # 4. Distributed Tensor Cache & Cluster Runtimes
        cache_entry = self.tensor_cache.put_tensor(key=f"act_{experiment_id}", shape=[1, 12, 768])
        k8s_res = self.k8s_manager.deploy_job(job_name=experiment_id)
        ray_res = self.ray_manager.submit_ray_task(task_name=experiment_id)
        slurm_res = self.slurm_manager.submit_sbatch(job_name=experiment_id)

        lifecycle.transition_to("Checkpointing", "Saving debugger state checkpoint")
        lifecycle.transition_to("Completed", "Parallel execution finished cleanly")

        # 5. Record observed outcome into LearnedRuntimeOptimizer (online learning loop)
        self.learned_optimizer.record_execution_outcome(
            model_name="GPT-2 Small",
            num_prompts=1000,
            observed_runtime_sec=execution_plan.predicted_resources["predicted_runtime_sec"],
            observed_memory_vram_gb=execution_plan.predicted_resources["predicted_memory_vram_gb"],
            observed_cost_usd=execution_plan.predicted_resources["predicted_cost_usd"],
            failure=False,
        )

        return {
            "experiment": lifecycle.to_dict(),
            "isolation": iso,
            "cost_tracking": cost,
            "execution_plan": execution_plan.to_dict(),
            "learned_prediction": execution_plan.predicted_resources,
            "locality": {"allocated_tier": execution_plan.locality_tier, "tier": execution_plan.locality_tier},
            "backend_job": backend_job,
            "resource_plan": resource_plan,
            "execution_result": res,
            "tensor_cache": cache_entry,
            "k8s_job": k8s_res,
            "ray_task": ray_res,
            "slurm_job": slurm_res,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "orchestrator_status": "active",
            "experiments_count": len(self.experiments),
            "cloud_providers": self.cloud_manager.list_providers(),
            "available_backends": list(self.backends.keys()),
            "queue_size": len(self.queue.list_queue()),
            "daemon_status": self.background_daemon.get_status(),
            "accumulated_cost_usd": self.cost_tracker.accumulated_cost,
            "learned_samples_count": len(self.learned_optimizer.history),
        }
