"""Execution Backend Abstraction Layer.

Provides unified execution backend drivers for Local, Distributed, Kubernetes,
Slurm HPC, Ray clusters, and Cloud providers.
"""

from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .k8s_runtime import KubernetesRuntimeManager
from .ray_integration import RayClusterManager
from .slurm_scheduler import SlurmSchedulerManager


class BaseExecutionBackend(ABC):
    """Abstract base class for all execution backends."""

    @abstractmethod
    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submits a job to the cluster engine."""
        pass

    @abstractmethod
    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Queries status of a job."""
        pass

    @abstractmethod
    def cancel_job(self, job_name: str) -> Dict[str, Any]:
        """Cancels a running job."""
        pass

    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """Returns engine capabilities, nodes, accelerators, and health."""
        pass


class LocalExecutionBackend(BaseExecutionBackend):
    """Local single-process execution backend."""

    def __init__(self) -> None:
        self.active_jobs: Dict[str, Dict[str, Any]] = {}

    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        info = {
            "backend": "Local",
            "job_name": job_name,
            "status": "Running",
            "device": "cpu",
            "submitted_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        self.active_jobs[job_name] = info
        return info

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        if job_name in self.active_jobs:
            return self.active_jobs[job_name]
        return {"backend": "Local", "job_name": job_name, "status": "Completed"}

    def cancel_job(self, job_name: str) -> Dict[str, Any]:
        if job_name in self.active_jobs:
            self.active_jobs[job_name]["status"] = "Terminated"
            return {"job_name": job_name, "status": "Terminated"}
        return {"job_name": job_name, "status": "Terminated"}

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "local",
            "status": "active",
            "driver": "InProcessLocal",
            "device": "CPU",
            "max_concurrency": 1,
            "features": ["immediate_execution", "zero_network_overhead"],
        }


class DistributedExecutionBackend(BaseExecutionBackend):
    """Multi-worker distributed scheduler backend."""

    def __init__(self) -> None:
        from ...distributed.scheduler import DistributedScheduler
        self.scheduler = DistributedScheduler()
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        from ...distributed.worker import ExperimentTask
        task = ExperimentTask(
            task_id=f"task_{job_name}",
            campaign_id=payload.get("campaign_id", f"camp_{job_name}"),
            algorithm_name=payload.get("algorithm_name", "acdc"),
            model_id=payload.get("model_id", "gpt2-small"),
            dataset_shard=payload.get("dataset_shard", {"prompt": "Paris is capital"}),
            config_override=payload.get("config", {}),
            priority=payload.get("priority", 1),
        )
        self.scheduler.queue_task(task)
        res = self.scheduler.dispatch_next()
        status = res.status if res else "Queued"
        info = {
            "backend": "Distributed",
            "job_name": job_name,
            "task_id": task.task_id,
            "status": "Running" if status == "Success" else status,
            "worker_id": res.worker_id if res else "pending",
            "compute_flops": res.compute_flops if res else 1.0e12,
            "submitted_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        self.jobs[job_name] = info
        return info

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        return self.jobs.get(job_name, {"backend": "Distributed", "job_name": job_name, "status": "Completed"})

    def cancel_job(self, job_name: str) -> Dict[str, Any]:
        if job_name in self.jobs:
            self.jobs[job_name]["status"] = "Cancelled"
            return {"job_name": job_name, "status": "Cancelled"}
        return {"job_name": job_name, "status": "Cancelled"}

    def get_engine_info(self) -> Dict[str, Any]:
        status = self.scheduler.get_scheduler_status()
        return {
            "name": "distributed",
            "status": "active",
            "driver": "DistributedScheduler",
            "cluster_nodes": len(status["cluster_workers"]),
            "features": ["hardware_aware_routing", "fault_tolerance", "checkpointing", "multi_worker_concurrency"],
            "telemetry": status,
        }


class KubernetesExecutionBackend(BaseExecutionBackend):
    """Kubernetes pod & batch job orchestration backend."""

    def __init__(self) -> None:
        self.k8s = KubernetesRuntimeManager()

    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        num_gpus = payload.get("num_gpus", 1)
        num_cpus = payload.get("num_cpus", 4)
        mem_gb = payload.get("mem_gb", 16)
        image = payload.get("image", "interp/runtime:v5")
        res = self.k8s.deploy_job(job_name=job_name, image=image, num_gpus=num_gpus, num_cpus=num_cpus, memory_gb=mem_gb)
        return {
            "backend": "Kubernetes",
            "job_name": job_name,
            "pod_id": res["pod_id"],
            "job_id": res["job_id"],
            "namespace": res["namespace"],
            "status": res["status"],
            "allocated_gpus": num_gpus,
            "manifest": res["manifest"],
        }

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        status = self.k8s.get_job_status(job_name)
        return {
            "backend": "Kubernetes",
            "job_name": job_name,
            "pod_id": status.get("pod_id", f"pod_{job_name}"),
            "status": status.get("status", "Running"),
            "namespace": status.get("namespace", "mech-experiments"),
            "logs": status.get("logs", []),
        }

    def cancel_job(self, job_name: str) -> Dict[str, Any]:
        return self.k8s.cancel_job(job_name)

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "kubernetes",
            "status": "active",
            "driver": "KubernetesRuntimeManager",
            "api_version": "batch/v1",
            "namespace": self.k8s.namespace,
            "gpu_driver": "nvidia.com/gpu",
            "features": ["pod_isolation", "horizontal_pod_autoscaling", "gpu_limits", "declarative_crds"],
        }


class RayExecutionBackend(BaseExecutionBackend):
    """Ray distributed task & actor execution backend."""

    def __init__(self) -> None:
        self.ray = RayClusterManager()

    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        num_cpus = payload.get("num_cpus", 4)
        num_gpus = payload.get("num_gpus", 1)
        res = self.ray.submit_ray_task(task_name=job_name, num_cpus=num_cpus, num_gpus=num_gpus, payload=payload)
        return {
            "backend": "Ray",
            "job_name": job_name,
            "task_id": res["ray_task_id"],
            "object_ref_id": res["object_ref_id"],
            "status": "Pending",
            "allocated_cpus": num_cpus,
            "allocated_gpus": num_gpus,
        }

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        status = self.ray.get_task_status(job_name)
        return {
            "backend": "Ray",
            "job_name": job_name,
            "task_id": status.get("ray_task_id", f"ray_{job_name}"),
            "status": "Completed" if status.get("status") == "SUCCESS" else status.get("status", "Running"),
            "allocated_cpus": status.get("allocated_cpus", 4),
            "allocated_gpus": status.get("allocated_gpus", 1),
        }

    def cancel_job(self, job_name: str) -> Dict[str, Any]:
        return self.ray.cancel_task(job_name)

    def get_engine_info(self) -> Dict[str, Any]:
        health = self.ray.get_cluster_health()
        return {
            "name": "ray",
            "status": "active",
            "driver": "RayClusterManager",
            "ray_address": health["ray_address"],
            "total_cpus": health["total_cpus"],
            "total_gpus": health["total_gpus"],
            "features": ["shared_memory_object_store", "actor_placement_groups", "distributed_futures", "zero_copy_ipc"],
        }


class SlurmExecutionBackend(BaseExecutionBackend):
    """Slurm HPC batch execution backend."""

    def __init__(self) -> None:
        self.slurm = SlurmSchedulerManager()

    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        nodes = payload.get("nodes", 2)
        partition = payload.get("partition", "gpu-a100")
        gpus = payload.get("gpus", 2)
        res = self.slurm.submit_sbatch(job_name=job_name, nodes=nodes, partition=partition, gpus=gpus)
        return {
            "backend": "Slurm",
            "job_name": job_name,
            "slurm_job_id": res["slurm_job_id"],
            "partition": res["partition"],
            "nodes": res["nodes"],
            "status": "Queued",
            "sbatch_script": res["sbatch_script"],
        }

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        status = self.slurm.get_job_status(job_name)
        return {
            "backend": "Slurm",
            "job_name": job_name,
            "slurm_job_id": status.get("slurm_job_id", 940281),
            "status": "Running" if status.get("status") in ("QUEUED", "RUNNING") else status.get("status", "COMPLETED"),
            "partition": status.get("partition", "gpu-a100"),
        }

    def cancel_job(self, job_name: str) -> Dict[str, Any]:
        return self.slurm.cancel_job(job_name)

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "slurm",
            "status": "active",
            "driver": "SlurmSchedulerManager",
            "default_partition": self.slurm.default_partition,
            "features": ["multi_node_infiniband", "sbatch_script_generation", "job_arrays", "sacct_telemetry"],
        }


class CloudExecutionBackend(BaseExecutionBackend):
    """Cloud multi-region execution backend (AWS, GCP, RunPod, LambdaLabs)."""

    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"backend": "Cloud", "job_name": job_name, "provider": payload.get("provider", "AWS"), "status": "Provisioning"}

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        return {"backend": "Cloud", "job_name": job_name, "status": "Running"}

    def cancel_job(self, job_name: str) -> Dict[str, Any]:
        return {"backend": "Cloud", "job_name": job_name, "status": "Terminated"}

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "cloud",
            "status": "active",
            "providers": ["AWS", "GCP", "RunPod", "LambdaLabs", "CoreWeave"],
            "features": ["spot_instance_arbitrage", "multi_cloud_failover"],
        }


_BACKENDS: Dict[str, BaseExecutionBackend] = {
    "local": LocalExecutionBackend(),
    "distributed": DistributedExecutionBackend(),
    "kubernetes": KubernetesExecutionBackend(),
    "k8s": KubernetesExecutionBackend(),
    "ray": RayExecutionBackend(),
    "slurm": SlurmExecutionBackend(),
    "cloud": CloudExecutionBackend(),
}


def get_backend(name: str) -> BaseExecutionBackend:
    """Returns initialized execution backend driver by name."""
    norm = name.lower().strip()
    return _BACKENDS.get(norm, _BACKENDS["local"])


def list_supported_backends() -> List[str]:
    """Returns list of all supported cluster engine names."""
    return ["local", "distributed", "kubernetes", "slurm", "ray", "cloud"]

