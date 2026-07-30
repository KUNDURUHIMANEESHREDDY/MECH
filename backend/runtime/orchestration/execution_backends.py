"""Execution Backend Abstraction Layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
import datetime as _dt
from typing import Any, Dict


class BaseExecutionBackend(ABC):
    """Abstract base class for all execution backends."""

    @abstractmethod
    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        pass


class LocalExecutionBackend(BaseExecutionBackend):
    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"backend": "Local", "job_name": job_name, "status": "Running"}

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        return {"backend": "Local", "job_name": job_name, "status": "Completed"}


class KubernetesExecutionBackend(BaseExecutionBackend):
    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"backend": "Kubernetes", "job_name": job_name, "pod_id": f"pod_{job_name}", "status": "Running"}

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        return {"backend": "Kubernetes", "job_name": job_name, "status": "Running"}


class RayExecutionBackend(BaseExecutionBackend):
    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"backend": "Ray", "job_name": job_name, "task_id": f"ray_{job_name}", "status": "Pending"}

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        return {"backend": "Ray", "job_name": job_name, "status": "Completed"}


class SlurmExecutionBackend(BaseExecutionBackend):
    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"backend": "Slurm", "job_name": job_name, "slurm_job_id": 940281, "status": "Queued"}

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        return {"backend": "Slurm", "job_name": job_name, "status": "Running"}


class CloudExecutionBackend(BaseExecutionBackend):
    def submit_job(self, job_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"backend": "Cloud", "job_name": job_name, "provider": "AWS", "status": "Provisioning"}

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        return {"backend": "Cloud", "job_name": job_name, "status": "Running"}
