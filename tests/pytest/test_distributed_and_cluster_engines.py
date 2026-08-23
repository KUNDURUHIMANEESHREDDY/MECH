"""Unit and Integration Tests for Distributed Runtime & Cluster Scaling Drivers.

Verifies:
1. DistributedScheduler with multi-worker concurrent execution & thread safety.
2. DistributedWorker dynamic adapter resolution & algorithm execution.
3. Kubernetes cluster driver & Batch/v1 Job manifest generator.
4. Slurm HPC cluster driver & #SBATCH script generator.
5. Ray distributed task, actor, and cluster manager.
6. Unified Execution Backend registry and Dispatcher API endpoints.
"""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def test_distributed_worker_dynamic_adapter():
    from backend.distributed.worker import DistributedWorker, ExperimentTask

    worker = DistributedWorker(worker_id="gpu_worker_0")
    task = ExperimentTask(
        task_id="t_dyn_1",
        campaign_id="c_dyn",
        algorithm_name="acdc",
        model_id="gemma-2b",
        dataset_shard={"prompt": "Paris is the capital of France", "threshold": 0.05},
        config_override={"mock_mode": True},
    )
    result = worker.execute_task(task)
    assert result.status == "Success"
    assert result.task_id == "t_dyn_1"
    assert result.worker_id == "gpu_worker_0"
    assert result.execution_time_ms >= 0.0
    assert result.compute_flops > 0


def test_distributed_scheduler_multi_worker_parallel():
    from backend.distributed.scheduler import DistributedScheduler
    from backend.distributed.worker import ExperimentTask

    scheduler = DistributedScheduler(default_workers=4)
    tasks = [
        ExperimentTask(
            task_id=f"task_par_{i}",
            campaign_id="camp_par_1",
            algorithm_name="acdc",
            model_id="gpt2-small",
            dataset_shard={"prompt": f"Prompt sample {i}"},
            config_override={"mock_mode": True},
            priority=i,
        )
        for i in range(5)
    ]

    results = scheduler.dispatch_batch_parallel(tasks, max_workers=3)
    assert len(results) == 5
    assert all(r.status == "Success" for r in results)

    # Verify checkpoint
    cp = scheduler.checkpoint("camp_par_1")
    assert cp.completed_tasks == 5
    assert cp.failed_tasks == 0


def test_kubernetes_runtime_driver():
    from backend.runtime.orchestration.k8s_runtime import KubernetesRuntimeManager

    k8s = KubernetesRuntimeManager(default_namespace="mech-test")
    manifest = k8s.generate_job_manifest(
        job_name="exp-circuit-acdc",
        image="interp/runtime:v6",
        num_gpus=2,
        num_cpus=8,
        memory_gb=32,
        env={"MODEL_NAME": "gemma-2b"},
    )
    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["namespace"] == "mech-test"
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"

    deploy_res = k8s.deploy_job("exp-circuit-acdc", num_gpus=2)
    assert deploy_res["status"] == "Running"
    assert "pod_id" in deploy_res

    status = k8s.get_job_status("exp-circuit-acdc")
    assert status["status"] == "Running"
    assert len(status["logs"]) > 0

    cancel_res = k8s.cancel_job("exp-circuit-acdc")
    assert cancel_res["status"] == "Terminated"


def test_slurm_hpc_driver():
    from backend.runtime.orchestration.slurm_scheduler import SlurmSchedulerManager

    slurm = SlurmSchedulerManager(default_partition="gpu-h100")
    script = slurm.generate_sbatch_script(
        job_name="mech_huge_trace",
        nodes=4,
        gpus=4,
        partition="gpu-h100",
        mem_gb=128,
        time_limit="08:00:00",
    )
    assert "#SBATCH --job-name=mech_huge_trace" in script
    assert "#SBATCH --partition=gpu-h100" in script
    assert "#SBATCH --gres=gpu:4" in script
    assert "#SBATCH --nodes=4" in script
    assert "module load cuda/12.2" in script

    sub_res = slurm.submit_sbatch("mech_huge_trace", nodes=4, gpus=4)
    assert sub_res["status"] == "QUEUED"
    assert sub_res["slurm_job_id"] >= 940281

    status = slurm.get_job_status(sub_res["slurm_job_id"])
    assert status["status"] in ("QUEUED", "RUNNING")

    cancel_res = slurm.cancel_job(sub_res["slurm_job_id"])
    assert cancel_res["status"] == "CANCELLED"


def test_ray_cluster_driver():
    from backend.runtime.orchestration.ray_integration import RayClusterManager

    ray_mgr = RayClusterManager()
    task_res = ray_mgr.submit_ray_task(
        task_name="ray_activation_patch",
        num_cpus=8,
        num_gpus=2,
    )
    assert task_res["status"] == "PENDING"
    assert "ray_task_id" in task_res
    assert "object_ref_id" in task_res

    actor_res = ray_mgr.register_actor("worker_actor_0", num_gpus=1)
    assert actor_res["status"] == "ALIVE"

    health = ray_mgr.get_cluster_health()
    assert health["cluster_status"] == "ONLINE"
    assert health["allocated_cpus"] == 8
    assert health["allocated_gpus"] == 2
    assert health["active_actors_count"] == 1


def test_execution_backends_and_api_dispatcher():
    from backend.runtime.orchestration.execution_backends import get_backend, list_supported_backends
    from backend.api.dispatcher import list_runtime_engines, runtime_status, runtime_submit_job, runtime_get_job_status, runtime_cancel_job

    supported = list_supported_backends()
    assert "local" in supported
    assert "distributed" in supported
    assert "kubernetes" in supported
    assert "slurm" in supported
    assert "ray" in supported

    # Test all backends submit and get status
    for eng in ["local", "distributed", "kubernetes", "slurm", "ray"]:
        backend = get_backend(eng)
        info = backend.get_engine_info()
        assert info["status"] == "active"

        sub = backend.submit_job(f"test_job_{eng}", {"algorithm_name": "acdc", "model_id": "gpt2-small"})
        assert "job_name" in sub or "backend" in sub

    # Test API dispatcher functions
    engines_res = list_runtime_engines()
    assert len(engines_res["engines"]) >= 5
    assert "drivers" in engines_res
    assert "kubernetes" in engines_res["drivers"]
    assert "slurm" in engines_res["drivers"]
    assert "ray" in engines_res["drivers"]

    status_res = runtime_status()
    assert status_res["status"] == "connected"
    assert len(status_res["engines"]) >= 5

    # Test API job lifecycle
    api_sub = runtime_submit_job({"engine": "kubernetes", "job_name": "api_k8s_job", "num_gpus": 1})
    assert api_sub["backend"] == "Kubernetes"
    assert api_sub["status"] == "Running"

    api_stat = runtime_get_job_status("kubernetes", "api_k8s_job")
    assert api_stat["status"] == "Running"

    api_cancel = runtime_cancel_job("kubernetes", "api_k8s_job")
    assert api_cancel["status"] == "Terminated"
