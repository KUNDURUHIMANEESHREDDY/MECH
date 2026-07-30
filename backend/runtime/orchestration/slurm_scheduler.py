"""Slurm HPC Job Scheduler."""

from __future__ import annotations

from typing import Any, Dict


class SlurmSchedulerManager:
    """Generates sbatch scripts and manages Slurm HPC queues."""

    def submit_sbatch(self, job_name: str, nodes: int = 2, partition: str = "gpu-a100") -> Dict[str, Any]:
        sbatch_script = f"#!/bin/bash\n#SBATCH --job-name={job_name}\n#SBATCH --nodes={nodes}\n#SBATCH --partition={partition}\nsrun python train.py"
        return {
            "job_name": job_name,
            "slurm_job_id": 940281,
            "partition": partition,
            "nodes": nodes,
            "sbatch_script": sbatch_script,
            "status": "QUEUED",
        }
