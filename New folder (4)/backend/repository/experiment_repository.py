"""Experiment Repository.

Manages persistence and lookups for circuit ablation and feature steering runs.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional


class ExperimentRepository:
    """Repository for experiment matrix runs."""

    def __init__(self) -> None:
        self._experiments: Dict[str, Dict[str, Any]] = {
            "exp_1": {
                "id": "exp_1",
                "title": "Layer 9 Head 9 Ablation",
                "targetCircuit": "L9_H9",
                "status": "completed",
                "created_at": _dt.datetime.utcnow().isoformat() + "Z",
            }
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        return list(self._experiments.values())

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        return self._experiments.get(exp_id)

    def save_experiment(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        eid = exp.get("id") or f"exp_{len(self._experiments) + 1}"
        exp["id"] = eid
        exp["updated_at"] = _dt.datetime.utcnow().isoformat() + "Z"
        self._experiments[eid] = exp
        return exp


_exp_repo_instance: ExperimentRepository | None = None


def get_experiment_repository() -> ExperimentRepository:
    global _exp_repo_instance
    if _exp_repo_instance is None:
        _exp_repo_instance = ExperimentRepository()
    return _exp_repo_instance
