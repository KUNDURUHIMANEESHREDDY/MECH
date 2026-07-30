"""Autonomous Dependency-Aware Experiment Planner Engine."""

from __future__ import annotations

from typing import Any, Dict, List


class AutonomousPlannerEngine:
    """Generates dependency-aware experiment execution DAG plans with SAE-adaptive strategy."""

    def create_plan(self, goal: str = "Discover IOI Circuit", sae_available: bool = False) -> Dict[str, Any]:
        stages = []

        if sae_available:
            # SAE-Prioritized Path
            stages.append({
                "stage_id": "sae_stage_1",
                "name": "Sparse Feature Clustering",
                "dependencies": [],
                "action": "run_feature_clustering",
            })
            stages.append({
                "stage_id": "sae_stage_2",
                "name": "Transcoder Discovery",
                "dependencies": ["sae_stage_1"],
                "action": "run_transcoder_search",
            })
        else:
            # Baseline Neuron-based Path
            stages.append({
                "stage_id": "exp_stage_1",
                "name": "Activation Search & Causal Tracing",
                "dependencies": [],
                "action": "run_causal_tracing",
            })

        stages.append({
            "stage_id": "exp_stage_final",
            "name": "Circuit Graph Extraction",
            "dependencies": [stages[-1]["stage_id"]],
            "action": "discover_circuit",
        })

        return {
            "plan_id": f"plan_{hash(goal) & 0xffffffff:08x}",
            "goal": goal,
            "experiment_stages": stages,
            "status": "planned",
            "strategy": "SAE-Enhanced" if sae_available else "Neuron-Baseline"
        }
