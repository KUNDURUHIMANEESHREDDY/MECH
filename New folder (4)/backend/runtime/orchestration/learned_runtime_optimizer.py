"""Learned Runtime Optimizer Engine with Per-Model Online Learning & Confidence Intervals."""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, List


class LearnedRuntimeOptimizer:
    """Predicts runtime, memory, GPU utilization, cost, and failure probability per-model

    with confidence bounds, continuously learning from empirical execution outcomes.
    """

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []
        self.global_runtime_multiplier: float = 1.0
        self.global_vram_multiplier: float = 1.0
        self.global_cost_multiplier: float = 1.0
        self.model_multipliers: Dict[str, Dict[str, float]] = {}

    def get_multipliers_for_model(self, model_name: str) -> Dict[str, float]:
        if model_name not in self.model_multipliers:
            return {
                "runtime": self.global_runtime_multiplier,
                "vram": self.global_vram_multiplier,
                "cost": self.global_cost_multiplier,
            }
        return self.model_multipliers[model_name]

    def predict_execution(self, model_name: str, num_prompts: int = 1000) -> Dict[str, Any]:
        mults = self.get_multipliers_for_model(model_name)
        base_runtime = num_prompts * 0.005
        base_vram = 8.4
        base_cost = num_prompts * 0.0002

        predicted_runtime = round(base_runtime * mults["runtime"], 2)
        predicted_vram = round(base_vram * mults["vram"], 2)
        predicted_cost = round(base_cost * mults["cost"], 4)

        # Compute model-specific sample count and confidence bounds
        model_history = [h for h in self.history if h["model_name"] == model_name]
        sample_count = len(model_history)
        confidence = min(0.98, round(0.70 + 0.05 * math.log1p(sample_count), 4))

        std_dev_runtime = round(max(0.2, predicted_runtime * (0.25 / math.sqrt(sample_count + 1))), 2)
        std_dev_vram = round(max(0.1, predicted_vram * (0.15 / math.sqrt(sample_count + 1))), 2)

        runtime_interval = [round(max(0.1, predicted_runtime - 1.96 * std_dev_runtime), 2), round(predicted_runtime + 1.96 * std_dev_runtime, 2)]
        vram_interval = [round(max(1.0, predicted_vram - 1.96 * std_dev_vram), 2), round(predicted_vram + 1.96 * std_dev_vram, 2)]

        failure_prob = 0.015 if sample_count == 0 else max(0.001, round(sum(1 for h in model_history if h["failure"]) / sample_count, 4))

        return {
            "model_name": model_name,
            "num_prompts": num_prompts,
            "predicted_runtime_sec": predicted_runtime,
            "predicted_memory_vram_gb": predicted_vram,
            "predicted_gpu_utilization_pct": 92.5,
            "predicted_cost_usd": predicted_cost,
            "prediction_confidence": confidence,
            "runtime_std_dev": std_dev_runtime,
            "runtime_interval": runtime_interval,
            "vram_interval": vram_interval,
            "failure_probability": failure_prob,
            "recommended_backend": "Ray" if num_prompts > 500 else "Local",
            "learned_samples_count": sample_count,
            "total_samples_count": len(self.history),
            "adaptation_multipliers": {
                "runtime": round(mults["runtime"], 3),
                "vram": round(mults["vram"], 3),
                "cost": round(mults["cost"], 3),
            },
        }

    def record_execution_outcome(
        self,
        model_name: str,
        num_prompts: int,
        observed_runtime_sec: float,
        observed_memory_vram_gb: float,
        observed_cost_usd: float,
        failure: bool = False,
    ) -> Dict[str, Any]:
        pred = self.predict_execution(model_name=model_name, num_prompts=num_prompts)
        record = {
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "model_name": model_name,
            "num_prompts": num_prompts,
            "predicted": pred,
            "observed": {
                "runtime_sec": observed_runtime_sec,
                "vram_gb": observed_memory_vram_gb,
                "cost_usd": observed_cost_usd,
            },
            "failure": failure,
        }
        self.history.append(record)

        alpha = 0.3
        expected_base_runtime = max(0.001, num_prompts * 0.005)
        expected_base_vram = 8.4
        expected_base_cost = max(0.0001, num_prompts * 0.0002)

        obs_runtime_mult = observed_runtime_sec / expected_base_runtime
        obs_vram_mult = observed_memory_vram_gb / expected_base_vram
        obs_cost_mult = observed_cost_usd / expected_base_cost

        # Update model-specific scaling multipliers
        curr_mults = self.get_multipliers_for_model(model_name)
        new_mults = {
            "runtime": round((1 - alpha) * curr_mults["runtime"] + alpha * obs_runtime_mult, 4),
            "vram": round((1 - alpha) * curr_mults["vram"] + alpha * obs_vram_mult, 4),
            "cost": round((1 - alpha) * curr_mults["cost"] + alpha * obs_cost_mult, 4),
        }
        self.model_multipliers[model_name] = new_mults

        # Update global fallbacks
        self.global_runtime_multiplier = round((1 - alpha) * self.global_runtime_multiplier + alpha * obs_runtime_mult, 4)
        self.global_vram_multiplier = round((1 - alpha) * self.global_vram_multiplier + alpha * obs_vram_mult, 4)
        self.global_cost_multiplier = round((1 - alpha) * self.global_cost_multiplier + alpha * obs_cost_mult, 4)

        return {
            "status": "OutcomeRecorded",
            "model_name": model_name,
            "learned_samples_count": len([h for h in self.history if h["model_name"] == model_name]),
            "updated_model_multipliers": new_mults,
            "updated_global_multipliers": {
                "runtime": self.global_runtime_multiplier,
                "vram": self.global_vram_multiplier,
                "cost": self.global_cost_multiplier,
            },
        }
