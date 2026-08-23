"""Controlled Experiment Runner for LLM interactions.

Runs the same prompts across multiple models/backends and computes summary
metrics to support controlled experiments.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from .engine import InteractionEngine
from .models import ExperimentResult, ExperimentSpec


class ExperimentRunner:
    """Runs controlled experiments across models and inputs."""

    def __init__(self, engine: Optional[InteractionEngine] = None) -> None:
        self.engine = engine or InteractionEngine()

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        """Run a controlled experiment and return results."""
        runs: List[Dict[str, Any]] = []

        for model in spec.models:
            for backend in spec.backends:
                for prompt in spec.prompts:
                    record = self.engine.send_prompt(
                        prompt=prompt,
                        model=model,
                        backend=backend,
                        temperature=spec.temperature,
                        max_tokens=spec.max_tokens,
                        session_id=f"experiment:{spec.name}",
                        metadata={"experiment": spec.name},
                    )
                    runs.append(
                        {
                            "model": model,
                            "backend": backend,
                            "prompt": prompt,
                            "response": record.response.to_dict(),
                            "metrics": self._compute_metrics(record.response.to_dict()),
                        }
                    )

        # Also run control prompts if provided
        control_runs: List[Dict[str, Any]] = []
        for model in spec.models:
            for backend in spec.backends:
                for prompt in spec.control_prompts:
                    record = self.engine.send_prompt(
                        prompt=prompt,
                        model=model,
                        backend=backend,
                        temperature=spec.temperature,
                        max_tokens=spec.max_tokens,
                        session_id=f"experiment:{spec.name}:control",
                        metadata={"experiment": spec.name, "control": True},
                    )
                    control_runs.append(
                        {
                            "model": model,
                            "backend": backend,
                            "prompt": prompt,
                            "response": record.response.to_dict(),
                            "metrics": self._compute_metrics(record.response.to_dict()),
                        }
                    )

        summary = self._summarize(runs, control_runs, spec)

        return ExperimentResult(
            experiment_id=f"exp_{abs(hash(spec.name)) % 100000}",
            spec=spec,
            runs=runs + control_runs,
            summary=summary,
        )

    def _compute_metrics(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Compute deterministic metrics from a response."""
        text = response.get("text", "")
        error = response.get("error")
        latency = response.get("latency_ms")
        usage = response.get("usage", {})

        metrics: Dict[str, Any] = {
            "response_length": len(text.split()) if text else 0,
            "char_length": len(text) if text else 0,
            "has_error": bool(error),
            "error": error,
        }
        if latency is not None:
            metrics["latency_ms"] = latency
        if usage:
            metrics["prompt_tokens"] = usage.get("prompt_tokens", 0)
            metrics["completion_tokens"] = usage.get("completion_tokens", 0)
            metrics["total_tokens"] = usage.get("total_tokens", 0)

        # Extract top token from raw if available
        raw = response.get("raw", {})
        if raw:
            top5 = raw.get("top5")
            if top5:
                metrics["top_token"] = top5[0].get("token", "")
                metrics["top_token_prob"] = top5[0].get("prob", 0.0)
            next_token = raw.get("next_token")
            if next_token:
                metrics["next_token"] = next_token

        return metrics

    def _summarize(
        self,
        runs: List[Dict[str, Any]],
        control_runs: List[Dict[str, Any]],
        spec: ExperimentSpec,
    ) -> Dict[str, Any]:
        """Compute aggregate summary statistics."""
        summary: Dict[str, Any] = {
            "total_runs": len(runs),
            "control_runs": len(control_runs),
            "successful_runs": sum(1 for r in runs if not r["metrics"].get("has_error")),
            "failed_runs": sum(1 for r in runs if r["metrics"].get("has_error")),
            "models_tested": spec.models,
            "backends_tested": spec.backends,
            "prompts_tested": len(spec.prompts),
            "per_model": {},
        }

        # Per-model aggregation
        for model in spec.models:
            model_runs = [r for r in runs if r["model"] == model]
            if not model_runs:
                continue
            lengths = [r["metrics"].get("response_length", 0) for r in model_runs]
            latencies = [r["metrics"].get("latency_ms", 0) for r in model_runs if r["metrics"].get("latency_ms") is not None]
            summary["per_model"][model] = {
                "runs": len(model_runs),
                "avg_response_length": round(statistics.mean(lengths), 2) if lengths else 0,
                "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
                "success_rate": round(
                    sum(1 for r in model_runs if not r["metrics"].get("has_error")) / len(model_runs),
                    4,
                ),
            }

        # Control comparison if controls exist
        if control_runs:
            treatment_lengths = [r["metrics"].get("response_length", 0) for r in runs]
            control_lengths = [r["metrics"].get("response_length", 0) for r in control_runs]
            summary["control_comparison"] = {
                "treatment_avg_length": round(statistics.mean(treatment_lengths), 2) if treatment_lengths else 0,
                "control_avg_length": round(statistics.mean(control_lengths), 2) if control_lengths else 0,
                "length_delta": round(
                    statistics.mean(treatment_lengths) - statistics.mean(control_lengths), 2
                )
                if treatment_lengths and control_lengths
                else 0,
            }

        return summary