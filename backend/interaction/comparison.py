"""Interaction Comparison Engine.

Compares model behavior across different inputs and models, computing
similarity metrics and behavioral differences.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from .engine import InteractionEngine
from .models import ComparisonResult, ComparisonSpec


class InteractionComparisonEngine:
    """Compares behavior between different inputs/models."""

    def __init__(self, engine: Optional[InteractionEngine] = None) -> None:
        self.engine = engine or InteractionEngine()

    def compare(self, spec: ComparisonSpec) -> ComparisonResult:
        """Run a comparison across models for a single prompt."""
        outputs: List[Dict[str, Any]] = []

        for idx, model in enumerate(spec.models):
            backend = spec.backends[idx] if idx < len(spec.backends) else "local"
            record = self.engine.send_prompt(
                prompt=spec.prompt,
                model=model,
                backend=backend,
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
                session_id=f"comparison:{spec.prompt[:40]}",
                metadata={"comparison": True},
            )
            response = record.response.to_dict()
            outputs.append(
                {
                    "model": model,
                    "backend": backend,
                    "response": response,
                    "metrics": self._compute_metrics(response),
                }
            )

        metrics = self._compute_comparison_metrics(outputs)

        return ComparisonResult(
            comparison_id=f"cmp_{abs(hash(spec.prompt + str(spec.models))) % 100000}",
            spec=spec,
            outputs=outputs,
            metrics=metrics,
        )

    def _compute_metrics(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Compute per-output metrics."""
        text = response.get("text", "")
        error = response.get("error")
        latency = response.get("latency_ms")
        usage = response.get("usage", {})

        metrics: Dict[str, Any] = {
            "response_length": len(text.split()) if text else 0,
            "char_length": len(text) if text else 0,
            "has_error": bool(error),
        }
        if latency is not None:
            metrics["latency_ms"] = latency
        if usage:
            metrics["total_tokens"] = usage.get("total_tokens", 0)

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

    def _compute_comparison_metrics(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute cross-model comparison metrics."""
        successful = [o for o in outputs if not o["metrics"].get("has_error")]

        # Token agreement
        top_tokens = [o["metrics"].get("top_token") for o in successful if o["metrics"].get("top_token")]
        token_agreement = 0.0
        if len(top_tokens) >= 2:
            token_agreement = round(
                sum(1 for t in top_tokens[1:] if t == top_tokens[0]) / (len(top_tokens) - 1),
                4,
            )

        # Response length similarity (coefficient of variation)
        lengths = [o["metrics"].get("response_length", 0) for o in successful]
        length_cv = 0.0
        if len(lengths) >= 2 and lengths:
            mean = statistics.mean(lengths)
            if mean > 0:
                stdev = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
                length_cv = round(stdev / mean, 4)

        # Latency comparison
        latencies = [o["metrics"].get("latency_ms") for o in successful if o["metrics"].get("latency_ms") is not None]

        return {
            "models_compared": len(outputs),
            "successful_outputs": len(successful),
            "failed_outputs": len(outputs) - len(successful),
            "top_token_agreement": token_agreement,
            "response_length_cv": length_cv,
            "avg_response_length": round(statistics.mean(lengths), 2) if lengths else 0,
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
            "fastest_model": (
                min(successful, key=lambda o: o["metrics"].get("latency_ms") or float("inf"))["model"]
                if successful and any(o["metrics"].get("latency_ms") for o in successful)
                else None
            ),
            "longest_response": (
                max(successful, key=lambda o: o["metrics"].get("response_length", 0))["model"]
                if successful
                else None
            ),
        }