"""Behavior Inspector for LLM interactions.

Inspects model behavior across probe inputs, computing metrics like token
probabilities, top tokens, attention strength, and response lengths to reveal
behavioral patterns.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from .engine import InteractionEngine
from .models import InspectionResult, InspectionSpec


class BehaviorInspector:
    """Inspects model behavior across a set of probe inputs."""

    def __init__(self, engine: Optional[InteractionEngine] = None) -> None:
        self.engine = engine or InteractionEngine()

    def inspect(self, spec: InspectionSpec) -> InspectionResult:
        """Run behavior inspection across probe inputs."""
        probes: List[Dict[str, Any]] = []

        # Always include the base prompt
        all_inputs = [spec.prompt] + spec.probe_inputs

        for input_text in all_inputs:
            record = self.engine.send_prompt(
                prompt=input_text,
                model=spec.model,
                backend=spec.backend,
                session_id=f"inspection:{spec.prompt[:40]}",
                metadata={"inspection": True},
            )
            response = record.response.to_dict()
            probes.append(
                {
                    "input": input_text,
                    "is_base": input_text == spec.prompt,
                    "response": response,
                    "metrics": self._compute_metrics(response),
                }
            )

        insights = self._generate_insights(probes, spec)

        return InspectionResult(
            inspection_id=f"insp_{abs(hash(spec.prompt + spec.model)) % 100000}",
            spec=spec,
            probes=probes,
            insights=insights,
        )

    def _compute_metrics(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Compute per-probe metrics."""
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
                metrics["top_tokens"] = [t.get("token", "") for t in top5]
            next_token = raw.get("next_token")
            if next_token:
                metrics["next_token"] = next_token

            # Attention strength from raw if available
            attention_maps = raw.get("attention_maps")
            if attention_maps:
                strengths = []
                for am in attention_maps:
                    matrix = am.get("matrix", [])
                    if matrix:
                        strengths.append(max(max(row) for row in matrix))
                if strengths:
                    metrics["attention_strength"] = round(statistics.mean(strengths), 4)

        return metrics

    def _generate_insights(self, probes: List[Dict[str, Any]], spec: InspectionSpec) -> List[str]:
        """Generate human-readable insights from probe results."""
        insights: List[str] = []
        successful = [p for p in probes if not p["metrics"].get("has_error")]

        if not successful:
            insights.append("All probes failed — check model availability.")
            return insights

        # Top token consistency
        top_tokens = [p["metrics"].get("top_token") for p in successful if p["metrics"].get("top_token")]
        if len(top_tokens) >= 2:
            unique = set(top_tokens)
            if len(unique) == 1:
                insights.append(f"Model consistently predicts '{top_tokens[0]}' across all probes.")
            else:
                insights.append(
                    f"Model behavior varies across probes — top tokens: {', '.join(str(t) for t in unique)}."
                )

        # Response length variation
        lengths = [p["metrics"].get("response_length", 0) for p in successful]
        if lengths:
            mean_len = statistics.mean(lengths)
            if len(lengths) >= 2:
                stdev = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
                cv = stdev / mean_len if mean_len > 0 else 0
                if cv < 0.1:
                    insights.append(f"Response length is highly consistent (CV={cv:.3f}).")
                elif cv > 0.5:
                    insights.append(f"Response length varies significantly across probes (CV={cv:.3f}).")

        # Probe sensitivity
        base_probe = next((p for p in probes if p.get("is_base")), None)
        if base_probe and len(probes) > 1:
            base_token = base_probe["metrics"].get("top_token")
            changed = sum(
                1 for p in probes if not p.get("is_base") and p["metrics"].get("top_token") != base_token
            )
            total_probes = len(probes) - 1
            if total_probes > 0:
                sensitivity = changed / total_probes
                if sensitivity == 0:
                    insights.append("Model is insensitive to probe input variations.")
                elif sensitivity > 0.5:
                    insights.append(f"Model is highly sensitive to input variations ({sensitivity:.0%} changed).")
                else:
                    insights.append(f"Model shows moderate sensitivity to input variations ({sensitivity:.0%} changed).")

        return insights