"""Mechanistic Benchmark Runner.

Executes the IOI benchmark against the live GPT-2 Small weights: clean
top-1 accuracy plus corrupted-variant robustness over a fixed prompt panel.
Anything else (including other benchmark names) fails closed rather than
returning a plausible score: a benchmark result is scientific evidence only
when a real executor returns measured fields.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.agents.evidence_policy import field_map


class MechanisticBenchmarkRunner:
    """Run live benchmarks when the model is connected, else fail closed."""

    def run_benchmark(self, benchmark_id: str = "bench_default") -> Dict[str, Any]:
        name = str(benchmark_id or "")
        short = name[6:] if name.startswith("bench_") else name
        if short.lower() != "ioi":
            return {
                "benchmark_id": benchmark_id,
                "status": "unavailable",
                "provenance": "unavailable",
                "field_provenance": field_map(
                    ("benchmark_id", "status", "reason"), "unavailable"
                ),
                "reason": (f"No live benchmark executor is connected for "
                           f"'{short}'."),
            }
        try:
            from backend.interpretability.discovery import live_measure as lm
            from backend.interpretability.discovery.live_discovery import (
                NAMES,
                prompt_panel,
            )
        except Exception as exc:
            return {
                "benchmark_id": benchmark_id,
                "status": "error",
                "provenance": "unavailable",
                "field_provenance": field_map(
                    ("benchmark_id", "status", "error"), "unavailable"
                ),
                "reason": f"Live benchmark imports failed: {exc}",
            }
        try:
            token_ids = lm.single_token_names(NAMES)
            missing = [n for n, tid in token_ids.items() if tid is None]
            if missing:
                raise lm.LiveUnavailable(
                    "IOI comparison tokens are not single vocabulary items: "
                    + ", ".join(sorted(missing)))
            pairs = prompt_panel(6)
            clean_hits = 0
            corr_hits = 0
            per_prompt = []
            for subj, io_name in pairs:
                clean = lm.clean_prompt(subj, io_name)
                corr = lm.corrupted_prompt(subj, io_name)
                io_id = token_ids[io_name]
                subj_id = token_ids[subj]
                assert io_id is not None and subj_id is not None
                clean_base = lm.baseline(clean, io_id, subj_id)
                corr_base = lm.baseline(corr, io_id, subj_id)
                clean_ok = clean_base["top1"].strip() == io_name
                # Corrupted variant swaps the giver: the subject becomes the
                # correct continuation.
                corr_ok = corr_base["top1"].strip() == subj
                clean_hits += 1 if clean_ok else 0
                corr_hits += 1 if corr_ok else 0
                per_prompt.append({
                    "subject": subj, "io": io_name,
                    "clean_top1": clean_base["top1"],
                    "clean_correct": clean_ok,
                    "corrupted_top1": corr_base["top1"],
                    "corrupted_correct": corr_ok,
                })
            accuracy = clean_hits / len(pairs)
            robustness = corr_hits / len(pairs)
            return {
                "benchmark_id": benchmark_id,
                "status": "completed",
                "provenance": "live",
                "field_provenance": field_map(
                    ("benchmark_id", "status", "accuracy",
                     "robustness_score", "eval_samples", "per_prompt"),
                    "live",
                ),
                "method": ("clean top-1 agreement plus corrupted-variant "
                           "top-1 agreement over a fixed IOI prompt panel"),
                "model_id": "gpt2-small",
                "accuracy": round(accuracy, 4),
                "robustness_score": round(robustness, 4),
                "eval_samples": len(pairs),
                "per_prompt": per_prompt,
            }
        except lm.LiveUnavailable as exc:
            return {
                "benchmark_id": benchmark_id,
                "status": "unavailable",
                "provenance": "unavailable",
                "field_provenance": field_map(
                    ("benchmark_id", "status", "reason"), "unavailable"
                ),
                "reason": str(exc),
            }
        except Exception as exc:
            return {
                "benchmark_id": benchmark_id,
                "status": "error",
                "provenance": "unavailable",
                "field_provenance": field_map(
                    ("benchmark_id", "status", "error"), "unavailable"
                ),
                "error": str(exc)[:500],
            }
