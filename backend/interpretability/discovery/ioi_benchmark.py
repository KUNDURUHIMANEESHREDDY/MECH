"""Live Indirect Object Identification (IOI) Benchmark Suite for MECH.

Measures empirical IOI task accuracy, logit difference (z_IO - z_S),
and probability margins on live language model forward passes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import torch

logger = logging.getLogger("MECH.ioi_benchmark")

# Canonical IOI template triples: (prompt, indirect_object, subject)
DEFAULT_IOI_PROMPTS: List[Tuple[str, str, str]] = [
    ("When Mary and John went to the store, John gave a drink to", " Mary", " John"),
    ("Then Alice and Bob had a chat, Bob gave a book to", " Alice", " Bob"),
    ("After Sarah and David visited the park, David handed a coffee to", " Sarah", " David"),
    ("While Michael and Jessica finished dinner, Jessica passed the salt to", " Michael", " Jessica"),
    ("When James and Emma walked into the office, Emma assigned the project to", " James", " Emma"),
    ("After Daniel and Olivia finished studying, Olivia returned the notes to", " Daniel", " Olivia"),
]


class IOIBenchmarkSuite:
    """Evaluates Indirect Object Identification (IOI) accuracy and logit diffs on live models."""

    def __init__(self, model: Any = None, tokenizer: Any = None, model_name: str = "gpt2") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

    def _ensure_model(self) -> None:
        if self.model is None or self.tokenizer is None:
            import backend.services.gpt2_engine as gpt2_engine
            gpt2_engine.load()
            self.model = gpt2_engine._model
            self.tokenizer = gpt2_engine._tokenizer

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Live model '{self.model_name}' is uninitialized for IOI benchmark.")

    def run_ioi_eval(
        self,
        prompts: Optional[List[Tuple[str, str, str]]] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs empirical IOI benchmark evaluating logit diffs and task accuracy."""
        self._ensure_model()
        eval_prompts = prompts if prompts is not None else DEFAULT_IOI_PROMPTS
        target_model = model_name or self.model_name

        total_logit_diff = 0.0
        correct_count = 0
        prompt_results: List[Dict[str, Any]] = []

        for prompt_text, io_token, s_token in eval_prompts:
            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            io_ids = self.tokenizer.encode(io_token)
            s_ids = self.tokenizer.encode(s_token)
            io_id = io_ids[-1] if io_ids else 0
            s_id = s_ids[-1] if s_ids else 0

            with torch.no_grad():
                out = self.model(**inputs)

            logits = out.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

            z_io = float(logits[io_id].item())
            z_s = float(logits[s_id].item())
            p_io = float(probs[io_id].item())
            p_s = float(probs[s_id].item())

            logit_diff = z_io - z_s
            is_correct = z_io > z_s

            total_logit_diff += logit_diff
            if is_correct:
                correct_count += 1

            prompt_results.append({
                "prompt": prompt_text,
                "io_token": io_token,
                "s_token": s_token,
                "z_io": round(z_io, 3),
                "z_s": round(z_s, 3),
                "logit_diff": round(logit_diff, 3),
                "p_io": round(p_io, 4),
                "p_s": round(p_s, 4),
                "is_correct": is_correct,
            })

        n = len(eval_prompts)
        avg_logit_diff = round(total_logit_diff / max(1, n), 4)
        accuracy = round(correct_count / max(1, n), 4)
        status = "Passed" if accuracy >= 0.60 else "Failed"

        return {
            "benchmark_name": "IOI Benchmark",
            "model_name": target_model,
            "ioi_accuracy": accuracy,
            "average_logit_diff": avg_logit_diff,
            "num_prompts": n,
            "status": status,
            "prompt_results": prompt_results,
            "provenance": "LIVE_PYTORCH",
        }
