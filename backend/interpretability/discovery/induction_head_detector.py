"""Empirical Induction Head Detection Engine for MECH.

Measures the in-context copy-and-paste attention pattern (A B ... A -> B)
by evaluating attention matrices across repeated token sequences on live model weights.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import torch

logger = logging.getLogger("MECH.induction_detector")


class InductionHeadDetector:
    """Detects induction heads across transformer attention layers using live weights."""

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
            raise RuntimeError(f"Live model '{self.model_name}' is uninitialized for induction head detection.")

    def detect_induction_heads(
        self,
        sequence_prefix: Optional[str] = None,
        threshold: float = 0.15,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Measures induction head scores across all (layer, head) pairs on a repeated sequence."""
        self._ensure_model()

        # 1. Construct repeated token sequence
        # In-context pattern: [T_1, T_2, ... T_K, T_1, T_2, ... T_K]
        words = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        )
        prompt = sequence_prefix if sequence_prefix is not None else words

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        
        if seq_len < 6:
            raise ValueError(f"Sequence length ({seq_len}) too short for induction head detection.")

        half_len = seq_len // 2
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # 2. Run live forward pass with attention capture
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        if outputs.attentions is None or len(outputs.attentions) == 0:
            raise RuntimeError("Model did not return attention matrices.")

        num_layers = len(outputs.attentions)
        results: List[Dict[str, Any]] = []

        # 3. Measure empirical induction score per head
        # An induction head at position (half_len + i) attends to position (i) (previous token instance)
        for layer_idx, layer_attn in enumerate(outputs.attentions):
            # layer_attn: [batch=1, num_heads, seq_len, seq_len]
            attn = layer_attn[0]  # [num_heads, seq_len, seq_len]
            num_heads = attn.shape[0]

            for head_idx in range(num_heads):
                mat = attn[head_idx]  # [seq_len, seq_len]
                scores = []
                for i in range(1, half_len):
                    curr_pos = half_len + i
                    if curr_pos < seq_len:
                        # Attention from current repeated token to the position immediately preceding/following in the first half
                        target_prev_pos = i  # previous token copy position
                        scores.append(float(mat[curr_pos, target_prev_pos].item()))

                induction_score = round(sum(scores) / max(1, len(scores)), 4)
                is_induction = bool(induction_score >= threshold)

                results.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "induction_score": induction_score,
                    "prefix_score": induction_score,
                    "is_induction_head": is_induction,
                    "provenance": "LIVE_PYTORCH",
                })

        # Sort descending by empirical induction score
        results.sort(key=lambda x: x["induction_score"], reverse=True)
        return results[:top_k]
