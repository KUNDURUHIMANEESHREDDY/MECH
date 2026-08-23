"""Virtual Unembedding Engine for Large Vocabulary Out-of-Core Projections.

Computes vocabulary projections, exact global top-k token heap reductions,
and exact target token probabilities via streaming LogSumExp without allocating
the monolithic [d_model x Vocab] projection matrix simultaneously in GPU VRAM.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


class VirtualUnembeddingEngine:
    """Computes streaming vocabulary projections with exact global reduction."""

    def __init__(self, chunk_size: int = 4096) -> None:
        self.chunk_size = chunk_size

    def project_hidden_state(
        self,
        hidden_state: torch.Tensor,
        unembedding_weights: torch.Tensor | Callable[[int, int], torch.Tensor],
        tokenizer: Any,
        top_k: int = 10,
        target_token: Optional[str] = None,
        target_token_id: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Performs chunked virtual unembedding projection.

        Args:
            hidden_state: [d_model] or [1, d_model] float tensor.
            unembedding_weights: Either a full [d_model, V] or [V, d_model] tensor,
                or a callable `get_chunk(start_idx, end_idx) -> Tensor[d_model, chunk_size]`.
            tokenizer: Tokenizer for vocabulary decoding.
            top_k: Number of highest logit tokens to gather globally.
            target_token: Target token string (e.g. ' Paris').
            target_token_id: Target token integer ID.
            vocab_size: Total vocabulary size.
        """
        # Ensure hidden_state is a 1D vector on CPU or current device
        if hidden_state.dim() > 1:
            h = hidden_state.squeeze()
            if h.dim() > 1:
                h = h[-1]  # Last sequence token
        else:
            h = hidden_state

        if torch.isnan(h).any() or torch.isinf(h).any():
            raise ValueError("Input hidden_state contains NaN or Inf values.")

        device = h.device
        d_model = h.shape[-1]

        # Resolve total vocab size
        if vocab_size is None:
            if callable(unembedding_weights):
                vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else 50257
            else:
                vocab_size = (
                    unembedding_weights.shape[0]
                    if unembedding_weights.shape[1] == d_model
                    else unembedding_weights.shape[1]
                )

        # Resolve target token ID
        if target_token_id is None and target_token is not None:
            encoded = tokenizer.encode(target_token)
            target_token_id = encoded[-1] if encoded else None

        # Min-heap for global top-k: stores (logit, vocab_id)
        global_top_heap: List[Tuple[float, int]] = []

        # Running online LogSumExp tracking: m = max(x), s = sum(exp(x - m))
        running_max = -float("inf")
        running_sum_exp = 0.0

        target_logit: Optional[float] = None
        target_rank: int = 0  # Count of all vocabulary logits > target_logit

        # First pass / unified pass in chunks of size `self.chunk_size`
        for start_idx in range(0, vocab_size, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, vocab_size)

            # Materialize only this chunk of W_U
            if callable(unembedding_weights):
                w_chunk = unembedding_weights(start_idx, end_idx)
            else:
                # Full tensor slice: check orientation
                if unembedding_weights.shape[0] == d_model:
                    w_chunk = unembedding_weights[:, start_idx:end_idx]
                else:
                    # [V, d_model] orientation (standard nn.Linear weight / embedding weight)
                    w_chunk = unembedding_weights[start_idx:end_idx, :].T

            # Project: h @ W_chunk -> logits_chunk [chunk_size]
            with torch.no_grad():
                chunk_logits = torch.matmul(h, w_chunk.to(device))

            chunk_max = float(torch.max(chunk_logits).item())
            
            # Online LogSumExp update
            if chunk_max > running_max:
                running_sum_exp = running_sum_exp * math.exp(running_max - chunk_max) + float(
                    torch.sum(torch.exp(chunk_logits - chunk_max)).item()
                )
                running_max = chunk_max
            else:
                running_sum_exp += float(torch.sum(torch.exp(chunk_logits - running_max)).item())

            # Check for target token logit inside this chunk
            if target_token_id is not None and start_idx <= target_token_id < end_idx:
                target_logit = float(chunk_logits[target_token_id - start_idx].item())

            # Top-k candidates within this chunk
            k_chunk = min(top_k, len(chunk_logits))
            vals, indices = torch.topk(chunk_logits, k=k_chunk)
            for v, idx in zip(vals.tolist(), indices.tolist()):
                global_idx = start_idx + idx
                if len(global_top_heap) < top_k:
                    heapq.heappush(global_top_heap, (v, global_idx))
                else:
                    if v > global_top_heap[0][0]:
                        heapq.heappushpop(global_top_heap, (v, global_idx))

        # Final LogSumExp
        global_lse = running_max + math.log(max(running_sum_exp, 1e-12))

        # Second sweep to compute exact global rank of target token if target_logit is known
        if target_logit is not None:
            target_rank = 0
            for start_idx in range(0, vocab_size, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, vocab_size)
                if callable(unembedding_weights):
                    w_chunk = unembedding_weights(start_idx, end_idx)
                else:
                    if unembedding_weights.shape[0] == d_model:
                        w_chunk = unembedding_weights[:, start_idx:end_idx]
                    else:
                        w_chunk = unembedding_weights[start_idx:end_idx, :].T
                with torch.no_grad():
                    chunk_logits = torch.matmul(h, w_chunk.to(device))
                target_rank += int((chunk_logits > target_logit).sum().item())

        # Sort global top-k in descending order
        sorted_top_k = sorted(global_top_heap, key=lambda x: x[0], reverse=True)
        top_candidates = [
            {
                "rank": i + 1,
                "token_id": tid,
                "token": tokenizer.decode([tid]) if tokenizer else f"id_{tid}",
                "logit": round(l, 4),
                "probability": round(math.exp(l - global_lse), 6),
            }
            for i, (l, tid) in enumerate(sorted_top_k)
        ]

        target_prob = math.exp(target_logit - global_lse) if target_logit is not None else None

        return {
            "top_candidates": top_candidates,
            "top_token": top_candidates[0]["token"] if top_candidates else "",
            "top_logit": top_candidates[0]["logit"] if top_candidates else 0.0,
            "target_token": target_token,
            "target_token_id": target_token_id,
            "target_logit": round(target_logit, 4) if target_logit is not None else None,
            "target_probability": round(target_prob, 6) if target_prob is not None else None,
            "target_rank": target_rank if target_logit is not None else None,
            "global_logsumexp": round(global_lse, 4),
            "vocab_size": vocab_size,
        }
