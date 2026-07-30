"""Shared execution data types."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenizedPrompt:
    """Prompt text converted to tokenizer tokens and tensor IDs."""

    prompt: str
    tokens: list[str]
    token_ids: list[int]
    input_ids_tensor: torch.Tensor
    attention_mask_tensor: torch.Tensor
