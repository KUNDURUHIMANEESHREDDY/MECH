"""GPT-2 model loading and tokenization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from backend.runtime.errors import (
    InvalidPromptError,
    MissingModelError,
    ModelOutOfMemoryError,
    is_out_of_memory_error,
)
from backend.runtime.execution.types import TokenizedPrompt


GPT2_ALIASES = {
    "gpt2": "gpt2",
    "openai-community/gpt2": "openai-community/gpt2",
}


@dataclass(frozen=True)
class LoadedModel:
    """Loaded causal language model plus tokenizer."""

    model_name: str
    model: torch.nn.Module
    tokenizer: object
    device: torch.device

    @property
    def num_layers(self) -> Optional[int]:
        config = getattr(self.model, "config", None)
        return getattr(config, "n_layer", None)

    def tokenize_prompt(self, prompt: str) -> TokenizedPrompt:
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidPromptError("Prompt must be a non-empty string.")

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        token_ids = input_ids[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return TokenizedPrompt(
            prompt=prompt,
            tokens=tokens,
            token_ids=token_ids,
            input_ids_tensor=input_ids,
            attention_mask_tensor=attention_mask,
        )


class GPT2ModelLoader:
    """Loads GPT-2 models from Hugging Face or a local model directory."""

    def __init__(
        self,
        *,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.local_files_only = local_files_only

    def load(self) -> LoadedModel:
        model_id = self._resolve_model_id(self.model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=self.local_files_only,
            )
            model = GPT2LMHeadModel.from_pretrained(
                model_id,
                local_files_only=self.local_files_only,
            )
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.to(self.device)
            model.eval()
            return LoadedModel(
                model_name=model_id,
                model=model,
                tokenizer=tokenizer,
                device=self.device,
            )
        except OSError as error:
            raise MissingModelError(
                f"Unable to load GPT-2 model '{model_id}'. "
                "Verify the model name, local path, or cached files."
            ) from error
        except RuntimeError as error:
            if is_out_of_memory_error(error):
                raise ModelOutOfMemoryError(
                    f"Out of memory while loading model '{model_id}'."
                ) from error
            raise

    @staticmethod
    def _resolve_model_id(model_name: str) -> str:
        key = model_name.lower()
        if key in GPT2_ALIASES:
            return GPT2_ALIASES[key]

        candidate = Path(model_name)
        if candidate.exists():
            return str(candidate)

        raise MissingModelError(
            f"Unsupported model '{model_name}'. Supported models: {', '.join(sorted(GPT2_ALIASES))}."
        )
