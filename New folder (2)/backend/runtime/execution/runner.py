"""Prompt execution through a loaded causal language model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from backend.runtime.cache import ActivationCache
from backend.runtime.errors import (
    InvalidPromptError,
    ModelOutOfMemoryError,
    RuntimeExecutionError,
    is_out_of_memory_error,
)
from backend.runtime.hooks import ActivationHookManager
from backend.runtime.models.gpt2 import GPT2ModelLoader, LoadedModel


@dataclass(frozen=True)
class ExecutionResult:
    """Result of one prompt forward pass."""

    model_name: str
    prompt: str
    tokens: list[str]
    token_ids: list[int]
    logits: torch.Tensor
    activations: ActivationCache
    metadata: dict[str, object]

    def to_dict(self, *, include_tensors: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "model_name": self.model_name,
            "prompt": self.prompt,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "metadata": self.metadata,
            "activations": self.activations.metadata(),
            "logits_shape": tuple(self.logits.shape),
        }
        if include_tensors:
            result["activation_tensors"] = self.activations.tensors()
            result["logits"] = self.logits
        return result


class ModelExecutor:
    """Loads a supported model and runs prompts with activation capture."""

    def __init__(
        self,
        *,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        local_files_only: bool = False,
        model_loader: Optional[GPT2ModelLoader] = None,
    ) -> None:
        self._loader = model_loader or GPT2ModelLoader(
            model_name=model_name,
            device=device,
            local_files_only=local_files_only,
        )
        self._loaded_model: Optional[LoadedModel] = None

    def load(self) -> LoadedModel:
        if self._loaded_model is None:
            self._loaded_model = self._loader.load()
        return self._loaded_model

    def run_prompt(
        self,
        prompt: str,
        *,
        full_capture: bool = False,
    ) -> ExecutionResult:
        """Run a prompt through the model and capture activations.

        Args:
            prompt: Input text prompt.
            full_capture: When True, captures attention weights and MLP
                intermediate states in addition to the standard activations.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidPromptError("Prompt must be a non-empty string.")

        loaded = self.load()
        tokenized = loaded.tokenize_prompt(prompt)

        input_ids = tokenized.input_ids_tensor.to(loaded.device)
        attention_mask = tokenized.attention_mask_tensor.to(loaded.device)
        cache = ActivationCache()
        hook_manager = ActivationHookManager(cache)

        try:
            if full_capture:
                hook_manager.attach_gpt2_full(loaded.model)
            else:
                hook_manager.attach_gpt2(loaded.model)
            with torch.inference_mode():
                outputs = loaded.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=full_capture,
                )
        except RuntimeError as error:
            if is_out_of_memory_error(error):
                raise ModelOutOfMemoryError(
                    f"Out of memory while running model '{loaded.model_name}'."
                ) from error
            raise RuntimeExecutionError(
                f"Model execution failed for '{loaded.model_name}': {error}"
            ) from error
        finally:
            hook_manager.remove()

        logits = self._extract_logits(outputs)
        cache.store("logits", logits, kind="logits")

        # When full_capture is enabled, also extract attention weights from
        # the model outputs as a fallback (in case hooks missed them).
        if full_capture:
            attentions = getattr(outputs, "attentions", None)
            if attentions is not None:
                for layer_idx, attn in enumerate(attentions):
                    if torch.is_tensor(attn):
                        cache.store(
                            f"layers.{layer_idx}.attention_weights",
                            attn,
                            kind="attention_weights",
                            layer=layer_idx,
                        )

        return ExecutionResult(
            model_name=loaded.model_name,
            prompt=prompt,
            tokens=tokenized.tokens,
            token_ids=tokenized.token_ids,
            logits=logits.detach().cpu().clone(),
            activations=cache,
            metadata={
                "device": str(loaded.device),
                "num_layers": loaded.num_layers,
                "sequence_length": len(tokenized.token_ids),
                "full_capture": full_capture,
            },
        )

    @staticmethod
    def _extract_logits(outputs: object) -> torch.Tensor:
        if hasattr(outputs, "logits"):
            return getattr(outputs, "logits")
        if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
            return outputs[0]
        if torch.is_tensor(outputs):
            return outputs
        raise RuntimeExecutionError("Model did not return logits.")
