"""Activation hook registration for GPT-2 style models."""

from __future__ import annotations

from typing import Optional

import torch

from backend.runtime.cache import ActivationCache


class ActivationHookManager:
    """Registers and removes forward hooks for activation capture."""

    def __init__(self, cache: ActivationCache) -> None:
        self._cache = cache
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach_gpt2(self, model: torch.nn.Module) -> None:
        transformer = getattr(model, "transformer", None)
        if transformer is None:
            self._register_if_present(model, "lm_head", "logits", "logits")
            return

        self._register_if_present(transformer, "drop", "embedding", "embedding")

        blocks = getattr(transformer, "h", [])
        for layer, block in enumerate(blocks):
            self._register_if_present(
                block,
                "attn",
                f"layers.{layer}.attention_output",
                "attention_output",
                layer=layer,
            )
            self._register_if_present(
                block,
                "mlp",
                f"layers.{layer}.mlp_output",
                "mlp_output",
                layer=layer,
            )
            self._register(
                block,
                f"layers.{layer}.residual",
                "residual",
                layer=layer,
            )

        self._register_if_present(model, "lm_head", "logits", "logits")

    def attach_gpt2_full(self, model: torch.nn.Module) -> None:
        """Attach hooks for the full reasoning journey.

        Captures embeddings, attention outputs, attention weights,
        MLP outputs, MLP intermediate states, residual streams, and logits.
        """
        transformer = getattr(model, "transformer", None)
        if transformer is None:
            self._register_if_present(model, "lm_head", "logits", "logits")
            return

        self._register_if_present(transformer, "drop", "embedding", "embedding")

        blocks = getattr(transformer, "h", [])
        for layer, block in enumerate(blocks):
            # Attention output + attention weights
            self._register_attention_full(
                block,
                layer,
            )
            # MLP output + MLP intermediate (post-activation)
            self._register_mlp_full(
                block,
                layer,
            )
            # Residual (block output)
            self._register(
                block,
                f"layers.{layer}.residual",
                "residual",
                layer=layer,
            )

        self._register_if_present(model, "lm_head", "logits", "logits")

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _register_if_present(
        self,
        parent: object,
        child_name: str,
        activation_name: str,
        kind: str,
        *,
        layer: Optional[int] = None,
    ) -> None:
        module = getattr(parent, child_name, None)
        if isinstance(module, torch.nn.Module):
            self._register(module, activation_name, kind, layer=layer)

    def _register(
        self,
        module: torch.nn.Module,
        activation_name: str,
        kind: str,
        *,
        layer: Optional[int] = None,
    ) -> None:
        def hook(
            hooked_module: torch.nn.Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            del hooked_module, inputs
            tensor = self._first_tensor(output)
            if tensor is not None:
                self._cache.store(
                    activation_name,
                    tensor,
                    kind=kind,
                    layer=layer,
                )

        self._handles.append(module.register_forward_hook(hook))

    def _register_attention_full(
        self,
        block: torch.nn.Module,
        layer: int,
    ) -> None:
        """Hook attention module to capture both output and attention weights."""
        attn = getattr(block, "attn", None)
        if not isinstance(attn, torch.nn.Module):
            # Fallback: register attention_output like the basic attach
            self._register_if_present(
                block,
                "attn",
                f"layers.{layer}.attention_output",
                "attention_output",
                layer=layer,
            )
            return

        def hook(
            hooked_module: torch.nn.Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            del hooked_module, inputs
            if isinstance(output, tuple) and len(output) >= 2:
                # (attn_output, attn_weights)
                attn_output = self._first_tensor(output[0])
                attn_weights = self._first_tensor(output[1])
                if attn_output is not None:
                    self._cache.store(
                        f"layers.{layer}.attention_output",
                        attn_output,
                        kind="attention_output",
                        layer=layer,
                    )
                if attn_weights is not None:
                    self._cache.store(
                        f"layers.{layer}.attention_weights",
                        attn_weights,
                        kind="attention_weights",
                        layer=layer,
                    )
            else:
                tensor = self._first_tensor(output)
                if tensor is not None:
                    self._cache.store(
                        f"layers.{layer}.attention_output",
                        tensor,
                        kind="attention_output",
                        layer=layer,
                    )

        self._handles.append(attn.register_forward_hook(hook))

    def _register_mlp_full(
        self,
        block: torch.nn.Module,
        layer: int,
    ) -> None:
        """Hook MLP module to capture both output and intermediate (post-activation)."""
        mlp = getattr(block, "mlp", None)
        if not isinstance(mlp, torch.nn.Module):
            self._register_if_present(
                block,
                "mlp",
                f"layers.{layer}.mlp_output",
                "mlp_output",
                layer=layer,
            )
            return

        # Hook the MLP output (c_proj output)
        self._register(
            mlp,
            f"layers.{layer}.mlp_output",
            "mlp_output",
            layer=layer,
        )

        # Hook the activation function to capture intermediate state
        act = getattr(mlp, "act", None)
        if isinstance(act, torch.nn.Module):
            def act_hook(
                hooked_module: torch.nn.Module,
                inputs: tuple[object, ...],
                output: object,
            ) -> None:
                del hooked_module, inputs
                tensor = self._first_tensor(output)
                if tensor is not None:
                    self._cache.store(
                        f"layers.{layer}.mlp_intermediate",
                        tensor,
                        kind="mlp_intermediate",
                        layer=layer,
                    )

            self._handles.append(act.register_forward_hook(act_hook))

    @classmethod
    def _first_tensor(cls, value: object) -> Optional[torch.Tensor]:
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = cls._first_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, dict):
            for item in value.values():
                tensor = cls._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None
