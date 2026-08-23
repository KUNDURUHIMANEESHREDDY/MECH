"""Model-Agnostic Architecture Introspection & Execution Engine.

Provides a completely model-agnostic execution specification (ExecutionSpec)
with polymorphic PositionStrategy and AttentionStrategy, eliminating all
model-specific branching (e.g. GPT-2, OPT, Llama, Qwen, Gemma, Mistral, Phi).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

try:
    from transformers.masking_utils import create_causal_mask
except ImportError:
    create_causal_mask = None

logger = logging.getLogger("MECH.model_introspector")


class PositionStrategy(Enum):
    """Strategy for computing and injecting positional information."""
    ABSOLUTE_LEARNED = auto()  # Explicit position embedding module (e.g. GPT-2 wpe, OPT embed_positions)
    ROTARY_ROPE = auto()       # Rotary position embeddings (e.g. Llama, Qwen, Mistral, Gemma, Phi)
    RELATIVE_BIAS = auto()     # Relative attention bias / ALiBi
    NONE = auto()


class AttentionStrategy(Enum):
    """Strategy for preparing attention masks."""
    CAUSAL_MASK = auto()       # Standard triangular causal attention mask
    SLIDING_WINDOW = auto()    # Local sliding window attention mask
    NONE = auto()


@dataclass
class ExecutionSpec:
    """Universal execution specification for an arbitrary transformer model."""
    model_id: str
    architecture_name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    vocab_size: int
    
    # Submodules
    input_embeddings: Optional[nn.Module] = None
    positional_embeddings: Optional[nn.Module] = None
    rotary_module: Optional[nn.Module] = None
    input_dropout: Optional[nn.Module] = None
    layer_stack: Optional[nn.ModuleList] = None
    layer_block_cls: Optional[Type[nn.Module]] = None
    final_norm: Optional[nn.Module] = None
    output_head: Optional[nn.Module] = None
    
    # Advanced Topology Metadata (MoE, GQA, Encoder-Decoder)
    is_moe: bool = False
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    is_encoder_decoder: bool = False
    encoder_layer_stack: Optional[nn.ModuleList] = None
    decoder_layer_stack: Optional[nn.ModuleList] = None
    
    # Execution Strategies
    position_strategy: PositionStrategy = PositionStrategy.ROTARY_ROPE
    attention_strategy: AttentionStrategy = AttentionStrategy.CAUSAL_MASK
    forward_parameter_names: List[str] = field(default_factory=list)
    model_config: Optional[Any] = None

    def prepare_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        """Prepares initial hidden states, attention masks, and position IDs polymorphically."""
        batch_size, seq_len = input_ids.shape
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 1. Input token embeddings
        if self.input_embeddings is not None:
            h = self.input_embeddings(input_ids)
        else:
            h = torch.randn(batch_size, seq_len, self.hidden_size, device=device, dtype=dtype)

        # 2. Position Injection based on PositionStrategy
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        extra_kwargs: Dict[str, Any] = {}

        if self.position_strategy == PositionStrategy.ABSOLUTE_LEARNED and self.positional_embeddings is not None:
            if hasattr(self.positional_embeddings, "forward"):
                sig = inspect.signature(self.positional_embeddings.forward)
                if "attention_mask" in sig.parameters:
                    pos = self.positional_embeddings(attention_mask, 0)
                    h = h + pos
                else:
                    h = h + self.positional_embeddings(position_ids)
            else:
                h = h + self.positional_embeddings(position_ids)
        elif self.position_strategy == PositionStrategy.ROTARY_ROPE:
            if self.rotary_module is not None and "position_embeddings" in self.forward_parameter_names:
                try:
                    extra_kwargs["position_embeddings"] = self.rotary_module(h, position_ids)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Swallowed exception: %s", exc)

        # 3. Input dropout if present
        if self.input_dropout is not None:
            h = self.input_dropout(h)

        h = h.to(device=device, dtype=dtype)

        # 4. Attention Mask based on AttentionStrategy
        causal_mask = None
        if self.attention_strategy == AttentionStrategy.CAUSAL_MASK and self.model_config is not None and create_causal_mask is not None:
            try:
                causal_mask = create_causal_mask(
                    config=self.model_config,
                    inputs_embeds=h,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    position_ids=position_ids,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Swallowed exception: %s", exc)
                causal_mask = None

        return h, causal_mask, position_ids, extra_kwargs

    def execute_layer(
        self,
        layer_module: nn.Module,
        hidden_states: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **extra_kwargs: Any,
    ) -> torch.Tensor:
        """Executes a single layer block forward pass using dynamic signature matching."""
        sig_params = self.forward_parameter_names
        if not sig_params:
            try:
                sig_params = list(inspect.signature(layer_module.forward).parameters.keys())
            except Exception as exc:  # noqa: BLE001
                logger.debug("Swallowed exception: %s", exc)
                sig_params = ["hidden_states"]

        call_kwargs: Dict[str, Any] = {}
        if "position_ids" in sig_params and position_ids is not None:
            call_kwargs["position_ids"] = position_ids
        if "attention_mask" in sig_params and causal_mask is not None:
            call_kwargs["attention_mask"] = causal_mask
        for k, v in extra_kwargs.items():
            if k in sig_params:
                call_kwargs[k] = v

        try:
            out = layer_module(hidden_states, **call_kwargs)
        except TypeError:
            try:
                out = layer_module(hidden_states, attention_mask=causal_mask)
            except TypeError:
                try:
                    out = layer_module(hidden_states, position_ids=position_ids)
                except TypeError:
                    out = layer_module(hidden_states)

        return out[0] if isinstance(out, tuple) else out

    def finalize(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Applies final normalization and output linear projection to obtain logits."""
        h = hidden_states
        if self.final_norm is not None:
            h = self.final_norm(h)
        if self.output_head is not None:
            logits = self.output_head(h)
        else:
            logits = h
        return logits


class ModelIntrospector:
    """Discovers architecture components and builds an ExecutionSpec."""

    @staticmethod
    def introspect(model: nn.Module) -> ExecutionSpec:
        """Dynamically introspects a PyTorch transformer model and creates an ExecutionSpec."""
        config = getattr(model, "config", None)
        model_id = getattr(config, "_name_or_path", model.__class__.__name__)
        arch_name = getattr(config, "model_type", model.__class__.__name__)

        # 1. Base model resolution
        base = getattr(model, "base_model", model)
        if base is None:
            base = model

        # 2. Input embeddings discovery
        input_embed = None
        if hasattr(model, "get_input_embeddings"):
            input_embed = model.get_input_embeddings()
        if input_embed is None and hasattr(base, "get_input_embeddings"):
            input_embed = base.get_input_embeddings()
        if input_embed is None:
            embeddings = [m for m in model.modules() if isinstance(m, nn.Embedding)]
            if embeddings:
                input_embed = embeddings[0]

        # 3. Layer stack (nn.ModuleList) discovery
        module_lists = [m for m in base.modules() if isinstance(m, nn.ModuleList) and len(m) > 1]
        if not module_lists:
            module_lists = [m for m in model.modules() if isinstance(m, nn.ModuleList) and len(m) > 1]

        if not module_lists:
            raise ValueError(f"Could not discover transformer layer ModuleList in {model.__class__.__name__}")

        layer_stack = max(module_lists, key=len)
        num_layers = len(layer_stack)
        first_block = layer_stack[0]
        layer_block_cls = first_block.__class__

        # Parameter signature for the layer forward
        try:
            sig = inspect.signature(first_block.forward)
            forward_params = list(sig.parameters.keys())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Swallowed exception: %s", exc)
            forward_params = ["hidden_states"]

        # 4. Final normalization discovery
        layer_submodules = set(layer_stack.modules())
        norm_candidates = [
            m for m in base.modules()
            if ("norm" in m.__class__.__name__.lower() or "ln" in m.__class__.__name__.lower())
            and m not in layer_submodules
        ]
        if not norm_candidates:
            norm_candidates = [
                m for m in model.modules()
                if ("norm" in m.__class__.__name__.lower() or "ln" in m.__class__.__name__.lower())
                and m not in layer_submodules
            ]
        final_norm = norm_candidates[-1] if norm_candidates else None

        # 5. Output head discovery
        output_head = None
        if hasattr(model, "get_output_embeddings"):
            output_head = model.get_output_embeddings()
        if output_head is None:
            linears = [
                m for m in model.modules()
                if isinstance(m, nn.Linear) and m not in layer_submodules
            ]
            if linears:
                output_head = linears[-1]

        # 6. Positional Strategy & Positional Embeddings discovery
        pos_embed = None
        rotary_module = None

        for m in base.modules():
            if m is not input_embed and isinstance(m, nn.Embedding) and m not in layer_submodules:
                pos_embed = m
                break
        
        # Check submodules for embed_positions / wpe / rotary
        if pos_embed is None:
            for name, m in base.named_modules():
                if any(term in name.lower() for term in ["embed_positions", "wpe", "position_embeddings"]) and m not in layer_submodules:
                    pos_embed = m
                    break

        for sm in base.modules():
            if "rotary" in sm.__class__.__name__.lower():
                rotary_module = sm
                break

        if pos_embed is not None:
            pos_strategy = PositionStrategy.ABSOLUTE_LEARNED
        elif rotary_module is not None:
            pos_strategy = PositionStrategy.ROTARY_ROPE
        else:
            pos_strategy = PositionStrategy.ROTARY_ROPE

        # 7. Input dropout module if present
        input_dropout = None
        for m in base.children():
            if isinstance(m, nn.Dropout):
                input_dropout = m
                break

        # 8. Model dimension & attention metadata (GQA / MQA / MoE)
        hidden_size = getattr(
            config, "hidden_size", getattr(config, "n_embd", getattr(config, "d_model", 768))
        )
        num_heads = getattr(
            config, "num_attention_heads", getattr(config, "n_head", getattr(config, "num_heads", 12))
        )
        num_kv_heads = getattr(
            config, "num_key_value_heads", getattr(config, "num_kv_heads", num_heads)
        )
        vocab_size = getattr(config, "vocab_size", 32000)

        # 9. Mixture of Experts (MoE) topology detection
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", getattr(config, "n_routed_experts", None)))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", getattr(config, "top_k", None))
        is_moe = num_experts is not None and num_experts > 1

        # 10. Encoder-Decoder topology detection
        is_enc_dec = getattr(config, "is_encoder_decoder", False)
        enc_stack = None
        dec_stack = layer_stack
        if is_enc_dec and hasattr(model, "encoder") and hasattr(model, "decoder"):
            enc_lists = [m for m in model.encoder.modules() if isinstance(m, nn.ModuleList) and len(m) > 1]
            dec_lists = [m for m in model.decoder.modules() if isinstance(m, nn.ModuleList) and len(m) > 1]
            if enc_lists:
                enc_stack = max(enc_lists, key=len)
            if dec_lists:
                dec_stack = max(dec_lists, key=len)

        return ExecutionSpec(
            model_id=model_id,
            architecture_name=arch_name,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            vocab_size=vocab_size,
            input_embeddings=input_embed,
            positional_embeddings=pos_embed,
            rotary_module=rotary_module,
            input_dropout=input_dropout,
            layer_stack=layer_stack,
            layer_block_cls=layer_block_cls,
            final_norm=final_norm,
            output_head=output_head,
            is_moe=is_moe,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            num_key_value_heads=num_kv_heads,
            is_encoder_decoder=is_enc_dec,
            encoder_layer_stack=enc_stack,
            decoder_layer_stack=dec_stack,
            position_strategy=pos_strategy,
            attention_strategy=AttentionStrategy.CAUSAL_MASK,
            forward_parameter_names=forward_params,
            model_config=config,
        )


# Backward-compatible alias
ArchitectureIntrospection = ExecutionSpec
