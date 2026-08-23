"""Model-Agnostic ModelAdapter Subsystem for MECH Platform.

Decouples mechanistic interpretability, hook attachment, activation capture,
and causal intervention from specific model architectures.

Supported Architectures:
- GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2)
- Generic CausalLM (LLaMA, Mistral, Qwen, Gemma, Pythia, OPT, GPT-NeoX)
"""

from __future__ import annotations

import abc
import hashlib
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("MECH.model_adapter")


class ModelAdapter(abc.ABC):
    """Abstract base class for model-agnostic mechanistic interpretability adapters."""

    def __init__(self, model_id: str, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> None:
        self.model_id = model_id
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype or torch.float32
        self._model: Optional[nn.Module] = None
        self._tokenizer: Optional[Any] = None
        self._model_hash: Optional[str] = None
        self._lock = threading.Lock()

    @abc.abstractmethod
    def load(self) -> None:
        """Loads model weights and tokenizer into memory."""
        pass

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self.load()
        return self._model  # type: ignore

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    @property
    @abc.abstractmethod
    def n_layers(self) -> int:
        """Number of transformer layers."""
        pass

    @property
    @abc.abstractmethod
    def n_heads(self) -> int:
        """Number of attention heads per layer."""
        pass

    @property
    @abc.abstractmethod
    def d_model(self) -> int:
        """Hidden embedding dimension."""
        pass

    @property
    @abc.abstractmethod
    def d_mlp(self) -> int:
        """MLP intermediate dimension."""
        pass

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size."""
        pass

    @property
    def parameter_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def model_hash(self) -> str:
        """Deterministic SHA-256 fingerprint of the model architecture and configuration."""
        if self._model_hash is None:
            meta_str = f"{self.model_id}:{self.n_layers}:{self.n_heads}:{self.d_model}:{self.d_mlp}:{self.vocab_size}:{self.parameter_count}"
            self._model_hash = hashlib.sha256(meta_str.encode()).hexdigest()
        return self._model_hash

    @abc.abstractmethod
    def get_layer_block(self, layer: int) -> nn.Module:
        """Returns the transformer layer block module."""
        pass

    @abc.abstractmethod
    def get_attention_module(self, layer: int) -> nn.Module:
        """Returns the attention sub-module for a layer."""
        pass

    @abc.abstractmethod
    def get_mlp_fc_module(self, layer: int) -> nn.Module:
        """Returns the first MLP expansion module (e.g. c_fc or gate_proj/up_proj)."""
        pass

    @abc.abstractmethod
    def get_mlp_proj_module(self, layer: int) -> nn.Module:
        """Returns the second MLP projection module (e.g. c_proj or down_proj)."""
        pass

    @abc.abstractmethod
    def get_unembedding_weight(self) -> torch.Tensor:
        """Returns the unembedding matrix W_U with shape [vocab_size, d_model]."""
        pass

    @abc.abstractmethod
    def reshape_attention_output(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Reshapes attention output tensor to [batch, seq_len, n_heads, head_dim]."""
        pass

    @abc.abstractmethod
    def flatten_attention_output(self, head_tensor: torch.Tensor) -> torch.Tensor:
        """Flattens head tensor [batch, seq_len, n_heads, head_dim] back to [batch, seq_len, d_model]."""
        pass

    def encode(self, text: str) -> List[int]:
        """Encodes text to a list of token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back to a string."""
        return self.tokenizer.decode(token_ids)

    def forward(
        self,
        prompt: str,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
    ) -> Any:
        """Executes a standard forward pass on prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            return self.model(
                **inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

    def get_architecture_info(self) -> Dict[str, Any]:
        """Returns introspected architecture dictionary."""
        return {
            "model_id": self.model_id,
            "architecture_class": self.__class__.__name__,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "d_mlp": self.d_mlp,
            "d_head": self.d_head,
            "vocab_size": self.vocab_size,
            "parameter_count": self.parameter_count,
            "model_hash": self.model_hash,
            "device": str(self._device),
            "dtype": str(self._dtype),
        }


class GPT2Adapter(ModelAdapter):
    """Adapter for GPT-2 family models (GPT-2, GPT-2 Medium, GPT-2 Large, GPT-2 XL, DistilGPT2)."""

    def load(self) -> None:
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            logger.info("Loading GPT2Adapter weights for '%s'...", self.model_id)
            config = AutoConfig.from_pretrained(
                self.model_id,
                attn_implementation="eager",
                output_attentions=True,
                output_hidden_states=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                config=config,
            )
            self._model.to(self._device)
            self._model.eval()

    @property
    def n_layers(self) -> int:
        return len(self.model.transformer.h)

    @property
    def n_heads(self) -> int:
        return self.model.config.n_head

    @property
    def d_model(self) -> int:
        return self.model.config.n_embd

    @property
    def d_mlp(self) -> int:
        return self.model.transformer.h[0].mlp.c_fc.weight.shape[1]

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    def get_layer_block(self, layer: int) -> nn.Module:
        return self.model.transformer.h[layer]

    def get_attention_module(self, layer: int) -> nn.Module:
        return self.model.transformer.h[layer].attn

    def get_mlp_fc_module(self, layer: int) -> nn.Module:
        return self.model.transformer.h[layer].mlp.c_fc

    def get_mlp_proj_module(self, layer: int) -> nn.Module:
        return self.model.transformer.h[layer].mlp.c_proj

    def get_unembedding_weight(self) -> torch.Tensor:
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None:
            return lm_head.weight.data
        return self.model.transformer.wte.weight.data

    def reshape_attention_output(self, attn_out: torch.Tensor) -> torch.Tensor:
        b, s, d = attn_out.shape
        return attn_out.view(b, s, self.n_heads, self.d_head)

    def flatten_attention_output(self, head_tensor: torch.Tensor) -> torch.Tensor:
        b, s, h, d = head_tensor.shape
        return head_tensor.reshape(b, s, h * d)


class GenericCausalLMAdapter(ModelAdapter):
    """Generic adapter for HuggingFace AutoModelForCausalLM (LLaMA, Mistral, Qwen, Gemma, Pythia, OPT)."""

    def load(self) -> None:
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            logger.info("Loading GenericCausalLMAdapter weights for '%s'...", self.model_id)
            config = AutoConfig.from_pretrained(
                self.model_id,
                attn_implementation="eager",
                output_attentions=True,
                output_hidden_states=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                config=config,
            )
            self._model.to(self._device)
            self._model.eval()

    def _get_layers_container(self) -> Any:
        m = self.model
        for attr in ("model.layers", "transformer.h", "transformer.layers", "gpt_neox.layers", "decoder.layers"):
            curr = m
            valid = True
            for part in attr.split("."):
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    valid = False
                    break
            if valid and hasattr(curr, "__len__"):
                return curr
        raise AttributeError(f"Could not automatically locate transformer layers in model {self.model_id}")

    @property
    def n_layers(self) -> int:
        cfg = getattr(self.model, "config", None)
        if cfg:
            for k in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
                if hasattr(cfg, k):
                    return getattr(cfg, k)
        return len(self._get_layers_container())

    @property
    def n_heads(self) -> int:
        cfg = getattr(self.model, "config", None)
        if cfg:
            for k in ("num_attention_heads", "n_head", "num_heads", "n_heads"):
                if hasattr(cfg, k):
                    return getattr(cfg, k)
        return 12

    @property
    def d_model(self) -> int:
        cfg = getattr(self.model, "config", None)
        if cfg:
            for k in ("hidden_size", "n_embd", "d_model"):
                if hasattr(cfg, k):
                    return getattr(cfg, k)
        return 768

    @property
    def d_mlp(self) -> int:
        cfg = getattr(self.model, "config", None)
        if cfg:
            for k in ("intermediate_size", "d_mlp", "n_inner"):
                if hasattr(cfg, k):
                    val = getattr(cfg, k)
                    if val is not None:
                        return val
        block = self.get_layer_block(0)
        mlp = getattr(block, "mlp", getattr(block, "feed_forward", None))
        if mlp:
            for mod in mlp.modules():
                if isinstance(mod, nn.Linear):
                    return max(mod.in_features, mod.out_features)
        return self.d_model * 4

    @property
    def vocab_size(self) -> int:
        cfg = getattr(self.model, "config", None)
        if cfg and hasattr(cfg, "vocab_size"):
            return cfg.vocab_size
        return len(self.tokenizer)

    def get_layer_block(self, layer: int) -> nn.Module:
        layers = self._get_layers_container()
        return layers[layer]

    def get_attention_module(self, layer: int) -> nn.Module:
        block = self.get_layer_block(layer)
        for attr in ("self_attn", "attn", "attention", "self_attention"):
            if hasattr(block, attr):
                return getattr(block, attr)
        raise AttributeError(f"Could not locate attention module in layer {layer}")

    def get_mlp_fc_module(self, layer: int) -> nn.Module:
        block = self.get_layer_block(layer)
        mlp = getattr(block, "mlp", getattr(block, "feed_forward", None))
        if mlp is None:
            raise AttributeError(f"Could not locate MLP module in layer {layer}")
        for attr in ("c_fc", "gate_proj", "up_proj", "fc_in", "dense_h_to_4h", "w1"):
            if hasattr(mlp, attr):
                return getattr(mlp, attr)
        # Fallback to first linear
        for mod in mlp.modules():
            if isinstance(mod, nn.Linear):
                return mod
        return mlp

    def get_mlp_proj_module(self, layer: int) -> nn.Module:
        block = self.get_layer_block(layer)
        mlp = getattr(block, "mlp", getattr(block, "feed_forward", None))
        if mlp is None:
            raise AttributeError(f"Could not locate MLP module in layer {layer}")
        for attr in ("c_proj", "down_proj", "fc_out", "dense_4h_to_h", "w2"):
            if hasattr(mlp, attr):
                return getattr(mlp, attr)
        # Fallback to last linear
        linears = [m for m in mlp.modules() if isinstance(mod, nn.Linear)]
        return linears[-1] if linears else mlp

    def get_unembedding_weight(self) -> torch.Tensor:
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            return lm_head.weight.data
        for mod in self.model.modules():
            if isinstance(mod, nn.Linear) and mod.out_features == self.vocab_size:
                return mod.weight.data
        # Embedding transpose fallback
        for mod in self.model.modules():
            if isinstance(mod, nn.Embedding) and mod.num_embeddings == self.vocab_size:
                return mod.weight.data
        raise AttributeError("Could not locate unembedding matrix W_U.")

    def reshape_attention_output(self, attn_out: torch.Tensor) -> torch.Tensor:
        b, s, d = attn_out.shape
        return attn_out.view(b, s, self.n_heads, self.d_head)

    def flatten_attention_output(self, head_tensor: torch.Tensor) -> torch.Tensor:
        b, s, h, d = head_tensor.shape
        return head_tensor.reshape(b, s, h * d)


_ADAPTER_CACHE: Dict[str, ModelAdapter] = {}
_CACHE_LOCK = threading.Lock()


def get_model_adapter(model_id: str = "gpt2", device: Optional[str] = None) -> ModelAdapter:
    """Factory creating or retrieving a cached model-agnostic adapter."""
    with _CACHE_LOCK:
        if model_id in _ADAPTER_CACHE:
            return _ADAPTER_CACHE[model_id]

        mid = model_id.lower().strip()
        if "gpt2" in mid or "distilgpt2" in mid:
            adapter = GPT2Adapter(model_id=model_id, device=device)
        else:
            adapter = GenericCausalLMAdapter(model_id=model_id, device=device)

        _ADAPTER_CACHE[model_id] = adapter
        return adapter
