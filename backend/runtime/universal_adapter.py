"""Universal Transformer Architecture Adapter for Multi-Model Scientific Runtimes.

Discovers topology, extracts components (embeddings, layer stacks, final norms,
lm_heads), and performs precision conversion (FP32/FP16/BF16/INT8) across GPT-2,
Llama, Mistral, Qwen, Gemma, and Pythia architectures.

Also serves as the single entry point for runtime creation via ``create_runtime()``,
abstracting over in-memory, out-of-core, and adaptive residency strategies.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from .interfaces import ModelArchitectureTopology, PrecisionProfile

logger = logging.getLogger("MECH.runtime.universal_adapter")


class UniversalModelAdapter:
    """Introspects and abstracts diverse transformer model architectures for out-of-core streaming."""

    def __init__(self, model: Optional[PreTrainedModel] = None, config: Optional[PretrainedConfig] = None) -> None:
        self.model = model
        self.config = config or (model.config if model is not None else None)
        if self.config is None:
            raise ValueError("Either model or config must be provided to UniversalModelAdapter.")

        self.topology = self._detect_topology()

    def _detect_topology(self) -> ModelArchitectureTopology:
        """Detects the architectural family, dimension, layer count, and normalization type."""
        arch_str = ""
        if hasattr(self.config, "architectures") and self.config.architectures:
            arch_str = self.config.architectures[0].lower()
        model_type = getattr(self.config, "model_type", "").lower()

        # 1. Family detection
        if "llama" in arch_str or "llama" in model_type:
            family = "llama"
            norm_type = "rmsnorm"
            has_rotary = True
            has_abs_pos = False
            act_fn = "silu"
        elif "mistral" in arch_str or "mistral" in model_type:
            family = "mistral"
            norm_type = "rmsnorm"
            has_rotary = True
            has_abs_pos = False
            act_fn = "silu"
        elif "qwen" in arch_str or "qwen" in model_type:
            family = "qwen"
            norm_type = "rmsnorm"
            has_rotary = True
            has_abs_pos = False
            act_fn = "silu"
        elif "gemma" in arch_str or "gemma" in model_type:
            family = "gemma"
            norm_type = "rmsnorm"
            has_rotary = True
            has_abs_pos = False
            act_fn = "gelu"
        elif "neox" in arch_str or "pythia" in arch_str or "neox" in model_type:
            family = "pythia"
            norm_type = "layernorm"
            has_rotary = True
            has_abs_pos = False
            act_fn = "gelu"
        else:
            family = "gpt2"
            norm_type = "layernorm"
            has_rotary = False
            has_abs_pos = True
            act_fn = "gelu"

        # 2. Dimensions
        num_layers = getattr(self.config, "n_layer", getattr(self.config, "num_hidden_layers", 12))
        hidden_dim = getattr(self.config, "n_embd", getattr(self.config, "hidden_size", 768))
        num_heads = getattr(self.config, "n_head", getattr(self.config, "num_attention_heads", 12))
        vocab_size = getattr(self.config, "vocab_size", 50257)

        return ModelArchitectureTopology(
            architecture_family=family,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            norm_type=norm_type,
            has_rotary_embeddings=has_rotary,
            has_absolute_pos_embeddings=has_abs_pos,
            mlp_activation_fn=act_fn,
        )

    def get_embedding_modules(self, model: PreTrainedModel) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module]]:
        """Returns (word_embeddings, position_embeddings_or_none, drop_or_none)."""
        if hasattr(model, "transformer"):
            # GPT-2 style
            wte = getattr(model.transformer, "wte", None)
            wpe = getattr(model.transformer, "wpe", None)
            drop = getattr(model.transformer, "drop", None)
            return wte, wpe, drop
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            # Llama / Mistral / Qwen / Gemma style
            embed = model.model.embed_tokens
            return embed, None, None
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "embed_in"):
            # GPT-NeoX / Pythia style
            embed = model.gpt_neox.embed_in
            return embed, None, None
        else:
            # Generic fallback: search for embedding module
            for name, module in model.named_modules():
                if isinstance(module, nn.Embedding):
                    return module, None, None
            raise ValueError("Could not extract embedding module from model.")

    def get_layer_stack(self, model: PreTrainedModel) -> List[nn.Module]:
        """Returns the ordered list of transformer layer blocks."""
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return list(model.gpt_neox.layers)
        else:
            # Generic fallback: find ModuleList with layer count
            for module in model.modules():
                if isinstance(module, nn.ModuleList) and len(module) == self.topology.num_layers:
                    return list(module)
            raise ValueError("Could not extract transformer layer blocks from model.")

    def get_final_norm(self, model: PreTrainedModel) -> nn.Module:
        """Returns the final LayerNorm or RMSNorm module."""
        if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            return model.transformer.ln_f
        elif hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
            return model.gpt_neox.final_layer_norm
        else:
            for name, module in model.named_modules():
                if "norm" in name.lower() or "ln_f" in name.lower():
                    return module
            # If none, return Identity
            return nn.Identity()

    def get_lm_head(self, model: PreTrainedModel) -> nn.Module:
        """Returns the output projection head."""
        if hasattr(model, "lm_head"):
            return model.lm_head
        elif hasattr(model, "embed_out"):
            return model.embed_out
        else:
            for name, module in model.named_children():
                if isinstance(module, nn.Linear) and module.out_features == self.topology.vocab_size:
                    return module
            raise ValueError("Could not extract lm_head from model.")

    @staticmethod
    def apply_precision(
        module: nn.Module,
        precision: PrecisionProfile,
        target_device: Optional[torch.device | str] = None,
    ) -> nn.Module:
        """Applies precision casting or dynamic quantization to a layer block."""
        dev = target_device or "cpu"

        # 1. Quantization scheme
        if precision.quantization_scheme == "dynamic_int8" and str(dev) == "cpu":
            try:
                # Dynamic quantization of linear layers
                quantized = torch.ao.quantization.quantize_dynamic(
                    module.cpu(),
                    {nn.Linear},
                    dtype=torch.qint8,
                )
                return quantized
            except (RuntimeError, TypeError, ValueError) as exc:
                # Fallback to standard float casting if dynamic quantization is unsupported for module
                logger.debug("Dynamic quantization failed, falling back to float cast: %s", exc)

        # 2. Dtype casting
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if str(dev) == "cpu" and precision.weight_dtype == "float16":
            # PyTorch CPU lacks native FP16 LayerNorm kernels; use BFloat16 on CPU for 16-bit precision
            target_dtype = torch.bfloat16
        else:
            target_dtype = dtype_map.get(precision.weight_dtype, torch.float32)

        # Cast to device and dtype
        try:
            return module.to(device=dev, dtype=target_dtype)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Precision cast failed, casting device only: %s", exc)
            return module.to(device=dev)

    @classmethod
    def create_runtime(
        cls,
        model_id: str = "gpt2",
        runtime_type: str = "auto",
        device: Optional[str] = None,
        precision: Optional[PrecisionProfile] = None,
        vram_budget_mb: float = 2048.0,
        ram_budget_mb: float = 8192.0,
        max_active_layers: int = 3,
        cache_dir: Optional["Path | str"] = None,
    ) -> "ModelRuntimeInterface":
        """Factory: creates the appropriate runtime backend for the given model.

        This is the single entry point for runtime creation across the platform.
        Callers should use ``UniversalModelAdapter.create_runtime(...)`` instead
        of importing concrete runtime classes directly.

        Args:
            model_id: HuggingFace model identifier (e.g. "gpt2", "EleutherAI/pythia-1b").
            runtime_type: "auto" | "in_memory" | "out_of_core" | "adaptive".
                - "auto": selects in_memory if sufficient RAM, out_of_core otherwise.
            device: Compute device ("cpu" or "cuda:N"). Auto-detected if None.
            precision: Precision profile; defaults to FP32.
            vram_budget_mb: GPU VRAM budget for out_of_core runtime.
            ram_budget_mb: System RAM budget for out_of_core runtime.
            max_active_layers: Max layers resident simultaneously in out_of_core.
            cache_dir: Disk cache directory for out_of_core runtime.

        Returns:
            A ModelRuntimeInterface instance.
        """
        from .interfaces import ModelRuntimeInterface  # noqa: F811
        import importlib

        if runtime_type == "auto":
            try:
                import psutil

                available_gb = psutil.virtual_memory().available / (1024**3)
                config = AutoConfig.from_pretrained(model_id)
                num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", 12))
                est_model_gb = (num_layers * 300_000_000) / (1024**3)  # rough estimate
                runtime_type = "in_memory" if available_gb > est_model_gb * 3 else "out_of_core"
            except (ImportError, OSError, ValueError) as exc:
                logger.debug("Auto runtime detection failed, defaulting to in_memory: %s", exc)
                runtime_type = "in_memory"

        module_map = {
            "in_memory": (".in_memory_runtime", "InMemoryRuntime"),
            "out_of_core": (".out_of_core_runtime", "OutOfCoreRuntime"),
        }

        # Try adaptive runtime (may not exist in all deployments)
        if runtime_type == "adaptive":
            try:
                module = importlib.import_module("backend.runtime.adaptive")
                return module.AdaptiveRuntime(
                    model_id=model_id,
                    device=device,
                    vram_budget_mb=vram_budget_mb,
                    ram_budget_mb=ram_budget_mb,
                    precision=precision,
                )
            except (ImportError, AttributeError, TypeError) as exc:
                logger.debug("Adaptive runtime unavailable, falling back: %s", exc)
                runtime_type = "auto"

        if runtime_type in module_map:
            mod_path, class_name = module_map[runtime_type]
            module = importlib.import_module(mod_path, package="backend.runtime")
            runtime_cls = getattr(module, class_name)
        else:
            # Fallback: try auto-detection
            module = importlib.import_module(".in_memory_runtime", package="backend.runtime")
            runtime_cls = getattr(module, "InMemoryRuntime")

        runtime_kwargs: Dict[str, Any] = dict(model_id=model_id)
        if device is not None:
            runtime_kwargs["device"] = device
        if precision is not None:
            runtime_kwargs["precision"] = precision
        if runtime_type == "out_of_core":
            runtime_kwargs["vram_budget_mb"] = vram_budget_mb
            runtime_kwargs["ram_budget_mb"] = ram_budget_mb
            runtime_kwargs["max_active_layers"] = max_active_layers
            if cache_dir is not None:
                runtime_kwargs["cache_dir"] = cache_dir

        return runtime_cls(**runtime_kwargs)

    @classmethod
    def get_runtime(cls, model_id: str = "gpt2", **kwargs: Any) -> "ModelRuntimeInterface":
        """Convenience alias for :meth:`create_runtime` with auto-selection."""
        return cls.create_runtime(model_id=model_id, runtime_type="auto", **kwargs)

