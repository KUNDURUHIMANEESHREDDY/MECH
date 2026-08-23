"""Disk Weight Store — on-demand per-layer NVMe weight streaming and sharding.

Enables models larger than available RAM/VRAM to reside on NVMe SSD as sharded
layer parameters, streaming single layers into compute memory on-demand and
freeing them immediately after layer execution.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    import safetensors.torch
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False

logger = logging.getLogger("MECH.disk_weight_store")

DEFAULT_WEIGHTS_DIR = Path("backend/storage/weights")


@dataclass
class LayerWeightMetadata:
    """Metadata describing a sharded layer on disk."""
    layer_idx: int
    component: str  # "embedding", "block", "ln_f", "lm_head"
    file_path: str
    num_parameters: int
    size_bytes: int
    dtype: str
    tensor_names: List[str] = field(default_factory=list)


@dataclass
class ShardedModelManifest:
    """Manifest describing a sharded disk-resident model."""
    model_id: str
    architecture: str
    num_layers: int
    hidden_size: int
    num_heads: int
    vocab_size: int
    total_parameters: int
    total_size_bytes: int
    layers: Dict[int, LayerWeightMetadata] = field(default_factory=dict)
    special_components: Dict[str, LayerWeightMetadata] = field(default_factory=dict)


class DiskWeightStore:
    """Manages disk-resident model layer weights and provides on-demand paging."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_WEIGHTS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_model_dir(self, model_id: str) -> Path:
        """Returns the storage path for a model's sharded weights."""
        sanitized = model_id.replace("/", "--").replace(" ", "_")
        path = self.base_dir / sanitized
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_layer_weights(
        self,
        model_id: str,
        layer_idx: int,
        state_dict: Dict[str, torch.Tensor],
        component: str = "block",
    ) -> LayerWeightMetadata:
        """Saves a single layer's weights to disk in safetensors or PyTorch format."""
        model_dir = self.get_model_dir(model_id)
        file_name = f"{component}_{layer_idx}.safetensors" if _HAS_SAFETENSORS else f"{component}_{layer_idx}.pt"
        target_path = model_dir / file_name

        total_params = sum(t.numel() for t in state_dict.values())
        dtype_str = str(next(iter(state_dict.values())).dtype) if state_dict else "torch.float32"
        tensor_names = list(state_dict.keys())

        # Save weights
        if _HAS_SAFETENSORS:
            # Ensure tensors are contiguous and on CPU
            cpu_dict = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
            safetensors.torch.save_file(cpu_dict, str(target_path))
        else:
            cpu_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
            torch.save(cpu_dict, target_path)

        size_bytes = target_path.stat().st_size
        return LayerWeightMetadata(
            layer_idx=layer_idx,
            component=component,
            file_path=str(target_path),
            num_parameters=total_params,
            size_bytes=size_bytes,
            dtype=dtype_str,
            tensor_names=tensor_names,
        )

    def load_layer_weights(
        self,
        model_id: str,
        layer_idx: int,
        component: str = "block",
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Loads a single layer's weights from disk into active memory on target device."""
        model_dir = self.get_model_dir(model_id)
        st_path = model_dir / f"{component}_{layer_idx}.safetensors"
        pt_path = model_dir / f"{component}_{layer_idx}.pt"

        if st_path.exists() and _HAS_SAFETENSORS:
            # Memory-mapped zero-copy read
            tensors = safetensors.torch.load_file(str(st_path), device=device)
        elif pt_path.exists():
            tensors = torch.load(pt_path, map_location=device, weights_only=True)
        else:
            raise FileNotFoundError(f"No sharded weights found for {model_id} layer {layer_idx} ({component}) at {model_dir}")

        if dtype is not None:
            tensors = {k: v.to(dtype=dtype) for k, v in tensors.items()}
        return tensors

    def load_layer_into_block(
        self,
        model_id: str,
        layer_idx: int,
        block_module: nn.Module,
        device: str = "cpu",
        component: str = "block",
    ) -> None:
        """Paginates weights from disk directly into a module's parameters."""
        weights = self.load_layer_weights(model_id, layer_idx, component=component, device=device)
        
        # Load directly if keys match, otherwise clean any prefix dynamically
        clean_weights = {}
        for k, v in weights.items():
            clean_k = k
            # Strip any hierarchical prefix leading up to the layer index if present
            if f".{layer_idx}." in clean_k:
                clean_k = clean_k.split(f".{layer_idx}.", 1)[1]
            elif clean_k.startswith(f"{layer_idx}."):
                clean_k = clean_k.split(f"{layer_idx}.", 1)[1]
            clean_weights[clean_k] = v

        try:
            block_module.load_state_dict(clean_weights, strict=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Swallowed exception: %s", exc)
            block_module.load_state_dict(clean_weights, strict=False)
        block_module.to(device)

    def unload_layer(self, block_module: nn.Module) -> None:
        """Frees layer module weights from active memory/VRAM."""
        block_module.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def shard_existing_hf_model(
        self,
        model: nn.Module,
        model_id: str,
    ) -> ShardedModelManifest:
        """Shards an in-memory or loaded HuggingFace model into layer-wise disk files dynamically."""
        from .model_introspector import ModelIntrospector
        
        model_dir = self.get_model_dir(model_id)
        intro = ModelIntrospector.introspect(model)
        
        layers_meta: Dict[int, LayerWeightMetadata] = {}
        specials_meta: Dict[str, LayerWeightMetadata] = {}

        # 1. Shard input and positional embeddings
        embed_dict = {}
        if intro.input_embeddings is not None:
            embed_dict.update({f"input_embeddings.{k}": v.data for k, v in intro.input_embeddings.state_dict().items()})
        if intro.positional_embeddings is not None:
            embed_dict.update({f"positional_embeddings.{k}": v.data for k, v in intro.positional_embeddings.state_dict().items()})

        if embed_dict:
            specials_meta["embedding"] = self.save_layer_weights(
                model_id=model_id,
                layer_idx=0,
                state_dict=embed_dict,
                component="embedding",
            )

        # 2. Shard transformer blocks dynamically
        if intro.layer_stack is not None:
            for idx, block in enumerate(intro.layer_stack):
                block_state = {k: v.data for k, v in block.state_dict().items()}
                meta = self.save_layer_weights(
                    model_id=model_id,
                    layer_idx=idx,
                    state_dict=block_state,
                    component="block",
                )
                layers_meta[idx] = meta

        # 3. Shard final normalization
        if intro.final_norm is not None:
            norm_state = {k: v.data for k, v in intro.final_norm.state_dict().items()}
            specials_meta["norm"] = self.save_layer_weights(
                model_id=model_id,
                layer_idx=intro.num_layers,
                state_dict=norm_state,
                component="norm",
            )

        # 4. Shard output head
        if intro.output_head is not None:
            head_state = {k: v.data for k, v in intro.output_head.state_dict().items()}
            specials_meta["head"] = self.save_layer_weights(
                model_id=model_id,
                layer_idx=intro.num_layers + 1,
                state_dict=head_state,
                component="head",
            )

        total_bytes = sum(m.size_bytes for m in layers_meta.values()) + sum(m.size_bytes for m in specials_meta.values())
        total_params = sum(m.num_parameters for m in layers_meta.values()) + sum(m.num_parameters for m in specials_meta.values())

        manifest = ShardedModelManifest(
            model_id=model_id,
            architecture=intro.architecture_name,
            num_layers=intro.num_layers,
            hidden_size=intro.hidden_size,
            num_heads=intro.num_heads,
            vocab_size=intro.vocab_size,
            total_parameters=total_params,
            total_size_bytes=total_bytes,
            layers=layers_meta,
            special_components=specials_meta,
        )

        manifest_path = model_dir / "manifest.json"
        manifest_data = {
            "model_id": manifest.model_id,
            "architecture": manifest.architecture,
            "num_layers": manifest.num_layers,
            "hidden_size": manifest.hidden_size,
            "num_heads": manifest.num_heads,
            "vocab_size": manifest.vocab_size,
            "total_parameters": manifest.total_parameters,
            "total_size_bytes": manifest.total_size_bytes,
            "layers": {k: v.__dict__ for k, v in manifest.layers.items()},
            "special_components": {k: v.__dict__ for k, v in manifest.special_components.items()},
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return manifest

    def create_synthetic_scaled_sharded_model(
        self,
        model_id: str,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        vocab_size: int = 32000,
        dtype: torch.dtype = torch.float16,
    ) -> ShardedModelManifest:
        """Creates an on-disk sharded model of arbitrary scale (e.g. 1B, 3B, 7B, 13B, 70B)
        
        Writes each layer directly to NVMe disk without ever allocating the full model
        matrix in RAM, demonstrating true out-of-core scalability.
        """
        model_dir = self.get_model_dir(model_id)
        layers_meta: Dict[int, LayerWeightMetadata] = {}
        specials_meta: Dict[str, LayerWeightMetadata] = {}

        # 1. Embedding layer
        embed_weight = torch.randn(vocab_size, hidden_size, dtype=dtype)
        embed_meta = self.save_layer_weights(
            model_id=model_id,
            layer_idx=0,
            state_dict={"wte.weight": embed_weight},
            component="embedding",
        )
        specials_meta["embedding"] = embed_meta
        del embed_weight
        gc.collect()

        # 2. Sharded Transformer blocks written layer-by-layer to NVMe
        head_dim = hidden_size // num_heads
        intermediate_size = hidden_size * 4

        for layer_idx in range(num_layers):
            # Realistic parameter shapes for modern causal attention + MLP
            block_dict = {
                "attn.c_attn.weight": torch.randn(hidden_size, 3 * hidden_size, dtype=dtype),
                "attn.c_attn.bias": torch.zeros(3 * hidden_size, dtype=dtype),
                "attn.c_proj.weight": torch.randn(hidden_size, hidden_size, dtype=dtype),
                "attn.c_proj.bias": torch.zeros(hidden_size, dtype=dtype),
                "ln_1.weight": torch.ones(hidden_size, dtype=dtype),
                "ln_1.bias": torch.zeros(hidden_size, dtype=dtype),
                "mlp.c_fc.weight": torch.randn(hidden_size, intermediate_size, dtype=dtype),
                "mlp.c_fc.bias": torch.zeros(intermediate_size, dtype=dtype),
                "mlp.c_proj.weight": torch.randn(intermediate_size, hidden_size, dtype=dtype),
                "mlp.c_proj.bias": torch.zeros(hidden_size, dtype=dtype),
                "ln_2.weight": torch.ones(hidden_size, dtype=dtype),
                "ln_2.bias": torch.zeros(hidden_size, dtype=dtype),
            }

            meta = self.save_layer_weights(
                model_id=model_id,
                layer_idx=layer_idx,
                state_dict=block_dict,
                component="block",
            )
            layers_meta[layer_idx] = meta
            del block_dict
            gc.collect()

        # 3. Final norm and LM Head
        head_dict = {
            "ln_f.weight": torch.ones(hidden_size, dtype=dtype),
            "ln_f.bias": torch.zeros(hidden_size, dtype=dtype),
            "lm_head.weight": torch.randn(vocab_size, hidden_size, dtype=dtype),
        }
        head_meta = self.save_layer_weights(
            model_id=model_id,
            layer_idx=num_layers,
            state_dict=head_dict,
            component="head",
        )
        specials_meta["head"] = head_meta
        del head_dict
        gc.collect()

        total_bytes = sum(m.size_bytes for m in layers_meta.values()) + sum(m.size_bytes for m in specials_meta.values())
        total_params = sum(m.num_parameters for m in layers_meta.values()) + sum(m.num_parameters for m in specials_meta.values())

        manifest = ShardedModelManifest(
            model_id=model_id,
            architecture="gpt2_or_llama",
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            vocab_size=vocab_size,
            total_parameters=total_params,
            total_size_bytes=total_bytes,
            layers=layers_meta,
            special_components=specials_meta,
        )

        manifest_path = model_dir / "manifest.json"
        manifest_data = {
            "model_id": manifest.model_id,
            "architecture": manifest.architecture,
            "num_layers": manifest.num_layers,
            "hidden_size": manifest.hidden_size,
            "num_heads": manifest.num_heads,
            "vocab_size": manifest.vocab_size,
            "total_parameters": manifest.total_parameters,
            "total_size_bytes": manifest.total_size_bytes,
            "layers": {k: v.__dict__ for k, v in manifest.layers.items()},
            "special_components": {k: v.__dict__ for k, v in manifest.special_components.items()},
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return manifest

    def cleanup_model(self, model_id: str) -> None:
        """Removes sharded weights for a model from disk."""
        model_dir = self.get_model_dir(model_id)
        if model_dir.exists():
            shutil.rmtree(model_dir)


_GLOBAL_WEIGHT_STORE: Optional[DiskWeightStore] = None


def get_disk_weight_store() -> DiskWeightStore:
    """Returns the process singleton DiskWeightStore."""
    global _GLOBAL_WEIGHT_STORE
    if _GLOBAL_WEIGHT_STORE is None:
        _GLOBAL_WEIGHT_STORE = DiskWeightStore()
    return _GLOBAL_WEIGHT_STORE
