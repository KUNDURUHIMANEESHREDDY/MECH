"""Layer Pager & Progressive Sequential Execution Engine.

Enables layer-by-layer progressive forward execution, paging transformer layer
weights into active compute memory on-demand and offloading immediately to minimize VRAM/RAM.
Integrates with Content-Addressed Storage (CAS) for instant sub-graph cache reuse.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil
import torch
import torch.nn as nn

try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block, create_causal_mask
except ImportError:
    try:
        from transformers.masking_utils import create_causal_mask
    except ImportError:
        create_causal_mask = None
    GPT2Block = None

from ..artifacts.cas_store import ArtifactStore, compute_artifact_key, get_artifact_store
from ..artifacts.models import ArtifactMetadata, ExecutionArtifact, Provenance
from .disk_weight_store import DiskWeightStore, ShardedModelManifest, get_disk_weight_store

from .model_introspector import ArchitectureIntrospection, ModelIntrospector

logger = logging.getLogger("MECH.layer_pager")


@dataclass
class LayerExecutionTrace:
    """Telemetry trace for a single layer execution step."""
    layer_idx: int
    status: str  # "COMPUTE", "CACHE_HIT", "INTERVENTION"
    execution_time_ms: float
    disk_read_bytes: int = 0
    rss_ram_mb: float = 0.0
    vram_mb: float = 0.0


@dataclass
class LayerExecutionOutput:
    """Output and telemetry from sequential layer-wise execution."""
    prompt: str
    tokens: List[str]
    final_hidden_state: torch.Tensor
    logits: torch.Tensor
    layer_residuals: Dict[int, torch.Tensor] = field(default_factory=dict)
    cached_artifact_keys: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    layers_executed: int = 0
    layers_cached: int = 0
    cache_hit_rate: float = 0.0
    peak_ram_mb: float = 0.0
    peak_vram_mb: float = 0.0
    total_disk_bytes_read: int = 0
    layer_disk_bytes_read: int = 0
    traces: List[LayerExecutionTrace] = field(default_factory=list)


def _get_process_ram_mb() -> float:
    """Returns current process RSS memory in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _get_gpu_vram_mb() -> float:
    """Returns current GPU allocated memory in megabytes."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


class LayerPager:
    """Progressive layer streamer, disk pager, and sequential execution runner."""

    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        offload_to_cpu: bool = False,
        store: Optional[ArtifactStore] = None,
        weight_store: Optional[DiskWeightStore] = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.offload_to_cpu = offload_to_cpu
        self.store = store or get_artifact_store()
        self.weight_store = weight_store or get_disk_weight_store()

    def run_sequential_forward(
        self,
        model: nn.Module,
        tokenizer: Any,
        prompt: str,
        interventions: Optional[Dict[int, Callable[[torch.Tensor], torch.Tensor]]] = None,
        capture_layers: Optional[List[int]] = None,
        session_id: str = "default_session",
    ) -> LayerExecutionOutput:
        """Executes an in-memory layer-by-layer sequential forward pass with on-demand paging and hook capture."""
        start_time = time.time()
        interventions = interventions or {}
        capture_layers = capture_layers or []
        traces: List[LayerExecutionTrace] = []
        peak_ram = _get_process_ram_mb()
        peak_vram = _get_gpu_vram_mb()

        # 1. Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]
        prompt_digest = str(hash(prompt))

        # Model provenance & dynamic introspection
        spec = ModelIntrospector.introspect(model)
        prov = Provenance(
            model_id=spec.model_id,
            precision=str(self.dtype),
            device=self.device,
            operation="sequential_layer_forward",
        )

        # 2. Embedding pass & causal mask preparation
        h, causal_mask, position_ids, extra_kwargs = spec.prepare_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=self.device,
            dtype=self.dtype,
        )

        layer_residuals: Dict[int, torch.Tensor] = {}
        cached_keys: List[str] = []
        layers_computed = 0
        layers_cached = 0

        # 3. Sequential block forward pass
        blocks = spec.layer_stack
        num_layers = len(blocks)
        for layer_idx in range(num_layers):
            layer_t0 = time.time()
            block = blocks[layer_idx]
            
            # Optional: Page block to target device if it was parked on CPU
            if self.offload_to_cpu and self.device != "cpu":
                block.to(self.device)

            h = spec.execute_layer(
                layer_module=block,
                hidden_states=h,
                causal_mask=causal_mask,
                position_ids=position_ids,
                **extra_kwargs,
            )
            layers_computed += 1

            status = "COMPUTE"
            # Apply any active interventions for this layer
            if layer_idx in interventions:
                h = interventions[layer_idx](h)
                status = "INTERVENTION"

            # Capture activation artifact if requested
            if layer_idx in capture_layers or not capture_layers:
                layer_residuals[layer_idx] = h.detach().cpu()
                
                cas_key = compute_artifact_key(
                    parent_ids=[f"L{layer_idx - 1}_residual" if layer_idx > 0 else "embedding"],
                    operation="layer_forward",
                    operation_params={"layer": layer_idx},
                    provenance_digest=prov.compute_digest(),
                    layer=layer_idx,
                    component="residual",
                )
                meta = ArtifactMetadata(
                    name=f"residual_L{layer_idx}",
                    component="residual",
                    layer=layer_idx,
                    shape=tuple(h.shape),
                    dtype=str(h.dtype),
                    device=str(h.device),
                    session_id=session_id,
                    prompt_id=prompt_digest,
                )
                art = ExecutionArtifact(
                    artifact_id=cas_key,
                    metadata=meta,
                    provenance=prov,
                    tensor=h.detach().cpu(),
                )
                self.store.put(art, persist_to_disk=False)
                cached_keys.append(cas_key)

            # Offload block from device if offloading enabled
            if self.offload_to_cpu and self.device != "cpu":
                block.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            cur_ram = _get_process_ram_mb()
            cur_vram = _get_gpu_vram_mb()
            peak_ram = max(peak_ram, cur_ram)
            peak_vram = max(peak_vram, cur_vram)
            traces.append(LayerExecutionTrace(
                layer_idx=layer_idx,
                status=status,
                execution_time_ms=(time.time() - layer_t0) * 1000.0,
                rss_ram_mb=cur_ram,
                vram_mb=cur_vram,
            ))

        # 4. Final Norm and Unembed
        if self.offload_to_cpu and self.device != "cpu":
            if spec.final_norm is not None:
                spec.final_norm.to(self.device)
            if spec.output_head is not None:
                spec.output_head.to(self.device)

        logits = spec.finalize(hidden_states=h)

        if self.offload_to_cpu and self.device != "cpu":
            if spec.final_norm is not None:
                spec.final_norm.to("cpu")
            if spec.output_head is not None:
                spec.output_head.to("cpu")

        elapsed = time.time() - start_time
        return LayerExecutionOutput(
            prompt=prompt,
            tokens=tokens,
            final_hidden_state=h.detach().cpu(),
            logits=logits.detach().cpu(),
            layer_residuals=layer_residuals,
            cached_artifact_keys=cached_keys,
            execution_time_seconds=elapsed,
            layers_executed=layers_computed,
            layers_cached=layers_cached,
            cache_hit_rate=layers_cached / max(num_layers, 1),
            peak_ram_mb=peak_ram,
            peak_vram_mb=peak_vram,
            traces=traces,
        )

    def run_disk_paged_forward(
        self,
        manifest: ShardedModelManifest,
        prompt: str,
        tokenizer: Any,
        reference_model: Optional[nn.Module] = None,
        interventions: Optional[Dict[int, Callable[[torch.Tensor], torch.Tensor]]] = None,
        capture_layers: Optional[List[int]] = None,
        session_id: str = "disk_paged_session",
    ) -> LayerExecutionOutput:
        """Executes a disk-resident model by streaming layers on-demand from NVMe SSD with CAS caching."""
        start_time = time.time()
        interventions = interventions or {}
        capture_layers = capture_layers or []
        traces: List[LayerExecutionTrace] = []
        peak_ram = _get_process_ram_mb()
        peak_vram = _get_gpu_vram_mb()
        total_disk_bytes = 0
        layer_disk_bytes = 0

        # 1. Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]
        prompt_digest = str(hash(prompt))

        # Model provenance
        prov = Provenance(
            model_id=manifest.model_id,
            precision=str(self.dtype),
            device=self.device,
            operation="disk_paged_layer_forward",
        )

        spec = ModelIntrospector.introspect(reference_model) if reference_model is not None else None

        # 2. Embedding Pass with CAS Cache Check
        embed_cas_key = compute_artifact_key(
            parent_ids=["prompt"],
            operation="embedding_forward",
            operation_params={"prompt_digest": prompt_digest},
            provenance_digest=prov.compute_digest(),
            component="embedding",
        )
        cached_embed = self.store.get(embed_cas_key)

        causal_mask = None
        position_ids = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)

        if cached_embed is not None and cached_embed.tensor is not None:
            h = cached_embed.tensor.to(device=self.device, dtype=self.dtype)
            if spec is not None:
                _, causal_mask, position_ids, _ = spec.prepare_hidden_states(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    device=self.device,
                    dtype=self.dtype,
                )
        else:
            embed_meta = manifest.special_components.get("embedding")
            if embed_meta:
                embed_weights = self.weight_store.load_layer_weights(
                    manifest.model_id, 0, component="embedding", device=self.device, dtype=self.dtype
                )
                total_disk_bytes += embed_meta.size_bytes

                if spec is not None:
                    # Load disk weights into input & positional embeddings modules
                    if spec.input_embeddings is not None:
                        sub_dict = {
                            k.replace("input_embeddings.", ""): v
                            for k, v in embed_weights.items()
                            if k.startswith("input_embeddings.")
                        }
                        if not sub_dict:
                            sub_dict = embed_weights
                        spec.input_embeddings.load_state_dict(sub_dict, strict=False)
                        spec.input_embeddings.to(self.device)

                    if spec.positional_embeddings is not None:
                        sub_dict = {
                            k.replace("positional_embeddings.", ""): v
                            for k, v in embed_weights.items()
                            if k.startswith("positional_embeddings.")
                        }
                        if sub_dict:
                            spec.positional_embeddings.load_state_dict(sub_dict, strict=False)
                            spec.positional_embeddings.to(self.device)

                    h, causal_mask, position_ids, _ = spec.prepare_hidden_states(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        device=self.device,
                        dtype=self.dtype,
                    )
                else:
                    # Fallback for synthetic scaling benchmarks
                    wte_weight = next(iter(embed_weights.values())) if embed_weights else None
                    if wte_weight is not None and wte_weight.ndim == 2:
                        h = nn.functional.embedding(input_ids, wte_weight)
                    else:
                        h = torch.randn(input_ids.shape[0], input_ids.shape[1], manifest.hidden_size, device=self.device, dtype=self.dtype)

                del embed_weights
                gc.collect()
            else:
                h = torch.randn(input_ids.shape[0], input_ids.shape[1], manifest.hidden_size, device=self.device, dtype=self.dtype)

            # Store embedding artifact in CAS
            embed_art = ExecutionArtifact(
                artifact_id=embed_cas_key,
                metadata=ArtifactMetadata(
                    name="embedding_output",
                    component="embedding",
                    shape=tuple(h.shape),
                    dtype=str(h.dtype),
                    device=str(h.device),
                    session_id=session_id,
                    prompt_id=prompt_digest,
                ),
                provenance=prov,
                tensor=h.detach().cpu(),
            )
            self.store.put(embed_art, persist_to_disk=False)

        # 3. Instantiate a single reusable layer block instance in memory (1 layer allocation only)
        if spec is not None and spec.layer_stack is not None and len(spec.layer_stack) > 0:
            reusable_block = spec.layer_stack[0]
            reusable_block.to(self.device)
            reusable_block.eval()
        else:
            reusable_block = nn.TransformerEncoderLayer(
                d_model=manifest.hidden_size,
                nhead=manifest.num_heads,
                dim_feedforward=manifest.hidden_size * 4,
                batch_first=True,
                dtype=self.dtype,
            ).to(self.device)
            reusable_block.eval()

        layer_residuals: Dict[int, torch.Tensor] = {}
        cached_keys: List[str] = [embed_cas_key]
        layers_computed = 0
        layers_cached = 0
        parent_cas_key = embed_cas_key
        has_diverged = False

        num_layers = manifest.num_layers
        for layer_idx in range(num_layers):
            layer_t0 = time.time()
            disk_bytes_this_layer = 0

            # Calculate deterministic CAS key for this layer's output
            cas_key = compute_artifact_key(
                parent_ids=[parent_cas_key],
                operation="disk_layer_forward",
                operation_params={"layer": layer_idx, "intervened": layer_idx in interventions},
                provenance_digest=prov.compute_digest(),
                layer=layer_idx,
                component="residual",
            )

            # Check if this layer's output is already cached and upstream has not diverged
            cached_artifact = self.store.get(cas_key) if not has_diverged else None

            if cached_artifact is not None and cached_artifact.tensor is not None and layer_idx not in interventions:
                # ── 100% CACHE HIT: ZERO DISK READ, ZERO COMPUTE ──
                h = cached_artifact.tensor.to(device=self.device, dtype=self.dtype)
                parent_cas_key = cas_key
                cached_keys.append(cas_key)
                layers_cached += 1
                status = "CACHE_HIT"
            else:
                # ── CACHE MISS OR INTERVENTION: STREAM LAYER FROM NVMe & COMPUTE ──
                has_diverged = True
                layer_meta = manifest.layers.get(layer_idx)
                if layer_meta:
                    self.weight_store.load_layer_into_block(
                        manifest.model_id, layer_idx, reusable_block, device=self.device, component="block"
                    )
                    disk_bytes_this_layer = layer_meta.size_bytes
                    total_disk_bytes += disk_bytes_this_layer
                    layer_disk_bytes += disk_bytes_this_layer
                    
                    with torch.no_grad():
                        if spec is not None:
                            h = spec.execute_layer(
                                layer_module=reusable_block,
                                hidden_states=h,
                                causal_mask=causal_mask,
                                position_ids=position_ids,
                            )
                        else:
                            h = reusable_block(h)
                else:
                    with torch.no_grad():
                        if spec is not None:
                            h = spec.execute_layer(
                                layer_module=reusable_block,
                                hidden_states=h,
                                causal_mask=causal_mask,
                                position_ids=position_ids,
                            )
                        else:
                            h = reusable_block(h)

                layers_computed += 1
                status = "COMPUTE"

                # Apply intervention if specified for this layer
                if layer_idx in interventions:
                    h = interventions[layer_idx](h)
                    status = "INTERVENTION"

                # Store output activation artifact in CAS
                meta = ArtifactMetadata(
                    name=f"residual_L{layer_idx}",
                    component="residual",
                    layer=layer_idx,
                    shape=tuple(h.shape),
                    dtype=str(h.dtype),
                    device=str(h.device),
                    session_id=session_id,
                    prompt_id=prompt_digest,
                )
                art = ExecutionArtifact(
                    artifact_id=cas_key,
                    metadata=meta,
                    provenance=prov,
                    tensor=h.detach().cpu(),
                )
                self.store.put(art, persist_to_disk=False)
                parent_cas_key = cas_key
                cached_keys.append(cas_key)

            if layer_idx in capture_layers or not capture_layers:
                layer_residuals[layer_idx] = h.detach().cpu()

            cur_ram = _get_process_ram_mb()
            cur_vram = _get_gpu_vram_mb()
            peak_ram = max(peak_ram, cur_ram)
            peak_vram = max(peak_vram, cur_vram)
            traces.append(LayerExecutionTrace(
                layer_idx=layer_idx,
                status=status,
                execution_time_ms=(time.time() - layer_t0) * 1000.0,
                disk_read_bytes=disk_bytes_this_layer,
                rss_ram_mb=cur_ram,
                vram_mb=cur_vram,
            ))

        # 4. Final Head & Logits with CAS Cache Check
        logits_cas_key = compute_artifact_key(
            parent_ids=[parent_cas_key],
            operation="head_forward",
            operation_params={},
            provenance_digest=prov.compute_digest(),
            component="logits",
        )
        cached_logits = self.store.get(logits_cas_key) if not has_diverged else None

        if cached_logits is not None and cached_logits.tensor is not None:
            logits = cached_logits.tensor.to(device=self.device, dtype=self.dtype)
        else:
            norm_meta = manifest.special_components.get("norm")
            if norm_meta:
                norm_weights = self.weight_store.load_layer_weights(
                    manifest.model_id, num_layers, component="norm", device=self.device, dtype=self.dtype
                )
                total_disk_bytes += norm_meta.size_bytes
                if spec is not None and spec.final_norm is not None:
                    spec.final_norm.load_state_dict(norm_weights, strict=False)
                    spec.final_norm.to(self.device)
                del norm_weights

            head_meta = manifest.special_components.get("head")
            if head_meta:
                head_weights = self.weight_store.load_layer_weights(
                    manifest.model_id, num_layers + 1 if norm_meta else num_layers, component="head", device=self.device, dtype=self.dtype
                )
                total_disk_bytes += head_meta.size_bytes
                if spec is not None and spec.output_head is not None:
                    spec.output_head.load_state_dict(head_weights, strict=False)
                    spec.output_head.to(self.device)
                    logits = spec.finalize(h)
                else:
                    lm_weight = None
                    for k, v in head_weights.items():
                        if any(term in k for term in ["lm_head.weight", "output.weight"]):
                            lm_weight = v
                    if lm_weight is None and head_weights:
                        for v in head_weights.values():
                            if v.ndim == 2 and (v.shape[0] == manifest.vocab_size or v.shape[1] == manifest.hidden_size):
                                lm_weight = v
                                break
                    if lm_weight is not None:
                        logits = torch.matmul(h, lm_weight.T)
                    else:
                        logits = h
                del head_weights
                gc.collect()
            else:
                if spec is not None:
                    logits = spec.finalize(h)
                else:
                    logits = h

            # Store logits in CAS
            logits_art = ExecutionArtifact(
                artifact_id=logits_cas_key,
                metadata=ArtifactMetadata(
                    name="model_logits",
                    component="logits",
                    shape=tuple(logits.shape),
                    dtype=str(logits.dtype),
                    device=str(logits.device),
                    session_id=session_id,
                    prompt_id=prompt_digest,
                ),
                provenance=prov,
                tensor=logits.detach().cpu(),
            )
            self.store.put(logits_art, persist_to_disk=False)

        del reusable_block
        gc.collect()

        elapsed = time.time() - start_time
        return LayerExecutionOutput(
            prompt=prompt,
            tokens=tokens,
            final_hidden_state=h.detach().cpu(),
            logits=logits.detach().cpu(),
            layer_residuals=layer_residuals,
            cached_artifact_keys=cached_keys,
            execution_time_seconds=elapsed,
            layers_executed=layers_computed,
            layers_cached=layers_cached,
            cache_hit_rate=layers_cached / max(num_layers, 1),
            peak_ram_mb=peak_ram,
            peak_vram_mb=peak_vram,
            total_disk_bytes_read=total_disk_bytes,
            layer_disk_bytes_read=layer_disk_bytes,
            traces=traces,
        )
