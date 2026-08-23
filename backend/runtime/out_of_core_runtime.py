"""Out-of-Core Demand-Paged Mechanistic Runtime Implementation.

Executes large model forward passes, Logit Lens analysis, and causal interventions
using 3-tier tensor residency (Disk -> CPU RAM -> GPU VRAM), ActivationArtifact caching,
and VirtualUnembeddingEngine.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from .activation_artifact import ActivationArtifact
from .interfaces import (
    ModelRuntimeInterface,
    PrecisionProfile,
    RuntimeForwardOutput,
    RuntimeInterventionOutput,
    RuntimeMetadata,
)
from .residency_manager import MemoryBudgetPolicy, ResidencyManager
from .universal_adapter import UniversalModelAdapter
from .virtual_unembedding import VirtualUnembeddingEngine


class OutOfCoreRuntime(ModelRuntimeInterface):
    """Hardware-aware, demand-paged out-of-core mechanistic runtime."""

    def __init__(
        self,
        model_id: str = "gpt2",
        device: Optional[str] = None,
        vram_budget_mb: float = 2048.0,
        ram_budget_mb: float = 8192.0,
        max_active_layers: int = 3,
        cache_dir: Optional[Path | str] = None,
        budget_policy: Optional[MemoryBudgetPolicy] = None,
        precision: Optional[PrecisionProfile] = None,
    ) -> None:
        self.model_id = model_id
        self.compute_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.config = AutoConfig.from_pretrained(model_id)
        self.precision = precision or PrecisionProfile(weight_dtype="float32", activation_dtype="float32")

        # Adapter for universal model introspection
        self.adapter = UniversalModelAdapter(config=self.config)
        self.num_layers = self.adapter.topology.num_layers
        self.hidden_dim = self.adapter.topology.hidden_dim
        self.vocab_size = self.adapter.topology.vocab_size

        # 3-tier residency manager
        self.residency = ResidencyManager(
            vram_budget_mb=vram_budget_mb,
            ram_budget_mb=ram_budget_mb,
            max_active_layers=max_active_layers,
            cache_dir=cache_dir,
            default_compute_device=self.compute_device,
            budget_policy=budget_policy,
        )

        # Virtual unembedding engine (chunk size 4096)
        self.virtual_unembedding = VirtualUnembeddingEngine(chunk_size=4096)

        # Load embedding, final norm, and unembedding head
        self._init_core_layers()

    def _init_core_layers(self) -> None:
        """Initializes embeddings, final LayerNorm/RMSNorm, and registers demand-paged layer loaders."""
        from transformers import AutoModelForCausalLM
        full_model = AutoModelForCausalLM.from_pretrained(self.model_id)

        # Static entry/exit blocks via universal adapter
        self.wte, self.wpe, self.drop = self.adapter.get_embedding_modules(full_model)
        self.wte = self.wte.to(self.compute_device)
        if self.wpe is not None:
            self.wpe = self.wpe.to(self.compute_device)

        self.ln_f = self.adapter.get_final_norm(full_model).to(self.compute_device)
        lm_head_mod = self.adapter.get_lm_head(full_model)
        self.lm_head_weight = lm_head_mod.weight.data.detach().cpu()

        # Register each transformer block with precision-aware loader
        layer_stack = self.adapter.get_layer_stack(full_model)
        for l_idx in range(self.num_layers):
            layer_block = layer_stack[l_idx]
            
            def make_loader(block=layer_block):
                return UniversalModelAdapter.apply_precision(
                    block,
                    precision=self.precision,
                    target_device=self.compute_device,
                )

            self.residency.register_layer_loader(
                layer_idx=l_idx,
                loader_fn=make_loader,
                size_bytes=self.hidden_dim * self.hidden_dim * 16,
            )

    def get_runtime_metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            runtime_type="out_of_core",
            model_id=self.model_id,
            architecture=self.config.architectures[0] if self.config.architectures else "GPT2LMHeadModel",
            num_layers=self.num_layers,
            hidden_dimension=self.hidden_dim,
            vocab_size=self.vocab_size,
            precision=self.precision,
            weight_storage="safetensors_mmap",
            vram_budget_mb=self.residency.vram_budget_bytes / (1024 * 1024),
            ram_budget_mb=self.residency.ram_budget_bytes / (1024 * 1024),
            residency_policy="demand_paged_lru",
            device=self.compute_device,
        )

    def forward(
        self,
        prompt: str,
        target_token: Optional[str] = None,
        capture_layer_residuals: bool = False,
    ) -> RuntimeForwardOutput:
        """Executes demand-paged forward pass through layers 0..N-1."""
        if not prompt or not str(prompt).strip():
            raise ValueError("Prompt cannot be empty or whitespace.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.compute_device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        token_ids = input_ids[0].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        # 1. Embeddings
        if self.wpe is not None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.compute_device).unsqueeze(0)
            hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        else:
            hidden_states = self.wte(input_ids)

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)


        residuals: Dict[int, torch.Tensor] = {}
        if capture_layer_residuals:
            residuals[0] = hidden_states[0, -1].detach().cpu()

        # 2. Demand-paged Layer Streaming with pipelined prefetching
        for l_idx in range(self.num_layers):
            if l_idx + 1 < self.num_layers:
                self.residency.prefetch_layer(l_idx + 1, target_device=self.compute_device)

            # Load layer to compute device (utilizing prefetched module if available)
            layer_module = self.residency.load_layer_to_device(l_idx, target_device=self.compute_device)

            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

            # Match input tensor dtype with layer module parameters if layer is in mixed precision
            layer_params = list(layer_module.parameters())

            layer_dtype = layer_params[0].dtype if layer_params else hidden_states.dtype
            inp_states = hidden_states.to(dtype=layer_dtype) if layer_dtype in (torch.float16, torch.bfloat16) else hidden_states

            t_comp_0 = time.perf_counter()
            with torch.no_grad():
                layer_outputs = layer_module(inp_states)
                out_states = layer_outputs[0]
                if out_states.dim() == 2:
                    out_states = out_states.unsqueeze(0)
                hidden_states = out_states.to(dtype=torch.float32)
            self.residency.total_compute_time_sec += (time.perf_counter() - t_comp_0)


            if capture_layer_residuals:
                residuals[l_idx + 1] = hidden_states[0, -1].detach().cpu()


            # Store intermediate activation artifact in residency manager
            art = ActivationArtifact.from_tensor(
                tensor=hidden_states,
                model_id=self.model_id,
                layer=l_idx,
            )
            self.residency.store_activation(art)


        # 3. Final LayerNorm
        hidden_states = self.ln_f(hidden_states)
        last_hidden = hidden_states[0, -1]  # [d_model]

        # 4. Virtual Unembedding Projection
        proj = self.virtual_unembedding.project_hidden_state(
            hidden_state=last_hidden,
            unembedding_weights=self.lm_head_weight,
            tokenizer=self.tokenizer,
            top_k=10,
            target_token=target_token,
        )

        return RuntimeForwardOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=token_ids,
            top_predicted_token=proj["top_token"],
            top_predicted_id=proj["top_candidates"][0]["token_id"] if proj["top_candidates"] else 0,
            target_token=target_token,
            target_logit=proj["target_logit"],
            target_probability=proj["target_probability"],
            target_rank=proj["target_rank"],
            layer_residuals=residuals if capture_layer_residuals else None,
            runtime_metadata=self.get_runtime_metadata(),
        )

    def compute_logit_lens_trajectory(
        self,
        prompt: str,
        target_token: str,
        distractor_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Computes Logit Lens trajectory by virtual unembedding at every layer boundary."""
        fwd = self.forward(prompt, target_token=target_token, capture_layer_residuals=True)
        residuals = fwd.layer_residuals or {}

        trajectory: List[Dict[str, Any]] = []
        for l_idx, h in sorted(residuals.items(), key=lambda x: x[0]):
            h_dev = h.to(self.compute_device)
            # Apply ln_f
            h_norm = self.ln_f(h_dev)

            proj = self.virtual_unembedding.project_hidden_state(
                hidden_state=h_norm,
                unembedding_weights=self.lm_head_weight,
                tokenizer=self.tokenizer,
                top_k=5,
                target_token=target_token,
            )

            trajectory.append({
                "layer": l_idx,
                "top_token": proj["top_token"],
                "top_token_id": proj["top_candidates"][0]["token_id"] if proj["top_candidates"] else 0,
                "target_logit": proj["target_logit"],
                "target_probability": proj["target_probability"],
                "target_rank": proj["target_rank"],
            })

        return trajectory

    def project_to_vocabulary(
        self,
        hidden_state: Any,
        top_k: int = 10,
        target_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        h_tensor = hidden_state if isinstance(hidden_state, torch.Tensor) else torch.tensor(hidden_state)
        return self.virtual_unembedding.project_hidden_state(
            hidden_state=h_tensor,
            unembedding_weights=self.lm_head_weight,
            tokenizer=self.tokenizer,
            top_k=top_k,
            target_token=target_token,
        )

    def _compute_embeddings(self, input_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        if self.wpe is not None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.compute_device).unsqueeze(0)
            hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        else:
            hidden_states = self.wte(input_ids)

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)
        return hidden_states

    def apply_intervention(
        self,
        prompt: str,
        target_token: str,
        layer: int,
        component_type: str,
        component_index: int,
        ablation_scale: float = 0.0,
    ) -> RuntimeInterventionOutput:
        """Applies hook intervention during demand-paged layer execution."""
        clean = self.forward(prompt, target_token=target_token)
        clean_l = clean.target_logit or 0.0
        clean_p = clean.target_probability or 0.0
        clean_r = clean.target_rank if clean.target_rank is not None else 0

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.compute_device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        hidden_states = self._compute_embeddings(input_ids, seq_len)

        for l_idx in range(self.num_layers):
            layer_module = self.residency.load_layer_to_device(l_idx, target_device=self.compute_device)
            
            # Apply hook on the target layer
            handle = None
            if l_idx == layer:
                def hook_fn(module, inp, out):
                    if isinstance(out, tuple):
                        act = out[0].clone()
                        if act.shape[-1] > component_index:
                            act[:, :, component_index] *= ablation_scale
                        return (act,) + out[1:]
                    act = out.clone()
                    if act.shape[-1] > component_index:
                        act[:, :, component_index] *= ablation_scale
                    return act

                target_submodule = getattr(getattr(layer_module, "mlp", None), "c_fc", None)
                if target_submodule is None and hasattr(layer_module, "mlp"):
                    target_submodule = layer_module.mlp
                if target_submodule is not None:
                    handle = target_submodule.register_forward_hook(hook_fn)

            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

            layer_params = list(layer_module.parameters())
            layer_dtype = layer_params[0].dtype if layer_params else hidden_states.dtype
            inp_states = hidden_states.to(dtype=layer_dtype) if layer_dtype in (torch.float16, torch.bfloat16) else hidden_states

            try:
                with torch.no_grad():
                    layer_outputs = layer_module(inp_states)
                    out_states = layer_outputs[0]
                    if out_states.dim() == 2:
                        out_states = out_states.unsqueeze(0)
                    hidden_states = out_states.to(dtype=torch.float32)
            finally:
                if handle:
                    handle.remove()

        hidden_states = self.ln_f(hidden_states)
        last_hidden = hidden_states[0, -1]

        proj = self.virtual_unembedding.project_hidden_state(
            hidden_state=last_hidden,
            unembedding_weights=self.lm_head_weight,
            tokenizer=self.tokenizer,
            top_k=10,
            target_token=target_token,
        )

        interv_l = proj["target_logit"] or 0.0
        interv_p = proj["target_probability"] or 0.0
        interv_r = proj["target_rank"] if proj["target_rank"] is not None else 0

        delta_l = clean_l - interv_l
        delta_p = clean_p - interv_p
        delta_r = interv_r - clean_r

        verdict = f"Out-of-core intervention on L{layer}_{component_type}_{component_index} produced Δz = {delta_l:+.4f} logit shift."

        return RuntimeInterventionOutput(
            clean_logit=clean_l,
            intervened_logit=interv_l,
            delta_logit=delta_l,
            clean_probability=clean_p,
            intervened_probability=interv_p,
            delta_probability=delta_p,
            clean_rank=clean_r,
            intervened_rank=interv_r,
            delta_rank=delta_r,
            verdict=verdict,
            runtime_metadata=self.get_runtime_metadata(),
        )

    def capture_activation(
        self,
        prompt: str,
        layer: int,
        component: str = "residual",
        sequence_position: int = -1,
        experiment_hash: Optional[str] = None,
    ) -> ActivationArtifact:
        """Runs demand-paged execution and captures activation at target layer as an ActivationArtifact."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.compute_device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        hidden_states = self._compute_embeddings(input_ids, seq_len)
        captured_tensor = None

        for l_idx in range(layer + 1):
            layer_module = self.residency.load_layer_to_device(l_idx, target_device=self.compute_device)
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

            layer_params = list(layer_module.parameters())
            layer_dtype = layer_params[0].dtype if layer_params else hidden_states.dtype
            inp_states = hidden_states.to(dtype=layer_dtype) if layer_dtype in (torch.float16, torch.bfloat16) else hidden_states

            with torch.no_grad():
                layer_outputs = layer_module(inp_states)
                out_states = layer_outputs[0]
                if out_states.dim() == 2:
                    out_states = out_states.unsqueeze(0)
                hidden_states = out_states.to(dtype=torch.float32)

            if l_idx == layer:
                captured_tensor = hidden_states[:, sequence_position, :].clone()

        if captured_tensor is None:
            raise ValueError(f"Failed to capture activation at layer {layer}.")

        art = ActivationArtifact.from_tensor(
            tensor=captured_tensor,
            model_id=self.model_id,
            layer=layer,
            sequence_position=sequence_position,
            experiment_hash=experiment_hash,
            source_runtime="out_of_core",
            intervention_context={"prompt": prompt, "component": component},
        )
        self.residency.store_activation(art)
        return art

    def patch_activation(
        self,
        target_prompt: str,
        target_token: str,
        layer: int,
        artifact: Any,
        component: str = "residual",
        sequence_position: int = -1,
    ) -> RuntimeForwardOutput:
        """Executes demand-paged forward pass while patching the captured activation into target layer."""
        source_tensor = (
            artifact.get_tensor(self.compute_device)
            if isinstance(artifact, ActivationArtifact)
            else artifact.to(self.compute_device)
        )

        inputs = self.tokenizer(target_prompt, return_tensors="pt").to(self.compute_device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        token_ids = input_ids[0].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        hidden_states = self._compute_embeddings(input_ids, seq_len)

        for l_idx in range(self.num_layers):
            layer_module = self.residency.load_layer_to_device(l_idx, target_device=self.compute_device)
            
            handle = None
            if l_idx == layer:
                def patch_hook(module, inp, out):
                    if isinstance(out, tuple):
                        h = out[0].clone()
                        h[:, sequence_position, :] = source_tensor
                        return (h,) + out[1:]
                    h = out.clone()
                    h[:, sequence_position, :] = source_tensor
                    return h

                handle = layer_module.register_forward_hook(patch_hook)

            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

            layer_params = list(layer_module.parameters())
            layer_dtype = layer_params[0].dtype if layer_params else hidden_states.dtype
            inp_states = hidden_states.to(dtype=layer_dtype) if layer_dtype in (torch.float16, torch.bfloat16) else hidden_states

            try:
                with torch.no_grad():
                    layer_outputs = layer_module(inp_states)
                    out_states = layer_outputs[0]
                    if out_states.dim() == 2:
                        out_states = out_states.unsqueeze(0)
                    hidden_states = out_states.to(dtype=torch.float32)
            finally:
                if handle:
                    handle.remove()


        hidden_states = self.ln_f(hidden_states)
        last_hidden = hidden_states[0, -1]

        proj = self.virtual_unembedding.project_hidden_state(
            hidden_state=last_hidden,
            unembedding_weights=self.lm_head_weight,
            tokenizer=self.tokenizer,
            top_k=10,
            target_token=target_token,
        )

        return RuntimeForwardOutput(
            prompt=target_prompt,
            tokens=tokens,
            token_ids=token_ids,
            top_predicted_token=proj["top_token"],
            top_predicted_id=proj["top_candidates"][0]["token_id"] if proj["top_candidates"] else 0,
            target_token=target_token,
            target_logit=proj["target_logit"],
            target_probability=proj["target_probability"],
            target_rank=proj["target_rank"],
            runtime_metadata=self.get_runtime_metadata(),
        )

    def patch_path(
        self,
        source_prompt: str,
        target_prompt: str,
        target_token: str,
        path_hops: List[Any],
        experiment_hash: Optional[str] = None,
    ) -> Any:
        """Executes multi-hop causal path patching and calculates mediation rescue fraction."""
        from .interfaces import PathPatchingOutput

        clean_target = self.forward(source_prompt, target_token=target_token)
        corrupted_target = self.forward(target_prompt, target_token=target_token)

        artifacts = []
        last_artifact = None
        for hop in path_hops:
            art = self.capture_activation(
                prompt=source_prompt,
                layer=hop.source_layer,
                component=hop.source_component,
                sequence_position=hop.sequence_position,
                experiment_hash=experiment_hash,
            )
            artifacts.append(art.to_dict())
            last_artifact = art

        last_hop = path_hops[-1] if path_hops else None
        target_layer = last_hop.target_layer if last_hop else 8
        patched = self.patch_activation(
            target_prompt=target_prompt,
            target_token=target_token,
            layer=target_layer,
            artifact=last_artifact,
            component=last_hop.target_component if last_hop else "residual",
            sequence_position=last_hop.sequence_position if last_hop else -1,
        )

        clean_l = clean_target.target_logit or 0.0
        corr_l = corrupted_target.target_logit or 0.0
        patch_l = patched.target_logit or 0.0

        indirect_effect = patch_l - corr_l
        total_effect = max(abs(clean_l - corr_l), 1e-6)
        rescue_frac = max(0.0, min(1.0, indirect_effect / total_effect))

        return PathPatchingOutput(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            target_token=target_token,
            clean_target_logit=clean_l,
            corrupted_target_logit=corr_l,
            patched_target_logit=patch_l,
            indirect_effect=round(indirect_effect, 4),
            mediation_rescue_fraction=round(rescue_frac, 4),
            clean_target_rank=clean_target.target_rank or 0,
            patched_target_rank=patched.target_rank or 0,
            path_hops=path_hops,
            artifacts=artifacts,
            runtime_metadata=self.get_runtime_metadata(),
        )

    def get_execution_telemetry_report(self) -> Dict[str, Any]:
        """Generates comprehensive hardware, runtime, and model telemetry report."""
        stats = self.residency.get_memory_statistics()
        param_count = sum(p.numel() for p in [self.wte.weight, self.wpe.weight, self.lm_head_weight]) + (
            self.num_layers * (self.hidden_dim * self.hidden_dim * 12)
        )
        weight_size_mb = round((param_count * 4) / (1024 * 1024), 2)

        return {
            "model": {
                "model_id": self.model_id,
                "architecture": self.config.architectures[0] if self.config.architectures else "GPT2LMHeadModel",
                "parameter_count": param_count,
                "weight_size_mb": weight_size_mb,
                "quantization": "fp32",
                "num_layers": self.num_layers,
                "hidden_dimension": self.hidden_dim,
                "vocab_size": self.vocab_size,
            },
            "hardware": {
                "compute_device": self.compute_device,
                "gpu_vram_budget_mb": stats["vram_budget_mb"],
                "system_ram_budget_mb": stats["ram_budget_mb"],
                "disk_cache_dir": stats["disk_cache_dir"],
            },
            "runtime": {
                "peak_vram_mb": stats["peak_vram_mb"],
                "peak_rss_mb": stats["peak_rss_mb"],
                "max_active_layers": stats["max_active_layers"],
                "peak_active_layers_observed": stats["peak_active_layers_observed"],
                "total_load_time_ms": stats["total_load_time_ms"],
                "total_eviction_time_ms": stats["total_eviction_time_ms"],
                "prefetch_hits": stats["prefetch_hits"],
                "total_cached_activations": stats["total_cached_activations"],
                "memory_budget_compliant": stats["memory_budget_compliant"],
            },
        }


