"""In-Memory Reference Runtime Implementation for MECH Platform.

Provides fully-resident standard execution for comparison, validation, and baseline benchmarking.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .interfaces import (
    ModelRuntimeInterface,
    PrecisionProfile,
    RuntimeForwardOutput,
    RuntimeInterventionOutput,
    RuntimeMetadata,
)
from .universal_adapter import UniversalModelAdapter
from .virtual_unembedding import VirtualUnembeddingEngine


class InMemoryRuntime(ModelRuntimeInterface):
    """Standard in-memory resident transformer runtime."""

    def __init__(
        self,
        model_id: str = "gpt2",
        device: Optional[str] = None,
        precision: Optional[PrecisionProfile] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.precision = precision or PrecisionProfile(weight_dtype="float32", activation_dtype="float32")
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.adapter = UniversalModelAdapter(model=self.model)
        self.virtual_unembedding = VirtualUnembeddingEngine(chunk_size=4096)
        self._config = self.model.config
        self.num_layers = self.adapter.topology.num_layers
        self.hidden_dim = self.adapter.topology.hidden_dim
        self.vocab_size = self.adapter.topology.vocab_size

    def get_runtime_metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            runtime_type="in_memory",
            model_id=self.model_id,
            architecture=self.model.__class__.__name__,
            num_layers=self.num_layers,
            hidden_dimension=self.hidden_dim,
            vocab_size=self.vocab_size,
            precision=self.precision,
            weight_storage="resident_ram",
            vram_budget_mb=16384.0,
            ram_budget_mb=32768.0,
            residency_policy="eager_all",
            device=self.device,
        )


    def forward(
        self,
        prompt: str,
        target_token: Optional[str] = None,
        capture_layer_residuals: bool = False,
    ) -> RuntimeForwardOutput:
        if not prompt or not str(prompt).strip():
            raise ValueError("Prompt cannot be empty or whitespace.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        token_ids = inputs["input_ids"][0].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_logits = outputs.logits[0, -1]
        top_id = int(torch.argmax(last_logits).item())
        top_tok = self.tokenizer.decode([top_id])

        target_logit = None
        target_prob = None
        target_rank = None

        if target_token is not None:
            t_ids = self.tokenizer.encode(target_token)
            if t_ids:
                t_id = t_ids[-1]
                target_logit = float(last_logits[t_id].item())
                probs = torch.softmax(last_logits, dim=-1)
                target_prob = float(probs[t_id].item())
                target_rank = int((last_logits > target_logit).sum().item())

        residuals = None
        if capture_layer_residuals and outputs.hidden_states is not None:
            residuals = {i: h[0, -1].detach().cpu() for i, h in enumerate(outputs.hidden_states)}

        return RuntimeForwardOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=token_ids,
            top_predicted_token=top_tok,
            top_predicted_id=top_id,
            target_token=target_token,
            target_logit=target_logit,
            target_probability=target_prob,
            target_rank=target_rank,
            layer_residuals=residuals,
            runtime_metadata=self.get_runtime_metadata(),
        )

    def compute_logit_lens_trajectory(
        self,
        prompt: str,
        target_token: str,
        distractor_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("Model did not return hidden_states.")

        # Final LayerNorm/RMSNorm and lm_head via adapter
        ln_f = self.adapter.get_final_norm(self.model)
        lm_head = self.adapter.get_lm_head(self.model)

        t_ids = self.tokenizer.encode(target_token)
        target_id = t_ids[-1]

        trajectory = []
        for l_idx, h in enumerate(hidden_states):
            h_last = h[0, -1]
            if ln_f is not None and l_idx < len(hidden_states) - 1:
                h_normed = ln_f(h_last)
            else:
                h_normed = h_last

            with torch.no_grad():
                layer_logits = lm_head(h_normed)

            top_id = int(torch.argmax(layer_logits).item())
            top_tok = self.tokenizer.decode([top_id])
            target_l = float(layer_logits[target_id].item())
            probs = torch.softmax(layer_logits, dim=-1)
            target_p = float(probs[target_id].item())
            target_r = int((layer_logits > target_l).sum().item())

            trajectory.append({
                "layer": l_idx,
                "top_token": top_tok,
                "top_token_id": top_id,
                "target_logit": round(target_l, 4),
                "target_probability": round(target_p, 6),
                "target_rank": target_r,
            })

        return trajectory

    def project_to_vocabulary(
        self,
        hidden_state: Any,
        top_k: int = 10,
        target_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        h_tensor = hidden_state if isinstance(hidden_state, torch.Tensor) else torch.tensor(hidden_state)
        # Weight matrix from lm_head
        lm_head = self.adapter.get_lm_head(self.model)
        w_u = lm_head.weight.data

        return self.virtual_unembedding.project_hidden_state(
            hidden_state=h_tensor,
            unembedding_weights=w_u,
            tokenizer=self.tokenizer,
            top_k=top_k,
            target_token=target_token,
        )

    def apply_intervention(
        self,
        prompt: str,
        target_token: str,
        layer: int,
        component_type: str,
        component_index: int,
        ablation_scale: float = 0.0,
    ) -> RuntimeInterventionOutput:
        clean = self.forward(prompt, target_token=target_token)
        clean_l = clean.target_logit or 0.0
        clean_p = clean.target_probability or 0.0
        clean_r = clean.target_rank if clean.target_rank is not None else 0

        t_ids = self.tokenizer.encode(target_token)
        target_id = t_ids[-1]

        # Setup hook
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                act = output[0].clone()
                if act.shape[-1] > component_index:
                    act[:, :, component_index] *= ablation_scale
                return (act,) + output[1:]
            act = output.clone()
            if act.shape[-1] > component_index:
                act[:, :, component_index] *= ablation_scale
            return act

        target_module = self.model.transformer.h[layer].mlp.c_fc
        handle = target_module.register_forward_hook(hook_fn)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                intervened_out = self.model(**inputs)
        finally:
            handle.remove()

        last_logits = intervened_out.logits[0, -1]
        interv_l = float(last_logits[target_id].item())
        probs = torch.softmax(last_logits, dim=-1)
        interv_p = float(probs[target_id].item())
        interv_r = int((last_logits > interv_l).sum().item())

        delta_l = clean_l - interv_l
        delta_p = clean_p - interv_p
        delta_r = interv_r - clean_r

        verdict = f"Intervention on L{layer}_{component_type}_{component_index} produced Δz = {delta_l:+.4f} logit shift."

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
    ) -> Any:
        from .activation_artifact import ActivationArtifact
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Layer 0 is embedding output, layers 1..N are outputs of blocks 0..N-1
        h_layer = outputs.hidden_states[layer + 1]  # [1, seq_len, d_model]
        target_tensor = h_layer[:, sequence_position, :].clone()

        return ActivationArtifact.from_tensor(
            tensor=target_tensor,
            model_id=self.model_id,
            layer=layer,
            sequence_position=sequence_position,
            experiment_hash=experiment_hash,
            source_runtime="in_memory",
            intervention_context={"prompt": prompt, "component": component},
        )

    def patch_activation(
        self,
        target_prompt: str,
        target_token: str,
        layer: int,
        artifact: Any,
        component: str = "residual",
        sequence_position: int = -1,
    ) -> RuntimeForwardOutput:
        from .activation_artifact import ActivationArtifact
        source_tensor = artifact.get_tensor(self.device) if isinstance(artifact, ActivationArtifact) else artifact.to(self.device)

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                h[:, sequence_position, :] = source_tensor
                return (h,) + output[1:]
            h = output.clone()
            h[:, sequence_position, :] = source_tensor
            return h

        target_block = self.model.transformer.h[layer]
        handle = target_block.register_forward_hook(patch_hook)

        try:
            inputs = self.tokenizer(target_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
        finally:
            handle.remove()

        last_logits = outputs.logits[0, -1]
        top_id = int(torch.argmax(last_logits).item())
        top_tok = self.tokenizer.decode([top_id])

        t_ids = self.tokenizer.encode(target_token)
        t_id = t_ids[-1]
        t_logit = float(last_logits[t_id].item())
        probs = torch.softmax(last_logits, dim=-1)
        t_prob = float(probs[t_id].item())
        t_rank = int((last_logits > t_logit).sum().item())

        return RuntimeForwardOutput(
            prompt=target_prompt,
            tokens=[self.tokenizer.decode([tid]) for tid in inputs["input_ids"][0].tolist()],
            token_ids=inputs["input_ids"][0].tolist(),
            top_predicted_token=top_tok,
            top_predicted_id=top_id,
            target_token=target_token,
            target_logit=t_logit,
            target_probability=t_prob,
            target_rank=t_rank,
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
        from .interfaces import PathPatchingOutput
        # 1. Baseline target forward (corrupted / target prompt)
        clean_target = self.forward(source_prompt, target_token=target_token)
        corrupted_target = self.forward(target_prompt, target_token=target_token)

        # 2. Capture source activations along hops
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

        # 3. Patch into target
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

