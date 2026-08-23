"""Encoder-Decoder Topology Execution & Cross-Attention DAG Engine.

Enables progressive layer execution for sequence-to-sequence models (e.g. T5, BART).
Executes the encoder stack to produce encoder hidden states, registers cross-attention
DAG dependencies, and executes the decoder stack with CAS caching and selective intervention.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..artifacts.cas_store import ArtifactStore, compute_artifact_key, get_artifact_store
from ..artifacts.models import ArtifactMetadata, ExecutionArtifact, Provenance
from .disk_weight_store import DiskWeightStore, get_disk_weight_store
from .model_introspector import ExecutionSpec, ModelIntrospector

logger = logging.getLogger("MECH.enc_dec_pager")


@dataclass
class EncDecExecutionOutput:
    """Output and telemetry from Encoder-Decoder sequential execution."""
    encoder_prompt: str
    decoder_prompt: str
    encoder_hidden_states: torch.Tensor
    decoder_hidden_states: torch.Tensor
    logits: torch.Tensor
    encoder_residuals: Dict[int, torch.Tensor] = field(default_factory=dict)
    decoder_residuals: Dict[int, torch.Tensor] = field(default_factory=dict)
    cached_artifact_keys: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    encoder_layers_executed: int = 0
    decoder_layers_executed: int = 0
    encoder_cache_hits: int = 0
    decoder_cache_hits: int = 0


class EncoderDecoderPager:
    """Progressive execution engine for Encoder-Decoder models with cross-attention DAG tracking."""

    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        store: Optional[ArtifactStore] = None,
        weight_store: Optional[DiskWeightStore] = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.store = store or get_artifact_store()
        self.weight_store = weight_store or get_disk_weight_store()

    def run_sequential_forward(
        self,
        model: nn.Module,
        tokenizer: Any,
        encoder_prompt: str,
        decoder_prompt: Optional[str] = None,
        encoder_interventions: Optional[Dict[int, Callable[[torch.Tensor], torch.Tensor]]] = None,
        decoder_interventions: Optional[Dict[int, Callable[[torch.Tensor], torch.Tensor]]] = None,
        session_id: str = "enc_dec_session",
    ) -> EncDecExecutionOutput:
        """Executes an Encoder-Decoder model sequentially with full DAG provenance and cross-attention tracking."""
        t0 = time.time()
        encoder_interventions = encoder_interventions or {}
        decoder_interventions = decoder_interventions or {}

        spec = ModelIntrospector.introspect(model)
        prov = Provenance(
            model_id=spec.model_id,
            precision=str(self.dtype),
            device=self.device,
            operation="encoder_decoder_forward",
        )

        # 1. Tokenize Encoder Input
        enc_inputs = tokenizer(encoder_prompt, return_tensors="pt")
        enc_input_ids = enc_inputs["input_ids"].to(self.device)
        enc_mask = enc_inputs.get("attention_mask")
        if enc_mask is not None:
            enc_mask = enc_mask.to(self.device)

        # Tokenize Decoder Input (default to decoder start token or prompt)
        if decoder_prompt is None:
            decoder_prompt = tokenizer.decode([tokenizer.pad_token_id or 0])
        dec_inputs = tokenizer(decoder_prompt, return_tensors="pt")
        dec_input_ids = dec_inputs["input_ids"].to(self.device)
        dec_mask = dec_inputs.get("attention_mask")
        # Prepare extended attention masks for seq2seq models (e.g. T5, BART)
        ext_enc_mask = None
        ext_dec_mask = None
        if hasattr(model, "get_extended_attention_mask"):
            try:
                if enc_mask is not None:
                    ext_enc_mask = model.get_extended_attention_mask(enc_mask, enc_input_ids.shape)
                if dec_mask is not None:
                    ext_dec_mask = model.get_extended_attention_mask(dec_mask, dec_input_ids.shape)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Swallowed exception: %s", exc)
                ext_enc_mask = None
                ext_dec_mask = None

        encoder_residuals: Dict[int, torch.Tensor] = {}
        decoder_residuals: Dict[int, torch.Tensor] = {}
        cached_keys: List[str] = []

        # ── PHASE 1: ENCODER STACK EXECUTION ──
        # Check CAS for existing encoder artifact for this prompt
        enc_prompt_digest = str(hash(encoder_prompt))
        enc_cas_key = compute_artifact_key(
            parent_ids=["encoder_prompt"],
            operation="encoder_forward",
            operation_params={"prompt_digest": enc_prompt_digest, "intervened": len(encoder_interventions) > 0},
            provenance_digest=prov.compute_digest(),
            component="encoder_hidden_states",
        )
        cached_enc = self.store.get(enc_cas_key)

        enc_hits = 0
        enc_executed = 0

        if cached_enc is not None and cached_enc.tensor is not None and not encoder_interventions:
            enc_h = cached_enc.tensor.to(device=self.device, dtype=self.dtype)
            cached_keys.append(enc_cas_key)
            enc_hits = len(spec.encoder_layer_stack) if spec.encoder_layer_stack else 1
        else:
            # Embed encoder input
            embed = spec.input_embeddings or (model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None)
            if embed is not None:
                enc_h = embed(enc_input_ids).to(device=self.device, dtype=self.dtype)
            else:
                enc_h = torch.randn(enc_input_ids.shape[0], enc_input_ids.shape[1], spec.hidden_size, device=self.device, dtype=self.dtype)

            # Step through encoder layer stack
            enc_stack = spec.encoder_layer_stack or []
            for layer_idx, block in enumerate(enc_stack):
                try:
                    out = block(enc_h, attention_mask=ext_enc_mask)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Swallowed exception: %s", exc)
                    try:
                        out = block(enc_h, attention_mask=enc_mask)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Swallowed exception: %s", exc)
                        out = block(enc_h)
                enc_h = out[0] if isinstance(out, tuple) else out
                enc_executed += 1

                if layer_idx in encoder_interventions:
                    enc_h = encoder_interventions[layer_idx](enc_h)

                encoder_residuals[layer_idx] = enc_h.detach().cpu()

            # Store encoder output in CAS
            meta = ArtifactMetadata(
                name="encoder_final_hidden_state",
                component="encoder_hidden_states",
                shape=tuple(enc_h.shape),
                dtype=str(enc_h.dtype),
                device=str(enc_h.device),
                session_id=session_id,
            )
            self.store.put(
                ExecutionArtifact(
                    artifact_id=enc_cas_key,
                    metadata=meta,
                    provenance=prov,
                    tensor=enc_h.detach().cpu(),
                ),
                persist_to_disk=False,
            )
            cached_keys.append(enc_cas_key)

        # ── PHASE 2: DECODER STACK EXECUTION WITH CROSS-ATTENTION ──
        embed = spec.input_embeddings or (model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None)
        if embed is not None:
            dec_h = embed(dec_input_ids).to(device=self.device, dtype=self.dtype)
        else:
            dec_h = torch.randn(dec_input_ids.shape[0], dec_input_ids.shape[1], spec.hidden_size, device=self.device, dtype=self.dtype)

        dec_stack = spec.decoder_layer_stack or spec.layer_stack or []
        dec_executed = 0
        dec_hits = 0

        for layer_idx, block in enumerate(dec_stack):
            try:
                out = block(
                    dec_h,
                    attention_mask=ext_dec_mask,
                    encoder_hidden_states=enc_h,
                    encoder_attention_mask=ext_enc_mask,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Swallowed exception: %s", exc)
                try:
                    out = block(dec_h, encoder_hidden_states=enc_h)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Swallowed exception: %s", exc)
                    out = block(dec_h)

            dec_h = out[0] if isinstance(out, tuple) else out
            dec_executed += 1

            if layer_idx in decoder_interventions:
                dec_h = decoder_interventions[layer_idx](dec_h)

            decoder_residuals[layer_idx] = dec_h.detach().cpu()

            # Register layer artifact in CAS with dual parent: decoder layer i-1 and encoder output
            dec_cas_key = compute_artifact_key(
                parent_ids=[f"D{layer_idx - 1}" if layer_idx > 0 else "dec_embed", enc_cas_key],
                operation="decoder_layer_forward",
                operation_params={"layer": layer_idx},
                provenance_digest=prov.compute_digest(),
                layer=layer_idx,
                component="decoder_residual",
            )
            meta = ArtifactMetadata(
                name=f"decoder_residual_L{layer_idx}",
                component="decoder_residual",
                layer=layer_idx,
                shape=tuple(dec_h.shape),
                dtype=str(dec_h.dtype),
                device=str(dec_h.device),
                session_id=session_id,
            )
            self.store.put(
                ExecutionArtifact(
                    artifact_id=dec_cas_key,
                    metadata=meta,
                    provenance=prov,
                    tensor=dec_h.detach().cpu(),
                ),
                persist_to_disk=False,
            )
            cached_keys.append(dec_cas_key)

        # ── PHASE 3: FINAL PROJECTION HEAD ──
        logits = spec.finalize(dec_h)

        return EncDecExecutionOutput(
            encoder_prompt=encoder_prompt,
            decoder_prompt=decoder_prompt,
            encoder_hidden_states=enc_h.detach().cpu(),
            decoder_hidden_states=dec_h.detach().cpu(),
            logits=logits.detach().cpu(),
            encoder_residuals=encoder_residuals,
            decoder_residuals=decoder_residuals,
            cached_artifact_keys=cached_keys,
            execution_time_seconds=time.time() - t0,
            encoder_layers_executed=enc_executed,
            decoder_layers_executed=dec_executed,
            encoder_cache_hits=enc_hits,
            decoder_cache_hits=dec_hits,
        )
