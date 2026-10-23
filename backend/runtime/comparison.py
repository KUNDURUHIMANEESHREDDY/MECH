"""Live, checkpoint-backed cross-model comparisons.

No statistic in this module is a fixture. Every emitted metric is reduced from
forward-pass tensors produced by two independently loaded Hugging Face models.
A metric that cannot be aligned safely is represented as unavailable with a
reason, never as a default zero or a guessed similarity.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from typing import Any, Dict, Optional


TO_COMPUTE = (
    "predictions.top_token_match",
    "predictions.model_a_top",
    "predictions.model_b_top",
    "activations.mlp_layer_cosine_similarity",
    "activations.mlp_mean_absolute_difference",
    "logits.kl_divergence_a_to_b",
    "residuals.layer_cosine_similarity",
    "residuals.mean_absolute_difference",
    "attention.tensor_cosine_similarity",
    "attention.mean_absolute_difference",
)


def _unavailable(prompt: str, model_a: str, model_b: str,
                 revision_a: str, revision_b: str, reason: str) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "implemented": True,
        "prompt": prompt,
        "model_a": model_a,
        "model_b": model_b,
        "revision_a_requested": revision_a,
        "revision_b_requested": revision_b,
        "provenance": "unavailable",
        "attested": False,
        "measured": False,
        "validation_eligible": False,
        "publication_eligible": False,
        "metrics_measured": False,
        "activations": {"layers": []},
        "predictions": {},
        "logits": {},
        "attention": {"layers": []},
        "residuals": {"layers": []},
        "metrics_required": list(TO_COMPUTE),
        "reason": reason,
    }


def _tokenizer_fingerprint(tokenizer: Any) -> Optional[str]:
    """Hash the actual tokenizer vocabulary and special-token mapping."""
    try:
        vocab = tokenizer.get_vocab()
        if not vocab:
            return None
        payload = {
            "vocab": sorted((str(token), int(index)) for token, index in vocab.items()),
            "special_tokens_map": {
                str(key): str(value)
                for key, value in sorted(tokenizer.special_tokens_map.items())
            },
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()
    except Exception:
        return None


def _model_metadata(model: Any, repo_id: str, requested_revision: str,
                    weight_fingerprint: str, tokenizer_fingerprint: str) -> Dict[str, Any]:
    config = model.config
    param = next(model.parameters())
    return {
        "repo_id_requested": repo_id,
        "revision_requested": requested_revision,
        "resolved_revision": getattr(config, "_commit_hash", None),
        "loaded_name_or_path": getattr(config, "_name_or_path", None),
        "weights_sha256": weight_fingerprint,
        "tokenizer_sha256": tokenizer_fingerprint,
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", None) or []),
        "num_hidden_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", None)),
        "num_attention_heads": getattr(config, "num_attention_heads", getattr(config, "n_head", None)),
        "hidden_size": getattr(config, "hidden_size", getattr(config, "n_embd", None)),
        "vocab_size": getattr(config, "vocab_size", None),
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        "device": str(param.device),
        "dtype": str(param.dtype),
    }


def _finite_number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


_MODEL_LOCKS_GUARD = threading.Lock()
_MODEL_LOCKS: Dict[int, Any] = {}


def _model_lock(model: Any) -> Any:
    """Serialize temporary forward hooks on a shared cached model instance."""
    key = id(model)
    with _MODEL_LOCKS_GUARD:
        if key not in _MODEL_LOCKS:
            _MODEL_LOCKS[key] = threading.RLock()
        return _MODEL_LOCKS[key]


def _layer_sequence(model: Any) -> Any:
    """Find common HF transformer block containers without assuming a family."""
    paths = (
        ("transformer", "h"),
        ("model", "layers"),
        ("gpt_neox", "layers"),
        ("transformer", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "blocks"),
        ("layers",),
    )
    for path in paths:
        current = model
        for part in path:
            current = getattr(current, part, None)
            if current is None:
                break
        if current is not None and not isinstance(current, (str, bytes)):
            try:
                if len(current) > 0 and hasattr(current, "__getitem__"):
                    return current
            except (TypeError, AttributeError):
                pass
    return ()


def _forward_with_activations(model: Any, inputs: Dict[str, Any], torch: Any):
    """Run a real forward pass and capture available MLP intermediate tensors."""
    captured: Dict[int, Any] = {}
    hooks = []
    lock = _model_lock(model)
    with lock:
        try:
            for index, block in enumerate(_layer_sequence(model)):
                mlp = getattr(block, "mlp", None)
                if mlp is None:
                    mlp = getattr(block, "feed_forward", getattr(block, "ffn", None))
                if mlp is None:
                    continue
                projection_name = next(
                    (name for name in (
                        "c_proj", "down_proj", "dense_4h_to_h", "fc2", "w2",
                    ) if getattr(mlp, name, None) is not None),
                    None,
                )
                if projection_name is not None:
                    projection = getattr(mlp, projection_name)

                    def capture_projection(_module, args, layer=index, name=projection_name):
                        if args and hasattr(args[0], "detach"):
                            captured[layer] = (args[0].detach(), "input_to_" + name)

                    hooks.append(projection.register_forward_pre_hook(capture_projection))
                else:
                    def capture_mlp(_module, _args, output, layer=index):
                        tensor = output[0] if isinstance(output, (tuple, list)) and output else output
                        if hasattr(tensor, "detach"):
                            captured[layer] = (tensor.detach(), "mlp_block_output")

                    hooks.append(mlp.register_forward_hook(capture_mlp))
            with torch.inference_mode():
                output = model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True,
                )
        finally:
            for hook in hooks:
                hook.remove()
    return output, captured


class HuggingFaceComparisonBackend:
    """Load and compare independently requested real checkpoints."""

    def compare(self, *, prompt: str, model_a: str, model_b: str,
                revision_a: str = "main", revision_b: str = "main") -> Dict[str, Any]:
        prompt = str(prompt or "")
        model_a = str(model_a or "").strip()
        model_b = str(model_b or "").strip()
        revision_a = str(revision_a or "main").strip()
        revision_b = str(revision_b or "main").strip()
        if not prompt.strip():
            return _unavailable(prompt, model_a, model_b, revision_a, revision_b,
                                "A non-empty input prompt is required; no input was fabricated.")
        if not model_a or not model_b:
            return _unavailable(prompt, model_a, model_b, revision_a, revision_b,
                                "Provide two independent Hugging Face checkpoint IDs.")

        try:
            import torch
            import torch.nn.functional as F
            from backend.science.models.adapter_base import hash_weights
            from backend.science.models.model_manager import ModelManager
        except Exception as exc:
            return _unavailable(prompt, model_a, model_b, revision_a, revision_b,
                                f"PyTorch/model dependencies are unavailable: {type(exc).__name__}: {exc}")

        try:
            manager = ModelManager()
            # Separate repo+revision cache keys ensure two independent loads.
            model_a_obj, tokenizer_a = manager.get_model_and_tokenizer(model_a, revision_a)
            model_b_obj, tokenizer_b = manager.get_model_and_tokenizer(model_b, revision_b)
            model_a_obj.eval()
            model_b_obj.eval()
            fingerprint_a = hash_weights(model_a_obj)
            fingerprint_b = hash_weights(model_b_obj)
            tokenizer_hash_a = _tokenizer_fingerprint(tokenizer_a)
            tokenizer_hash_b = _tokenizer_fingerprint(tokenizer_b)
            if not fingerprint_a or not fingerprint_b:
                return _unavailable(prompt, model_a, model_b, revision_a, revision_b,
                                    "At least one loaded model did not yield a verifiable weight fingerprint.")
            if not tokenizer_hash_a or not tokenizer_hash_b:
                return _unavailable(prompt, model_a, model_b, revision_a, revision_b,
                                    "At least one loaded tokenizer did not yield a verifiable vocabulary fingerprint.")

            encoded_a = tokenizer_a(prompt, return_tensors="pt")
            encoded_b = tokenizer_b(prompt, return_tensors="pt")
            device_a = next(model_a_obj.parameters()).device
            device_b = next(model_b_obj.parameters()).device
            encoded_a = {key: value.to(device_a) for key, value in encoded_a.items()}
            encoded_b = {key: value.to(device_b) for key, value in encoded_b.items()}
            ids_a = encoded_a["input_ids"][0].detach().cpu().tolist()
            ids_b = encoded_b["input_ids"][0].detach().cpu().tolist()
            tokens_a = tokenizer_a.convert_ids_to_tokens(ids_a)
            tokens_b = tokenizer_b.convert_ids_to_tokens(ids_b)

            out_a, mlp_a = _forward_with_activations(model_a_obj, encoded_a, torch)
            out_b, mlp_b = _forward_with_activations(model_b_obj, encoded_b, torch)

            logits_a = out_a.logits[0, -1].float()
            logits_b = out_b.logits[0, -1].float()
            if not torch.isfinite(logits_a).all().item() or not torch.isfinite(logits_b).all().item():
                return _unavailable(prompt, model_a, model_b, revision_a, revision_b,
                                    "A forward pass returned non-finite logits; no comparison was recorded.")

            prob_a = torch.softmax(logits_a, dim=-1)
            prob_b = torch.softmax(logits_b, dim=-1)
            top_id_a = int(torch.argmax(logits_a).item())
            top_id_b = int(torch.argmax(logits_b).item())
            top_token_a = str(tokenizer_a.decode([top_id_a]))
            top_token_b = str(tokenizer_b.decode([top_id_b]))
            top_prob_a = _finite_number(prob_a[top_id_a].item())
            top_prob_b = _finite_number(prob_b[top_id_b].item())

            same_tokenizer = tokenizer_hash_a == tokenizer_hash_b
            same_tokenization = ids_a == ids_b and tokens_a == tokens_b
            vocab_shape_matches = logits_a.shape == logits_b.shape
            logits_comparable = same_tokenizer and same_tokenization and vocab_shape_matches
            kl_ab = None
            kl_reason = None
            if logits_comparable:
                # KL(P_A || P_B), computed from the two live next-token distributions.
                # Promote before normalization to reduce underflow for large
                # vocabularies; probabilities and log probabilities come from
                # each model's actual last-position logits.
                log_p_a = torch.log_softmax(logits_a.double(), dim=-1)
                log_p_b = torch.log_softmax(logits_b.double(), dim=-1)
                p_a = log_p_a.exp()
                raw_kl = _finite_number(torch.sum(p_a * (log_p_a - log_p_b)).item())
                # KL is non-negative analytically; clamp only tiny negative
                # floating-point roundoff, never an unavailable measurement.
                kl_ab = max(0.0, raw_kl) if raw_kl is not None else None
                if kl_ab is None:
                    kl_reason = "KL reduction returned a non-finite value."
            else:
                kl_reason = (
                    "KL is only comparable when tokenizer vocabulary/special-token "
                    "fingerprints match, the current prompt tokenizes identically, and "
                    "both output logits have identical shapes."
                )

            config_a = model_a_obj.config
            config_b = model_b_obj.config
            same_family = getattr(config_a, "model_type", None) == getattr(config_b, "model_type", None)
            hidden_a = getattr(config_a, "hidden_size", getattr(config_a, "n_embd", None))
            hidden_b = getattr(config_b, "hidden_size", getattr(config_b, "n_embd", None))
            heads_a = getattr(config_a, "num_attention_heads", getattr(config_a, "n_head", None))
            heads_b = getattr(config_b, "num_attention_heads", getattr(config_b, "n_head", None))
            architecture_residual_compatible = (
                same_family and hidden_a is not None and hidden_a == hidden_b
                and same_tokenization
            )

            residual_layers = []
            residual_reason = None
            hidden_states_a = getattr(out_a, "hidden_states", None)
            hidden_states_b = getattr(out_b, "hidden_states", None)
            if not same_tokenization:
                residual_reason = "The checkpoints tokenize this prompt differently; token-position alignment is unsafe."
            elif not same_family or hidden_a != hidden_b:
                residual_reason = "Residual coordinates are only compared for matching model families and hidden dimensions."
            elif hidden_states_a is None or hidden_states_b is None:
                residual_reason = "At least one model did not return hidden states."
            else:
                # HF hidden_states[0] is the embedding output, not a
                # transformer residual block. Compare post-block states only.
                for state_index in range(1, min(len(hidden_states_a), len(hidden_states_b))):
                    tensor_a = hidden_states_a[state_index]
                    tensor_b = hidden_states_b[state_index]
                    a = tensor_a.detach().float().cpu()
                    b = tensor_b.detach().float().cpu()
                    if a.shape != b.shape:
                        continue
                    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
                    cosine = _finite_number(F.cosine_similarity(flat_a, flat_b, dim=0).item())
                    mae = _finite_number(torch.mean(torch.abs(flat_a - flat_b)).item())
                    if cosine is not None and mae is not None:
                        layer_index = state_index - 1
                        residual_layers.append({
                            "layer_a": layer_index, "layer_b": layer_index,
                            "shape": list(a.shape),
                            "cosine_similarity": cosine,
                            "mean_absolute_difference": mae,
                        })
                if not residual_layers:
                    residual_reason = "No residual layers had exactly aligned finite tensors."

            activation_layers = []
            activation_reason = None
            activation_kinds = sorted({
                value[1] for value in list(mlp_a.values()) + list(mlp_b.values())
                if isinstance(value, tuple) and len(value) == 2
            })
            if not same_tokenization:
                activation_reason = "The checkpoints tokenize this prompt differently; activation positions cannot be aligned."
            elif not same_family:
                activation_reason = "MLP activation comparison requires the same model family."
            else:
                for layer_a in sorted(set(mlp_a).intersection(mlp_b)):
                    item_a, item_b = mlp_a[layer_a], mlp_b[layer_a]
                    if not isinstance(item_a, tuple) or not isinstance(item_b, tuple):
                        continue
                    tensor_a, kind_a = item_a
                    tensor_b, kind_b = item_b
                    if kind_a != kind_b:
                        continue
                    a = tensor_a.detach().float().cpu()
                    b = tensor_b.detach().float().cpu()
                    if a.shape != b.shape or not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
                        continue
                    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
                    cosine = _finite_number(F.cosine_similarity(flat_a, flat_b, dim=0).item())
                    mae = _finite_number(torch.mean(torch.abs(flat_a - flat_b)).item())
                    if cosine is not None and mae is not None:
                        activation_layers.append({
                            "layer_a": layer_a, "layer_b": layer_a,
                            "shape": list(a.shape),
                            "activation_kind": kind_a,
                            "cosine_similarity": cosine,
                            "mean_absolute_difference": mae,
                        })
                if not activation_layers:
                    activation_reason = (
                        "No MLP activation tensors had matching layer indices, capture kinds, "
                        "token positions, and finite tensor shapes."
                    )

            attention_layers = []
            attention_reason = None
            attentions_a = getattr(out_a, "attentions", None)
            attentions_b = getattr(out_b, "attentions", None)
            if not same_tokenization:
                attention_reason = "The checkpoints tokenize this prompt differently; attention positions cannot be aligned."
            elif not same_family or heads_a is None or heads_a != heads_b:
                attention_reason = "Attention tensors are only aligned for matching model families and head counts."
            elif attentions_a is None or attentions_b is None:
                attention_reason = "At least one model runtime did not return attention tensors."
            else:
                for layer_a, tensor_a in enumerate(attentions_a):
                    if layer_a >= len(attentions_b):
                        break
                    tensor_b = attentions_b[layer_a]
                    a = tensor_a.detach().float().cpu()
                    b = tensor_b.detach().float().cpu()
                    if a.shape != b.shape:
                        continue
                    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
                    cosine = _finite_number(F.cosine_similarity(flat_a, flat_b, dim=0).item())
                    mae = _finite_number(torch.mean(torch.abs(flat_a - flat_b)).item())
                    if cosine is not None and mae is not None:
                        attention_layers.append({
                            "layer_a": layer_a, "layer_b": layer_a,
                            "shape": list(a.shape),
                            "tensor_cosine_similarity": cosine,
                            "mean_absolute_difference": mae,
                        })
                if not attention_layers:
                    attention_reason = "No attention layers had exactly aligned finite tensors."

            metadata_a = _model_metadata(model_a_obj, model_a, revision_a, fingerprint_a, tokenizer_hash_a)
            metadata_b = _model_metadata(model_b_obj, model_b, revision_b, fingerprint_b, tokenizer_hash_b)
            comparable_token = top_token_a == top_token_b
            return {
                "status": "completed",
                "implemented": True,
                "prompt": prompt,
                "input": {
                    "text": prompt,
                    "sha256": "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "token_ids_a": ids_a,
                    "token_ids_b": ids_b,
                    "tokens_a": tokens_a,
                    "tokens_b": tokens_b,
                },
                "model_a": model_a,
                "model_b": model_b,
                "models": {"a": metadata_a, "b": metadata_b},
                "settings": {
                    "inference_mode": "torch.inference_mode",
                    "output_hidden_states": True,
                    "output_attentions": True,
                    "logits_position": "last input token",
                    "residual_alignment": "same tokenization, model family, hidden width, layer index and tensor shape",
                    "attention_alignment": "same tokenization, model family, head count, layer index and tensor shape",
                },
                "provenance": "live",
                "attested": True,
                "measured": True,
                "validation_eligible": False,
                "publication_eligible": False,
                "metrics_measured": True,
                "field_provenance": {
                    "status": "live",
                    "prompt": "live",
                    "input": "live",
                    "models": "live",
                    "settings": "reference",
                    "predictions": "live",
                    "logits": "live" if kl_ab is not None else "unavailable",
                    "activations": "live" if activation_layers else "unavailable",
                    "residuals": "live" if residual_layers else "unavailable",
                    "attention": "live" if attention_layers else "unavailable",
                },
                "activations": {
                    "comparable": bool(activation_layers),
                    "layers": activation_layers,
                    "not_comparable_reason": activation_reason,
                    "capture_kinds_observed": activation_kinds,
                    "interpretation": "Live MLP intermediate tensors; coordinates are compared numerically and are not asserted to have identical neuron semantics.",
                },
                "predictions": {
                    "model_a_top": {"token": top_token_a, "token_id": top_id_a, "probability": top_prob_a},
                    "model_b_top": {"token": top_token_b, "token_id": top_id_b, "probability": top_prob_b},
                    "top_token_match": comparable_token,
                    "token_match_basis": "decoded output token string",
                },
                "logits": {
                    "kl_divergence_a_to_b": kl_ab,
                    "comparable": logits_comparable,
                    "not_comparable_reason": kl_reason,
                    "output_shape_a": list(logits_a.shape),
                    "output_shape_b": list(logits_b.shape),
                    "vocabulary_fingerprints_match": same_tokenizer,
                },
                "residuals": {
                    "comparable": bool(residual_layers),
                    "layers": residual_layers,
                    "not_comparable_reason": residual_reason,
                    "coordinate_interpretation": "Numeric tensor comparison only; no claim that individual neurons have equivalent semantics.",
                },
                "attention": {
                    "comparable": bool(attention_layers),
                    "layers": attention_layers,
                    "not_comparable_reason": attention_reason,
                    "head_interpretation": "Head indices are aligned numerically; no causal-equivalence claim is made.",
                },
                "metrics_required": list(TO_COMPUTE),
            }
        except Exception as exc:
            return _unavailable(prompt, model_a, model_b, revision_a, revision_b,
                                f"Real checkpoint comparison failed: {type(exc).__name__}: {exc}")


class ModelComparisonEngine:
    """Computes comparisons via an injected real model backend."""

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend

    def compare_available(self) -> bool:
        # A protocol-compatible object may still return fabricated dictionaries.
        # Only the built-in implementation is permitted to attest live results.
        return type(self.backend) is HuggingFaceComparisonBackend

    def compare(self, prompt: str, model_a: str, model_b: str,
                revision_a: str = "main", revision_b: str = "main") -> Dict[str, Any]:
        if type(self.backend) is not HuggingFaceComparisonBackend:
            return _unavailable(
                prompt, model_a, model_b, revision_a, revision_b,
                "No trusted Hugging Face comparison backend is configured; "
                "custom/self-attested result dictionaries are refused.",
            )
        result = self.backend.compare(
            prompt=prompt, model_a=model_a, model_b=model_b,
            revision_a=revision_a, revision_b=revision_b,
        )
        if not isinstance(result, dict) or result.get("status") != "completed":
            return result if isinstance(result, dict) else _unavailable(
                prompt, model_a, model_b, revision_a, revision_b,
                "The comparison backend did not return a structured measured result.",
            )
        # Validate the attestation envelope rather than accepting plausible
        # numbers from an injected or accidentally substituted implementation.
        models = result.get("models")
        input_record = result.get("input")
        if (
            result.get("provenance") != "live"
            or result.get("attested") is not True
            or result.get("measured") is not True
            or not isinstance(models, dict)
            or not isinstance(input_record, dict)
            or not isinstance(input_record.get("sha256"), str)
            or not input_record["sha256"].startswith("sha256:")
        ):
            return _unavailable(prompt, model_a, model_b, revision_a, revision_b,
                                "The result is missing live-forward attestation or an input commitment.")
        from backend.science.models.adapter_base import is_canonical_fingerprint
        for side in ("a", "b"):
            identity = models.get(side)
            digest = identity.get("weights_sha256") if isinstance(identity, dict) else None
            tokenizer_digest = identity.get("tokenizer_sha256") if isinstance(identity, dict) else None
            if not is_canonical_fingerprint(digest) or not isinstance(tokenizer_digest, str) or not tokenizer_digest.startswith("sha256:"):
                return _unavailable(prompt, model_a, model_b, revision_a, revision_b,
                                    "Comparison lacks a weight-derived or tokenizer-derived fingerprint.")
        return result
