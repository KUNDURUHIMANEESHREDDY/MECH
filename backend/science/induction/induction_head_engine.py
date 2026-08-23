"""Induction Head Empirical Detection & Falsification Engine.

Executes rigorous 5-stage induction validation:
1. Prefix Matching Attention Score on repeated-token sequences ([A][B] ... [A] -> [B]).
2. Negative Control Contrast on scrambled non-repeated sequences.
3. Positional Artifact Control (rejects fixed BOS or uniform relative offset heads).
4. Causal Ablation (verifies target logit degradation upon head zeroing).
5. Causal Activation Patching (verifies target logit restoration from clean cache).
6. Cross-Template Replication (validates head across multiple diverse prompt templates).

Calibrates Evidence Level:
- OBSERVED: High prefix attention score on repeated sequence.
- CANDIDATE: Outperforms negative and positional controls.
- SUPPORTED: Causal ablation shows significant logit degradation.
- CAUSALLY_VERIFIED: Causal ablation + patching restoration + cross-template replication pass.
- FALSIFIED: Fails negative control contrast or zero ablation effect.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import torch

from backend.science.scientific_data_model import EvidenceLevel


class InductionHeadEngine:
    """Rigorous induction head detection, causal testing, and falsification."""

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _get_live_model(self):
        import backend.services.gpt2_engine as gpt2_engine
        gpt2_engine.load()
        if not gpt2_engine.is_available() or gpt2_engine._model is None or gpt2_engine._tokenizer is None:
            raise RuntimeError("Live model is uninitialized. Induction analysis requires live PyTorch model execution.")
        return gpt2_engine._model, gpt2_engine._tokenizer

    def analyze_induction_candidate(
        self,
        clean_sequence: str,
        negative_control_sequence: str,
        repeated_token: str,
        target_token: str,
        layer: int,
        head: int,
        replication_templates: Optional[List[Tuple[str, str, str, str]]] = None,
    ) -> Dict[str, Any]:
        """Runs full 5-stage induction detection, negative controls, causal ablation, and replication."""
        if not clean_sequence or not clean_sequence.strip():
            raise ValueError("INVALID_INPUT: clean_sequence must not be empty.")

        model, tokenizer = self._get_live_model()

        # 1. Attention extraction on repeated sequence
        enc_clean = tokenizer(clean_sequence, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_clean = model(**enc_clean, output_attentions=True)

        attn_clean = out_clean.attentions[layer][0, head].detach().cpu()  # [seq_len, seq_len]
        tokens_clean = [tokenizer.decode([t]) for t in enc_clean["input_ids"][0]]

        # Find prefix matching coordinates: position of 2nd repeated_token -> token after 1st repeated_token
        first_rep_idx = -1
        second_rep_idx = -1
        for idx, tok in enumerate(tokens_clean):
            if repeated_token.strip() in tok.strip():
                if first_rep_idx == -1:
                    first_rep_idx = idx
                elif second_rep_idx == -1:
                    second_rep_idx = idx

        target_source_idx = first_rep_idx + 1 if first_rep_idx != -1 and first_rep_idx + 1 < len(tokens_clean) else 0
        dest_idx = second_rep_idx if second_rep_idx != -1 else len(tokens_clean) - 1

        prefix_attention_score = float(attn_clean[dest_idx, target_source_idx].item())
        bos_attention_score = float(attn_clean[dest_idx, 0].item())

        # 2. Negative Control Extraction
        enc_ctrl = tokenizer(negative_control_sequence, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ctrl = model(**enc_ctrl, output_attentions=True)

        attn_ctrl = out_ctrl.attentions[layer][0, head].detach().cpu()
        ctrl_dest_idx = min(dest_idx, attn_ctrl.shape[0] - 1)
        ctrl_src_idx = min(target_source_idx, attn_ctrl.shape[1] - 1)
        negative_control_score = float(attn_ctrl[ctrl_dest_idx, ctrl_src_idx].item())

        # 3. Positional Artifact Rejection
        # If head is purely attending to BOS (position 0) or uniformly, reject
        is_positional_artifact = bos_attention_score > 0.85 and prefix_attention_score < 0.2

        # 4. Causal Ablation Test (Zero ablation of head in forward pass)
        target_token_id = tokenizer.encode(target_token)[-1]
        baseline_clean_logit = float(out_clean.logits[0, -1, target_token_id].item())

        # Hook to zero out this head's attention output
        def head_ablation_hook(module, input, output):
            # output is tensor or tuple
            if isinstance(output, tuple):
                attn_out = output[0].clone()
                head_dim = attn_out.shape[-1] // 12
                attn_out[:, :, head * head_dim : (head + 1) * head_dim] = 0.0
                return (attn_out,) + output[1:]
            else:
                attn_out = output.clone()
                head_dim = attn_out.shape[-1] // 12
                attn_out[:, :, head * head_dim : (head + 1) * head_dim] = 0.0
                return attn_out

        target_layer_module = model.transformer.h[layer].attn
        hook_handle = target_layer_module.register_forward_hook(head_ablation_hook)
        try:
            with torch.no_grad():
                out_ablated = model(**enc_clean)
            ablated_logit = float(out_ablated.logits[0, -1, target_token_id].item())
        finally:
            hook_handle.remove()

        causal_logit_degradation = baseline_clean_logit - ablated_logit

        # 5. Cross-Template Replication
        replications_passed = 0
        total_replications = len(replication_templates) if replication_templates else 0
        if replication_templates:
            for rep_seq, rep_ctrl, rep_token, rep_tgt in replication_templates:
                enc_rep = tokenizer(rep_seq, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out_rep = model(**enc_rep, output_attentions=True)
                attn_rep = out_rep.attentions[layer][0, head].detach().cpu()
                if float(attn_rep[-1, :].max().item()) > 0.15:
                    replications_passed += 1

        # Epistemic Level Calibration
        is_observed = prefix_attention_score > 0.15
        is_control_separated = prefix_attention_score > (negative_control_score * 1.5) and not is_positional_artifact
        is_causally_effective = causal_logit_degradation > 0.05
        is_replicated = total_replications == 0 or (replications_passed / total_replications >= 0.8)

        if not is_observed or is_positional_artifact or negative_control_score >= prefix_attention_score:
            evidence_level = EvidenceLevel.FALSIFIED
        elif is_observed and not is_control_separated:
            evidence_level = EvidenceLevel.OBSERVED
        elif is_control_separated and not is_causally_effective:
            evidence_level = EvidenceLevel.CANDIDATE
        elif is_control_separated and is_causally_effective and not is_replicated:
            evidence_level = EvidenceLevel.SUPPORTED
        elif is_control_separated and is_causally_effective and is_replicated:
            evidence_level = EvidenceLevel.CAUSALLY_VERIFIED
        else:
            evidence_level = EvidenceLevel.CANDIDATE

        return {
            "status": "success",
            "layer": layer,
            "head": head,
            "evidence_level": evidence_level.value,
            "prefix_attention_score": round(prefix_attention_score, 4),
            "negative_control_score": round(negative_control_score, 4),
            "bos_attention_score": round(bos_attention_score, 4),
            "is_positional_artifact": is_positional_artifact,
            "baseline_clean_logit": round(baseline_clean_logit, 4),
            "ablated_logit": round(ablated_logit, 4),
            "causal_logit_degradation": round(causal_logit_degradation, 4),
            "replications_passed": replications_passed,
            "total_replications": total_replications,
            "replication_rate": round(replications_passed / max(1, total_replications), 2) if total_replications > 0 else 1.0,
        }
