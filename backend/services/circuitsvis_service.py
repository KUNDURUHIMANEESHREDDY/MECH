"""CircuitsVis Interactive Circuit Visualization Service.

Directly utilizes the official circuitsvis library (TransformerLensOrg / Jesse Vig & Neel Nanda)
to generate canonical interactive Attention Patterns and Attention Heads visualizations
powered exclusively by live PyTorch model forward execution tensors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch

try:
    import circuitsvis
    from circuitsvis.attention import attention_patterns, attention_heads
    _CIRCUITSVIS_AVAILABLE = True
except ImportError:
    _CIRCUITSVIS_AVAILABLE = False


class CircuitsVisService:
    """Provides canonical CircuitsVis interactive visualization bundles from live model execution."""

    def __init__(self, model_name: str = "gpt2") -> None:
        self.model_name = model_name

    def _ensure_live_execution(self, prompt: str):
        if not _CIRCUITSVIS_AVAILABLE:
            raise RuntimeError("The official 'circuitsvis' library is not installed in the active Python environment.")

        if not prompt or not prompt.strip():
            raise ValueError("INVALID_INPUT: Prompt must not be empty or whitespace only.")

        import backend.services.gpt2_engine as gpt2_engine
        gpt2_engine.load()
        if not gpt2_engine.is_available() or gpt2_engine._model is None or gpt2_engine._tokenizer is None:
            raise RuntimeError("Live model is uninitialized. CircuitsVis requires live PyTorch model execution.")

        model = gpt2_engine._model
        tokenizer = gpt2_engine._tokenizer

        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**enc, output_attentions=True)

        if not hasattr(outputs, "attentions") or outputs.attentions is None or len(outputs.attentions) == 0:
            raise RuntimeError("Model forward pass did not return attention tensors. Cannot generate CircuitsVis visualization.")

        input_ids = enc["input_ids"][0].tolist()
        tokens = [tokenizer.decode([tok_id]).replace("Ġ", " ") for tok_id in input_ids]

        return model, tokenizer, outputs.attentions, tokens

    def get_attention_patterns(self, prompt: str, layer: int = 8) -> Dict[str, Any]:
        """Generates canonical CircuitsVis Attention Patterns interactive HTML."""
        model, tokenizer, attentions, tokens = self._ensure_live_execution(prompt)

        # attentions[layer]: [batch=1, num_heads=12, seq_len, seq_len]
        if layer < 0 or layer >= len(attentions):
            raise IndexError(f"Layer {layer} out of bounds [0, {len(attentions) - 1}].")

        layer_attn = attentions[layer][0].detach().cpu().float()  # [num_heads, seq_len, seq_len]

        html_obj = attention_patterns(
            tokens=tokens,
            attention=layer_attn,
        )

        html_data = str(html_obj)

        return {
            "status": "success",
            "view_type": "circuitsvis_attention_patterns",
            "library": "circuitsvis",
            "library_version": getattr(circuitsvis, "__version__", "1.43.3"),
            "model_name": self.model_name,
            "prompt": prompt,
            "tokens": tokens,
            "layer": layer,
            "html": html_data,
            "provenance": "OFFICIAL_CIRCUITSVIS_LIVE_PYTORCH",
        }

    def get_attention_heads(self, prompt: str) -> Dict[str, Any]:
        """Generates canonical CircuitsVis Attention Heads interactive HTML across all layers."""
        model, tokenizer, attentions, tokens = self._ensure_live_execution(prompt)

        # Stack into [num_layers, num_heads, seq_len, seq_len]
        all_attn = torch.stack([a[0] for a in attentions], dim=0).detach().cpu().float()

        html_obj = attention_heads(
            tokens=tokens,
            attention=all_attn,
        )

        html_data = str(html_obj)

        return {
            "status": "success",
            "view_type": "circuitsvis_attention_heads",
            "library": "circuitsvis",
            "library_version": getattr(circuitsvis, "__version__", "1.43.3"),
            "model_name": self.model_name,
            "prompt": prompt,
            "tokens": tokens,
            "html": html_data,
            "provenance": "OFFICIAL_CIRCUITSVIS_LIVE_PYTORCH",
        }
