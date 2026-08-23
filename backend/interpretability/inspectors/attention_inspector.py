"""Attention Inspector.

Analyzes multi-head attention patterns, query-key attention scores,
and attention head specialization (induction heads, previous-token heads).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AttentionInspector:
    """Inspector for multi-head attention distributions."""

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id
        self._adapter = None

    def _get_adapter(self):
        """Lazy-load model adapter."""
        if self._adapter is None and self.model_id:
            try:
                from backend.core.model_adapter import get_model_adapter
                self._adapter = get_model_adapter(model_id=self.model_id)
                self._adapter.load()
            except Exception:
                self._adapter = None
        return self._adapter

    def inspect(self, layer: int, head: int, tokens: List[str] | None = None) -> Dict[str, Any]:
        """Inspect attention pattern matrix for specified head.

        Args:
            layer: Layer index.
            head: Attention head index.
            tokens: Input token strings.

        Returns:
            Dict containing attention weights matrix and head classification.
            If model is unavailable, returns NOT_EXECUTABLE status.
        """
        toks = tokens or ["The", "capital", "of", "France"]
        
        adapter = self._get_adapter()
        if adapter is None:
            return {
                "layer": layer,
                "head": head,
                "head_id": f"L{layer}_H{head}",
                "tokens": toks,
                "head_type": "UNKNOWN",
                "attention_matrix": [],
                "status": "NOT_EXECUTABLE",
                "error": "No model loaded. Cannot compute real attention patterns.",
            }
        
        try:
            import torch
            
            # Tokenize and run forward pass
            inputs = adapter.tokenizer(" ".join(toks), return_tensors="pt")
            inputs = {k: v.to(adapter._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                out = adapter.model(**inputs, output_attentions=True, return_dict=True)
            
            # Extract attention matrix for specified layer and head
            # attentions is a tuple of (n_layers,) tensors, each [batch, n_heads, seq_len, seq_len]
            attentions = out.attentions
            if layer < len(attentions):
                attn_tensor = attentions[layer][0, head]  # [seq_len, seq_len]
                matrix = attn_tensor.cpu().tolist()
                # Round for readability
                matrix = [[round(val, 4) for val in row] for row in matrix]
            else:
                matrix = []
            
            # Classify head type based on attention pattern
            head_type = self._classify_head_type(matrix, head)
            
            return {
                "layer": layer,
                "head": head,
                "head_id": f"L{layer}_H{head}",
                "tokens": toks,
                "head_type": head_type,
                "attention_matrix": matrix,
                "status": "COMPLETED",
            }
            
        except Exception as e:
            return {
                "layer": layer,
                "head": head,
                "head_id": f"L{layer}_H{head}",
                "tokens": toks,
                "head_type": "UNKNOWN",
                "attention_matrix": [],
                "status": "FAILED",
                "error": f"Failed to compute attention patterns: {e}",
            }

    def _classify_head_type(self, matrix: List[List[float]], head: int) -> str:
        """Classify attention head type based on pattern."""
        if not matrix or len(matrix) < 2:
            return "UNKNOWN"
        
        seq_len = len(matrix)
        
        # Check for previous-token head: high attention to position i-1
        prev_token_score = 0.0
        for i in range(1, seq_len):
            if matrix[i][i-1] > 0.3:
                prev_token_score += matrix[i][i-1]
        
        if prev_token_score > 0.5:
            return "Previous-Token Head"
        
        # Check for induction head: pattern A B ... A B (induction pattern)
        # Simplified: look for high attention from later positions to earlier similar positions
        induction_score = 0.0
        for i in range(2, seq_len):
            for j in range(i-2, -1, -1):
                if matrix[i][j] > 0.2:
                    induction_score += matrix[i][j]
        
        if induction_score > 0.4:
            return "Induction Head"
        
        # Check for name mover head (high attention to final token)
        if seq_len > 1:
            final_token_attn = sum(matrix[i][-1] for i in range(seq_len)) / seq_len
            if final_token_attn > 0.3:
                return "Name Mover Head"
        
        return "General Head"
