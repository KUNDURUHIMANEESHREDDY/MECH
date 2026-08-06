from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch

warnings.filterwarnings("ignore")

from transformer_lens import HookedTransformer


class GPT2Model:
    """Wraps GPT-2 Small via TransformerLens for real model inference.

    Provides tokenization, forward pass, logit extraction, and IOI analysis
    using the actual model — no hardcoded token IDs or synthetic values.
    """

    def __init__(self, model_name: str = "gpt2-small"):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = HookedTransformer.from_pretrained(
            model_name, device=self._device
        )
        self._model.eval()
        self._prompt: str = ""
        self._tokens: torch.Tensor | None = None
        self._logits: np.ndarray | None = None
        self._cache: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------ #
    #  Model config properties
    # ------------------------------------------------------------------ #

    @property
    def num_layers(self) -> int:
        return self._model.cfg.n_layers

    @property
    def num_heads(self) -> int:
        return self._model.cfg.n_heads

    @property
    def hidden_dim(self) -> int:
        return self._model.cfg.d_model

    @property
    def vocab_size(self) -> int:
        return self._model.cfg.d_vocab

    @property
    def device(self) -> str:
        return self._device

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def str_tokens(self) -> list[str]:
        return self._model.to_str_tokens(self._prompt) if self._prompt else []

    @property
    def token_ids(self) -> list[int]:
        return self._tokens[0].tolist() if self._tokens is not None else []

    # ------------------------------------------------------------------ #
    #  Forward pass
    # ------------------------------------------------------------------ #

    def run(self, prompt: str) -> GPT2Model:
        """Run a forward pass on the prompt, caching all activations."""
        self._prompt = prompt
        self._tokens = self._model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits, cache = self._model.run_with_cache(self._tokens)
        self._logits = logits[0].cpu().numpy()
        self._cache = cache
        return self

    # ------------------------------------------------------------------ #
    #  Tokenization helpers
    # ------------------------------------------------------------------ #

    def to_token_id(self, token_str: str) -> int:
        return self._model.to_single_token(token_str)

    def to_token_str(self, token_id: int) -> str:
        return self._model.to_string(token_id)

    def tokenize(self, text: str) -> list[str]:
        return self._model.to_str_tokens(text)

    # ------------------------------------------------------------------ #
    #  Logit access
    # ------------------------------------------------------------------ #

    @property
    def last_position_logits(self) -> np.ndarray | None:
        """Logits at the last token position: shape (vocab_size,)."""
        if self._logits is None:
            return None
        return self._logits[-1, :]

    def top_k_predictions(self, k: int = 20) -> list[dict[str, Any]]:
        """Return the top-k predicted tokens from the last position."""
        logits = self.last_position_logits
        if logits is None:
            return []
        stable = logits - np.max(logits)
        exp = np.exp(stable)
        probs = exp / np.sum(exp)
        top_indices = np.argsort(logits)[::-1][:k]
        return [
            {
                "token_id": int(idx),
                "token_str": self._model.to_string(int(idx)),
                "logit": float(logits[idx]),
                "probability": float(probs[idx]),
            }
            for idx in top_indices
        ]

    def token_logit(self, token_str: str) -> dict[str, Any]:
        """Return the logit for a specific token string at the last position."""
        token_id = self.to_token_id(token_str)
        logits = self.last_position_logits
        if logits is None or token_id >= len(logits):
            return {"token": token_str, "token_id": token_id, "logit": None}
        return {
            "token": token_str,
            "token_id": token_id,
            "logit": float(logits[token_id]),
            "token_str_detokenized": self._model.to_string(token_id),
        }

    # ------------------------------------------------------------------ #
    #  Text generation
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        """Greedy text continuation.

        Uses greedy decoding (do_sample=False) which is deterministic
        — the same prompt always yields the same continuation. For a
        124 M parameter GPT-2 this is fast and reproducible, though
        longer runs may repeat. For more varied output, callers should
        sample token-by-token via the inference engine in
        ``backend.services.gpt2_engine``.
        """
        with torch.no_grad():
            return self._model.generate(
                prompt, max_new_tokens=max_new_tokens, do_sample=False
            )

    # ------------------------------------------------------------------ #
    #  IOI Analysis
    # ------------------------------------------------------------------ #

    def ioi_analysis(self) -> dict[str, Any]:
        """Indirect Object Identification using the standard IOI templates.

        All token IDs and logits come from the actual model.
        No hardcoded IDs or synthetic values.
        """
        clean_prompt = "When Mary and John went to the store, John gave the bag to"
        corr_prompt = "When John and Mary went to the store, Mary gave the bag to"

        mary_id = self.to_token_id(" Mary")
        john_id = self.to_token_id(" John")

        def _ld(prompt: str) -> dict[str, Any]:
            toks = self._model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                lgts = self._model(toks)
            last = lgts[0, -1, :].cpu().numpy()
            mary_l = float(last[mary_id])
            john_l = float(last[john_id])
            return {
                "mary_logit": mary_l,
                "john_logit": john_l,
                "logit_difference": mary_l - john_l,
                "mary_greater": mary_l > john_l,
                "top1_token_id": int(last.argmax()),
                "top1_token_str": self._model.to_string(int(last.argmax())),
            }

        return {
            "clean_prompt": clean_prompt,
            "corrupted_prompt": corr_prompt,
            "clean": _ld(clean_prompt),
            "corrupted": _ld(corr_prompt),
        }

    # ------------------------------------------------------------------ #
    #  Activation / Cache access
    # ------------------------------------------------------------------ #

    def get_cache(self, hook_name: str) -> np.ndarray | None:
        """Get a cached activation by hook name."""
        if self._cache is None:
            return None
        return self._cache[hook_name][0].cpu().numpy()

    def cache_shapes(self) -> dict[str, dict[str, dict]]:
        """Return shapes and basic stats of all cache entries per layer.

        Includes: resid_pre, resid_mid, resid_post, attn_out, mlp_out,
        pattern, z — directly from the cache, no post-processing.
        """
        hook_names = [
            "hook_resid_pre",
            "hook_resid_mid",
            "hook_resid_post",
            "hook_attn_out",
            "hook_mlp_out",
            "attn.hook_pattern",
            "attn.hook_z",
        ]
        result: dict[str, dict[str, dict]] = {}
        for i in range(self.num_layers):
            block = f"blocks.{i}"
            layer_result: dict[str, dict] = {}
            for hname in hook_names:
                full = f"{block}.{hname}"
                arr = self.get_cache(full)
                if arr is not None:
                    layer_result[hname] = {
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                        "min": round(float(arr.min()), 6),
                        "max": round(float(arr.max()), 6),
                        "mean": round(float(arr.mean()), 6),
                    }
            result[block] = layer_result
        return result

    def collect_all_layer_data(self) -> dict[str, dict[str, np.ndarray]]:
        """Return activations, attention, and residual for all layers."""
        data: dict[str, dict[str, np.ndarray]] = {}
        for i in range(self.num_layers):
            name = f"blocks.{i}"
            data[name] = {
                "activations": self.get_cache(f"{name}.hook_mlp_out"),
                "attention": self.get_cache(f"{name}.attn.hook_pattern"),
                "residual": self.get_cache(f"{name}.hook_resid_post"),
            }
        return data

    # ------------------------------------------------------------------ #
    #  Attention pattern viewer (raw from cache)
    # ------------------------------------------------------------------ #

    def attention_pattern(self, layer: int, head: int) -> dict[str, Any]:
        """Return the raw attention pattern for a specific layer and head.

        Directly from cache['pattern', layer] — no normalization.
        """
        if self._cache is None:
            return {"error": "No cache. Run a prompt first."}

        pattern_key = f"blocks.{layer}.attn.hook_pattern"
        if pattern_key not in self._cache:
            return {"error": f"No pattern for layer {layer}"}

        pattern = self._cache[pattern_key][0, head].cpu().numpy()  # (seq_len, seq_len)
        return {
            "layer": layer,
            "head": head,
            "shape": list(pattern.shape),
            "min": float(pattern.min()),
            "max": float(pattern.max()),
            "mean": float(pattern.mean()),
            "matrix": pattern.tolist(),
            "source": f"cache['blocks.{layer}.attn.hook_pattern'][0, {head}]",
        }

    # ------------------------------------------------------------------ #
    #  Zero ablation (single or multi-head)
    # ------------------------------------------------------------------ #

    def run_with_head_ablation(
        self, prompt: str, layer: int, head: int
    ) -> dict[str, Any]:
        """Run forward pass with one head's z (value*attention) zeroed.

        Verifies the hook actually affects the computation by comparing
        clean vs ablated logits.
        """
        return self.run_with_multi_ablation(prompt, [(layer, head)])

    def run_with_multi_ablation(
        self, prompt: str, heads: list[tuple[int, int]]
    ) -> dict[str, Any]:
        """Run forward pass with multiple heads' z zeroed.

        heads: list of (layer, head_index) tuples.

        Returns clean vs ablated logits, top predictions, and whether
        the output actually changed.
        """
        # Clean run
        self.run(prompt)
        clean_logits = self.last_position_logits
        clean_top1 = int(clean_logits.argmax()) if clean_logits is not None else -1

        # Build a set of (layer, head) per layer for efficient hooking
        layer_heads: dict[int, list[int]] = {}
        for l, h in heads:
            layer_heads.setdefault(l, []).append(h)

        def make_hook(layer: int, head_list: list[int]):
            def _zero_hook(z: torch.Tensor, **kwargs: object) -> torch.Tensor:
                for h in head_list:
                    z[:, :, h, :] = 0.0
                return z
            return _zero_hook

        fwd_hooks = [
            (f"blocks.{l}.attn.hook_z", make_hook(l, hl))
            for l, hl in layer_heads.items()
        ]

        tokens = self._model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            ablated = self._model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        ablated_logits = ablated[0, -1, :].cpu().numpy()
        ablated_top1 = int(ablated_logits.argmax())

        same_top1 = clean_top1 == ablated_top1
        max_diff = float(np.max(np.abs(ablated_logits - clean_logits))) if clean_logits is not None else 0.0

        mary_id = self.to_token_id(" Mary")
        john_id = self.to_token_id(" John")

        def _extract(logits, top1_id):
            return {
                "top1_token_id": top1_id,
                "top1_token_str": self._model.to_string(top1_id) if top1_id >= 0 else "",
                "top1_logit": float(logits[top1_id]) if logits is not None else None,
                "mary_logit": float(logits[mary_id]) if logits is not None and mary_id < len(logits) else None,
                "john_logit": float(logits[john_id]) if logits is not None and john_id < len(logits) else None,
            }

        return {
            "prompt": prompt,
            "heads": [f"L{l}H{h}" for l, h in heads],
            "clean": _extract(clean_logits, clean_top1),
            "ablated": _extract(ablated_logits, ablated_top1),
            "same_top1_prediction": same_top1,
            "max_logit_difference": max_diff,
            "logits_changed": max_diff > 1e-6,
        }

    # ------------------------------------------------------------------ #
    #  Activation Patching Matrix — Δ logit diff per (layer, head)
    # ------------------------------------------------------------------ #

    def compute_patching_matrix(self, prompt: str) -> dict[str, Any]:
        """Build a 12×12 grid of Δ logit diff for each (layer, head).

        Each cell = |clean_logit_diff - ablated_logit_diff| when that
        head's z is zeroed.  Blue = little effect, Red = large effect.

        Uses batched forward passes (all 12 heads per layer at once)
        for speed: 12 forward passes instead of 144.
        """
        self.run(prompt)
        mary_id = self.to_token_id(" Mary")
        john_id = self.to_token_id(" John")
        clean_last = self.last_position_logits
        clean_ld = float(clean_last[mary_id] - clean_last[john_id])

        n_heads = self.num_heads
        tokens = self._model.to_tokens(prompt, prepend_bos=True)  # (1, seq)
        batch_tokens = tokens.expand(n_heads, -1).contiguous()     # (n_heads, seq)

        matrix = [[0.0] * n_heads for _ in range(self.num_layers)]

        for layer in range(self.num_layers):
            # One batched forward pass: batch element h zeros head h
            def _batch_hook(z: torch.Tensor, **kwargs: object) -> torch.Tensor:
                for h in range(n_heads):
                    z[h, :, h, :] = 0.0
                return z

            with torch.no_grad():
                out = self._model.run_with_hooks(
                    batch_tokens,
                    fwd_hooks=[(f"blocks.{layer}.attn.hook_z", _batch_hook)],
                )
            for head in range(n_heads):
                ablated = out[head, -1, :].cpu().numpy()
                ablated_ld = float(ablated[mary_id] - ablated[john_id])
                delta = round(abs(clean_ld - ablated_ld), 4)
                matrix[layer][head] = delta

        flat = [v for row in matrix for v in row]
        vmin = min(flat) if flat else 0.0
        vmax = max(flat) if flat else 1.0

        return {
            "prompt": prompt,
            "clean_logit_diff": round(clean_ld, 4),
            "matrix": matrix,
            "layers": self.num_layers,
            "heads": self.num_heads,
            "vmin": vmin,
            "vmax": vmax,
        }

    # ------------------------------------------------------------------ #
    #  Logit Lens — residual → unembed at every layer
    # ------------------------------------------------------------------ #

    # Category keywords for semantic classification (lowercased)
    _CATEGORIES: dict[str, list[str]] = {
        "Geography": [
            "france", "paris", "germany", "berlin", "italy", "rome", "spain", "madrid",
            "china", "beijing", "japan", "tokyo", "india", "delhi", "london", "england",
            "united kingdom", "uk", "usa", "america", "washington", "new york", "california",
            "capital", "city", "country", "russia", "moscow", "brazil", "australia", "europe",
            "asia", "africa", "ocean", "mountain", "river", "lake", "island", "empire",
            "nation", "state", "province", "region", "border", "map", "continent",
        ],
        "History": [
            "war", "battle", "king", "queen", "ancient", "medieval", "century", "empire",
            "revolution", "treaty", "president", "prime minister", "dictator", "dynasty",
            "colonel", "general", "invasion", "independence", "civil war", "cold war",
            "world war", "wwi", "wwii", "roman", "greek", "egyptian", "nazi", "soviet",
            "fascist", "democracy", "monarchy", "constitution", "republic", "empire",
        ],
        "Science": [
            "physics", "chemistry", "biology", "quantum", "molecule", "atom", "electron",
            "gravity", "force", "energy", "light", "wave", "particle", "experiment",
            "theory", "law", "equation", "formula", "reaction", "element", "compound",
            "cell", "dna", "gene", "evolution", "species", "organism", "bacteria", "virus",
            "protein", "enzyme", "telescope", "microscope", "scientist", "laboratory",
            "climate", "temperature", "pressure", "velocity", "acceleration",
        ],
        "Mathematics": [
            "number", "zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "hundred", "thousand", "million", "billion",
            "calculate", "computation", "algorithm", "equation", "theorem", "proof",
            "geometry", "algebra", "calculus", "statistics", "probability", "matrix",
            "vector", "integer", "fraction", "decimal", "prime", "square", "cube",
            "root", "sum", "difference", "product", "quotient", "modulo", "graph",
            "ratio", "percentage", "count", "add", "subtract", "multiply", "divide",
        ],
        "Programming": [
            "python", "javascript", "java", "c++", "rust", "golang", "typescript",
            "html", "css", "function", "variable", "class", "object", "array", "string",
            "integer", "boolean", "loop", "if else", "return", "import", "def", "const",
            "let", "var", "async", "await", "promise", "callback", "api", "http",
            "algorithm", "data structure", "json", "xml", "database", "sql", "server",
            "code", "compile", "debug", "error", "exception", "thread", "process",
            "memory", "pointer", "recursion", "stack", "queue", "tree", "hash",
        ],
        "People & Names": [
            "mary", "john", "bob", "alice", "james", "robert", "michael", "david",
            "william", "richard", "joseph", "thomas", "charles", "george", "henry",
            "susan", "linda", "barbara", "elizabeth", "jennifer", "maria", "patricia",
            "sarah", "karen", "nancy", "lisa", "betty", "margaret", "sandra", "ashley",
            "dorothy", "kimberly", "emily", "helen", "amy", "donna", "deborah", "carol",
            "person", "people", "man", "woman", "child", "baby", "boy", "girl",
            "mister", "mr", "mrs", "ms", "dr", "professor", "president",
        ],
        "Entertainment": [
            "movie", "film", "music", "song", "album", "concert", "actor", "actress",
            "director", "producer", "artist", "band", "guitar", "piano", "drum",
            "dance", "theater", "cinema", "hollywood", "celebrity", "famous",
            "book", "novel", "author", "writer", "poem", "poetry", "fiction",
            "game", "sport", "football", "soccer", "basketball", "baseball", "tennis",
            "olympic", "champion", "tournament", "league", "player", "coach",
        ],
        "Food & Drink": [
            "food", "water", "bread", "rice", "pasta", "meat", "chicken", "beef",
            "pork", "fish", "vegetable", "fruit", "apple", "banana", "orange",
            "coffee", "tea", "wine", "beer", "milk", "juice", "sugar", "salt",
            "butter", "cheese", "egg", "soup", "salad", "sauce", "chocolate",
            "cake", "cookie", "pizza", "burger", "sandwich", "breakfast", "lunch",
            "dinner", "restaurant", "kitchen", "cook", "recipe", "ingredient",
        ],
        "Technology": [
            "computer", "software", "hardware", "internet", "website", "email",
            "phone", "smartphone", "iphone", "android", "windows", "linux", "mac",
            "google", "facebook", "twitter", "amazon", "microsoft", "apple",
            "robot", "ai", "artificial intelligence", "machine learning", "data",
            "network", "server", "cloud", "digital", "electronic", "device",
            "screen", "keyboard", "mouse", "battery", "charger", "cable",
            "application", "platform", "system", "technology", "innovation",
        ],
        "Nature": [
            "tree", "flower", "plant", "forest", "garden", "grass", "leaf", "seed",
            "animal", "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep",
            "lion", "tiger", "bear", "wolf", "deer", "rabbit", "snake", "eagle",
            "sky", "sun", "moon", "star", "cloud", "rain", "snow", "wind", "storm",
            "earth", "ground", "rock", "sand", "sea", "ocean", "river", "lake",
            "garden", "field", "mountain", "valley", "desert", "island",
        ],
    }

    def logit_lens(
        self, prompt: str, top_k: int = 5, target_token: str | None = None
    ) -> dict[str, Any]:
        """For each layer, apply the unembedding to the residual stream.

        Shows what the model would predict if it stopped at each layer.
        If target_token is given, tracks its rank and logit per layer.
        """
        self.run(prompt)

        target_id = None
        if target_token is not None:
            target_id = self._model.to_single_token(target_token)
            if target_id is None or target_id == -1:
                target_id = self._model.to_single_token(" " + target_token)
            if target_id is None or target_id == -1:
                target_id = self._model.tokenizer.encode(target_token)[0]

        layers_data: list[dict[str, Any]] = []
        resid_vectors: list[np.ndarray] = []
        target_data: list[dict[str, Any]] = []

        for layer in range(self.num_layers):
            resid = self.get_cache(f"blocks.{layer}.hook_resid_post")  # (seq, d_model)
            last_resid = resid[-1, :]  # last token position
            resid_vectors.append(last_resid)

            # Apply ln_final + unembed to get logits at this layer
            resid_t = torch.from_numpy(last_resid).to(self._device)
            normalized = self._model.ln_final(resid_t)
            logits_t = self._model.unembed(normalized)
            logits_np = logits_t.detach().cpu().numpy()

            # Stable softmax
            stable = logits_np - np.max(logits_np)
            probs = np.exp(stable) / np.sum(np.exp(stable))
            top_idx = np.argsort(logits_np)[::-1][:top_k]

            predictions = [
                {
                    "token_str": self._model.to_string(int(idx)),
                    "token_id": int(idx),
                    "logit": round(float(logits_np[idx]), 4),
                    "probability": round(float(probs[idx]), 6),
                }
                for idx in top_idx
            ]

            layers_data.append(
                {
                    "layer": layer,
                    "predictions": predictions,
                }
            )

            # Track target token across layers
            if target_id is not None:
                t_logit = float(logits_np[target_id])
                t_prob = float(probs[target_id])
                argsort_desc = np.argsort(logits_np)[::-1]
                rank = int(np.where(argsort_desc == target_id)[0][0]) + 1
                target_data.append(
                    {
                        "layer": layer,
                        "logit": round(t_logit, 4),
                        "probability": round(t_prob, 6),
                        "rank": rank,
                        "token_id": int(target_id),
                    }
                )

        # Cosine similarity to final layer across layers
        final_resid = resid_vectors[-1]
        final_norm = np.linalg.norm(final_resid) + 1e-12
        cos_sims = []
        for i, vec in enumerate(resid_vectors):
            sim = float(np.dot(vec, final_resid) / (np.linalg.norm(vec) + 1e-12) / final_norm)
            cos_sims.append(round(sim, 6))

        result: dict[str, Any] = {
            "prompt": prompt,
            "tokens": self.str_tokens,
            "layers": layers_data,
            "cosine_similarity_to_final": cos_sims,
            "d_model": self.hidden_dim,
        }
        if target_data:
            result["target_data"] = target_data
            result["interpretation"] = self._interpret_logit_lens(target_data)
        # Always compute category scores from the predictions
        result["category_scores"] = self._compute_category_scores(layers_data)
        return result

    @staticmethod
    def _interpret_logit_lens(target_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze target token rank progression and generate an explanation."""
        ranks = [td["rank"] for td in target_data]
        n_layers = len(ranks)
        final_rank = ranks[-1]

        # Find first time token enters each threshold
        def _first_at_or_below(threshold: int) -> int | None:
            for td in target_data:
                if td["rank"] <= threshold:
                    return td["layer"]
            return None

        entry_top100 = _first_at_or_below(100)
        entry_top50 = _first_at_or_below(50)
        entry_top10 = _first_at_or_below(10)
        entry_top5 = _first_at_or_below(5)
        reached_rank1 = _first_at_or_below(1)

        # Biggest single-layer rank improvement
        jumps: list[tuple[int, int, int]] = []
        for i in range(1, n_layers):
            drop = ranks[i - 1] - ranks[i]  # positive = rank improved
            if drop > 0:
                jumps.append((i, i - 1, drop))
        jumps.sort(key=lambda x: -x[2])

        # Early-layer behavior (layers 0-3)
        early_flat = all(r > 40000 for r in ranks[:4])

        # Build narrative segments
        segments: list[str] = []

        # 1) Early layers
        if early_flat:
            start_active = entry_top100 or entry_top50 or entry_top10
            if start_active is not None and start_active > 0:
                segments.append(
                    f"Layers 0–{start_active - 1} show little evidence "
                    f'for the target token (rank ~{ranks[min(start_active - 1, n_layers - 1)]:,}).'
                )
            else:
                segments.append(
                    f"Early layers (0–3) show no meaningful evidence "
                    f"for the target token (rank ~{ranks[0]:,} of 50,257)."
                )
        else:
            segments.append(
                f"The target token begins to emerge early, appearing in "
                f"the top 100 by layer {entry_top100 or '?'}."
            )

        # 2) Middle layers — consolidation
        if entry_top10 and entry_top100 and entry_top10 > entry_top100:
            rise_start = entry_top100
            rise_end = entry_top10
            if jumps:
                biggest_layer, _, _ = jumps[0]
                if rise_start is not None and rise_end is not None:
                    segments.append(
                        f"Between layers {rise_start} and {rise_end} "
                        f"the token climbs rapidly into the top 10, indicating "
                        f"factual information is consolidated in mid-to-late "
                        f"transformer blocks."
                    )
                    if biggest_layer is not None:
                        segments.append(
                            f"The largest single-layer jump occurs at layer "
                            f"{biggest_layer} (rank improvement of "
                            f"{jumps[0][2]:,} positions)."
                        )

        # 3) Final rank summary
        if reached_rank1 is not None:
            segments.append(
                f"By layer {reached_rank1} it reaches rank #1, "
                f"becoming the model's most likely completion."
            )
        elif entry_top5 is not None:
            segments.append(
                f"It enters the top 5 at layer {entry_top5} and "
                f"finishes at rank #{final_rank}."
            )
        elif entry_top10 is not None:
            segments.append(
                f"It reaches the top 10 (rank #{final_rank}) but does not "
                f"enter the top 5, suggesting other tokens compete strongly."
            )
        elif entry_top50 is not None:
            segments.append(
                f"It peaks at rank #{min(ranks)} but never enters the "
                f"top 10, indicating the model considers but does not "
                f"strongly favor this token."
            )
        else:
            segments.append(
                f"The target token never rises above rank ~{min(ranks):,}, "
                f"suggesting the model does not associate this completion "
                f"with the prompt."
            )

        # 4) Pattern classification
        if reached_rank1 is not None and (reached_rank1 > n_layers * 0.6 if n_layers > 0 else True):
            pattern = "late_emergence"
        elif early_flat and final_rank <= 10:
            pattern = "late_emergence"
        elif final_rank <= 5:
            pattern = "strong_prediction"
        elif final_rank <= 100:
            pattern = "weak_prediction"
        else:
            pattern = "no_prediction"

        return {
            "summary": " ".join(segments),
            "pattern": pattern,
            "milestones": {
                "first_top_100": entry_top100,
                "first_top_10": entry_top10,
                "first_top_5": entry_top5,
                "reached_rank_1": reached_rank1,
            },
            "final_rank": final_rank,
            "biggest_jump": {
                "from_layer": jumps[0][1] if jumps else None,
                "to_layer": jumps[0][0] if jumps else None,
                "positions": jumps[0][2] if jumps else None,
            }
            if jumps
            else None,
        }

    @classmethod
    def _compute_category_scores(cls, layers_data: list[dict[str, Any]]) -> dict[str, float]:
        """Score semantic categories based on top-k predictions across layers.

        For each layer, matches predicted token strings against category keywords.
        Later layers are weighted more heavily. Returns confidence scores (0-1).
        """
        n_layers = len(layers_data)
        if n_layers == 0:
            return {}

        category_counts: dict[str, float] = {cat: 0.0 for cat in cls._CATEGORIES}
        total_weight = 0.0

        for layer_idx, layer in enumerate(layers_data):
            weight = 1.0 + layer_idx  # later layers matter more
            total_weight += weight
            for pred in layer["predictions"]:
                token_lower = pred["token_str"].lower().strip()
                # Check if any category keyword matches
                for cat, keywords in cls._CATEGORIES.items():
                    for kw in keywords:
                        if kw in token_lower or token_lower in kw:
                            category_counts[cat] += weight * (pred["probability"] * 2.0)
                            break

        # Normalize by total weight and cap at 1.0
        scores: dict[str, float] = {}
        for cat in cls._CATEGORIES:
            raw = category_counts[cat]
            scores[cat] = round(min(1.0, raw / max(total_weight, 1e-12)), 4)

        # Sort descending
        return dict(sorted(scores.items(), key=lambda x: -x[1]))

    # ------------------------------------------------------------------ #
    #  Neuron Inspector — activation values, histogram, ablation
    # ------------------------------------------------------------------ #

    def neuron_inspect(
        self,
        prompt: str,
        layer: int,
        neuron_index: int,
        target_token: str | None = None,
    ) -> dict[str, Any]:
        """Inspect a single neuron across all tokens in the prompt.

        Returns per-token activations, histogram, strongest token,
        percentile within the layer, and optional ablation effect
        on a target token's logit.
        """
        self.run(prompt)
        n_tokens = len(self.str_tokens)
        d_mlp = self._model.cfg.d_mlp

        # Per-token activations for this neuron
        activations_all = self.get_cache(f"blocks.{layer}.mlp.hook_post")  # (seq, d_mlp)
        neuron_acts = activations_all[:, neuron_index]  # (seq,)

        # Per-token activation values with token strings
        token_acts: list[dict[str, Any]] = []
        for i in range(n_tokens):
            token_acts.append({
                "token_index": i,
                "token_str": self.str_tokens[i],
                "activation": round(float(neuron_acts[i]), 6),
            })

        # Strongest token
        max_idx = int(np.argmax(neuron_acts))
        strongest_token: dict[str, Any] = token_acts[max_idx]

        # Statistics
        mean_act = float(neuron_acts.mean())
        std_act = float(neuron_acts.std())
        max_act = float(neuron_acts.max())
        min_act = float(neuron_acts.min())
        sparsity = float((neuron_acts == 0).sum() / n_tokens)

        # Histogram (20 bins)
        hist_counts, hist_edges = np.histogram(neuron_acts, bins=20)
        histogram = [
            {"bin_start": round(float(hist_edges[i]), 6),
             "bin_end": round(float(hist_edges[i + 1]), 6),
             "count": int(hist_counts[i])}
            for i in range(len(hist_counts))
        ]

        # Percentile within layer (last token position)
        all_neurons_last = activations_all[-1, :]  # (d_mlp,)
        percentile = float(
            (all_neurons_last > all_neurons_last[neuron_index]).sum() / len(all_neurons_last) * 100
        )
        # ^ percentage of neurons in this layer with higher activation than this one

        # Ablation effect on target token
        ablation_effect: dict[str, Any] | None = None
        if target_token is not None:
            target_id = self._model.to_single_token(target_token)
            if target_id is None or target_id == -1:
                target_id = self._model.to_single_token(" " + target_token)
            if target_id is None or target_id == -1:
                target_id = self._model.tokenizer.encode(target_token)[0]

            if target_id is not None:
                clean_logit = float(self.last_position_logits[target_id]) if self.last_position_logits is not None else None

                # Ablate: zero out this neuron's activation
                def _ablate_hook(post: torch.Tensor, **kwargs: object) -> torch.Tensor:
                    post[:, :, neuron_index] = 0.0
                    return post

                tokens = self._model.to_tokens(prompt, prepend_bos=True)
                with torch.no_grad():
                    ablated_out = self._model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", _ablate_hook)],
                    )
                ablated_logit = float(ablated_out[0, -1, target_id].cpu().numpy()) if target_id is not None else None

                if clean_logit is not None and ablated_logit is not None:
                    ablation_effect = {
                        "target_token": target_token,
                        "target_id": int(target_id),
                        "clean_logit": round(clean_logit, 4),
                        "ablated_logit": round(ablated_logit, 4),
                        "delta": round(ablated_logit - clean_logit, 4),
                        "abs_delta": round(abs(ablated_logit - clean_logit), 4),
                    }

        return {
            "prompt": prompt,
            "layer": layer,
            "neuron_index": neuron_index,
            "tokens": self.str_tokens,
            "d_mlp": d_mlp,
            "token_activations": token_acts,
            "strongest_token": strongest_token,
            "statistics": {
                "mean": round(mean_act, 6),
                "std": round(std_act, 6),
                "max": round(max_act, 6),
                "min": round(min_act, 6),
                "sparsity": round(sparsity, 6),
            },
            "histogram": histogram,
            "percentile_within_layer": round(percentile, 2),
            "ablation_effect": ablation_effect,
        }

    # ------------------------------------------------------------------ #
    #  Neuron Search — top-K neurons by activation for a prompt
    # ------------------------------------------------------------------ #

    def neuron_search(
        self, prompt: str, layer: int, top_k: int = 20
    ) -> dict[str, Any]:
        """Find the top-K most active neurons in a layer for a given prompt.

        Each neuron is scored by its max activation across all token
        positions. Returns the neuron index, which token fired it most,
        and summary stats.
        """
        self.run(prompt)
        d_mlp = self._model.cfg.d_mlp

        activations_all = self.get_cache(f"blocks.{layer}.mlp.hook_post")  # (seq, d_mlp)

        max_acts: np.ndarray = activations_all.max(axis=0)      # (d_mlp,)
        mean_acts: np.ndarray = activations_all.mean(axis=0)    # (d_mlp,)
        argmax_tokens: np.ndarray = activations_all.argmax(axis=0)  # (d_mlp,)

        # Sort descending by max activation
        sorted_indices = np.argsort(max_acts)[::-1][:top_k]

        neurons: list[dict[str, Any]] = []
        for idx in sorted_indices:
            tok_idx = int(argmax_tokens[idx])
            neurons.append(
                {
                    "neuron_index": int(idx),
                    "max_activation": round(float(max_acts[idx]), 6),
                    "mean_activation": round(float(mean_acts[idx]), 6),
                    "strongest_token_index": tok_idx,
                    "strongest_token_str": self.str_tokens[tok_idx],
                }
            )

        return {
            "prompt": prompt,
            "layer": layer,
            "tokens": self.str_tokens,
            "d_mlp": d_mlp,
            "top_k": top_k,
            "neurons": neurons,
        }

    # ------------------------------------------------------------------ #
    #  Correlated Neurons — find neurons that fire together across tokens
    #  (Pearson correlation over per-token activation vectors)
    # ------------------------------------------------------------------ #

    def correlated_neurons(
        self, prompt: str, layer: int, neuron_index: int, top_k: int = 20
    ) -> dict[str, Any]:
        """Find neurons across all layers whose activation pattern
        correlates with the given neuron (Pearson r over token positions).
        """
        self.run(prompt)

        target = self.get_cache(f"blocks.{layer}.mlp.hook_post")[:, neuron_index]
        t_centered = target - target.mean()
        t_norm = np.linalg.norm(t_centered)

        if t_norm < 1e-10:
            return {"error": "Target neuron has constant activation (zero variance)."}

        results: list[dict[str, Any]] = []

        for l in range(self.num_layers):
            acts = self.get_cache(f"blocks.{l}.mlp.hook_post")  # (seq, d_mlp)
            means = acts.mean(axis=0)
            centered = acts - means
            norms = np.linalg.norm(centered, axis=0)
            valid = norms > 1e-10

            if not valid.any():
                continue

            corrs = np.zeros(acts.shape[1], dtype=np.float64)
            corrs[valid] = (t_centered @ centered[:, valid]) / (t_norm * norms[valid])

            for n in range(acts.shape[1]):
                if l == layer and n == neuron_index:
                    continue
                if valid[n]:
                    r = float(corrs[n])
                    results.append({
                        "layer": l,
                        "neuron_index": n,
                        "correlation": round(r, 6),
                        "abs_correlation": round(abs(r), 6),
                    })

        results.sort(key=lambda x: -x["abs_correlation"])
        top = results[:top_k]

        # Compute causal scores for the top-k correlated neurons
        tokens_tensor = self._model.to_tokens(prompt, prepend_bos=True)
        for entry in top:
            score = self._causal_dependence(
                tokens_tensor, layer, neuron_index,
                entry["layer"], entry["neuron_index"]
            )
            entry["causal_score"] = score

        return {
            "prompt": prompt,
            "source": {"layer": layer, "neuron_index": neuron_index},
            "tokens": self.str_tokens,
            "top_k": top_k,
            "correlated_neurons": top,
        }

    def _causal_dependence(
        self,
        tokens: torch.Tensor,
        src_layer: int,
        src_neuron: int,
        tgt_layer: int,
        tgt_neuron: int,
    ) -> float:
        """Measure causal dependence: ablate source neuron, measure target's activation change.

        Runs two forward passes — clean and ablated — and returns
        how much the target activation changes (0 = no effect, 1 = disappears).
        """
        # Clean: get target neuron activation
        clean_cache: dict[str, torch.Tensor] = {}

        def _clean_cache_hook(activations: torch.Tensor, hook: object, **kwargs: object) -> None:
            clean_cache["tgt"] = activations[0, :, tgt_neuron].clone().detach()
            clean_cache["src"] = activations[0, :, src_neuron].clone().detach()

        with torch.no_grad():
            self._model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (f"blocks.{tgt_layer}.mlp.hook_post", _clean_cache_hook),
                ],
                reset_hooks_end=False,
            )

        tgt_clean = clean_cache.get("tgt")
        src_clean = clean_cache.get("src")
        if tgt_clean is None or src_clean is None:
            return 0.0

        tgt_clean_mean = float(tgt_clean.mean().cpu().numpy())
        src_clean_mean = float(src_clean.mean().cpu().numpy())

        if abs(tgt_clean_mean) < 1e-10:
            return 0.0

        # Ablated: zero source neuron, measure target
        def _ablate_hook(post: torch.Tensor, **kwargs: object) -> torch.Tensor:
            post[:, :, src_neuron] = 0.0
            return post

        ablated_cache: dict[str, torch.Tensor] = {}

        def _ablated_cache_hook(activations: torch.Tensor, hook: object, **kwargs: object) -> None:
            ablated_cache["tgt"] = activations[0, :, tgt_neuron].clone().detach()

        with torch.no_grad():
            self._model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (f"blocks.{src_layer}.mlp.hook_post", _ablate_hook),
                    (f"blocks.{tgt_layer}.mlp.hook_post", _ablated_cache_hook),
                ],
                reset_hooks_end=False,
            )

        tgt_ablated = ablated_cache.get("tgt")
        if tgt_ablated is None:
            return 0.0

        tgt_ablated_mean = float(tgt_ablated.mean().cpu().numpy())

        # Causal score: how much did the target change relative to its original?
        diff = abs(tgt_ablated_mean - tgt_clean_mean)
        causal = min(1.0, diff / (abs(tgt_clean_mean) + 1e-10))

        return round(causal, 4)

    # ------------------------------------------------------------------ #
    #  Neuron Evolution — a neuron's activation across layers for a token
    # ------------------------------------------------------------------ #

    def neuron_evolution(
        self, prompt: str, layer: int, neuron_index: int, token_index: int
    ) -> dict[str, Any]:
        """For a fixed token position and neuron index, show the neuron's
        activation in each layer.  Answers: 'At which layer does this
        feature activate for this token?'.
        """
        self.run(prompt)

        per_layer: list[dict[str, Any]] = []
        for l in range(self.num_layers):
            act = self.get_cache(f"blocks.{l}.mlp.hook_post")[token_index, neuron_index]
            per_layer.append(
                {
                    "layer": l,
                    "activation": round(float(act), 6),
                }
            )

        max_act = max(p["activation"] for p in per_layer) if per_layer else 0.0
        min_act = min(p["activation"] for p in per_layer) if per_layer else 0.0

        return {
            "prompt": prompt,
            "source": {"layer": layer, "neuron_index": neuron_index, "token_index": token_index},
            "token_str": self.str_tokens[token_index] if token_index < len(self.str_tokens) else "?",
            "tokens": self.str_tokens,
            "layers": per_layer,
            "max_activation": round(max_act, 6),
            "min_activation": round(min_act, 6),
        }

    # ------------------------------------------------------------------ #
    #  Dataset Activation — batch prompts to profile a neuron's behavior
    # ------------------------------------------------------------------ #

    DEFAULT_PROBES: list[str] = [
        "The capital of France is",
        "The largest ocean is",
        "The theory of relativity was developed by",
        "Water is composed of hydrogen and",
        "The human heart pumps",
        "The Eiffel Tower is located in",
        "The square root of 144 is",
        "Shakespeare wrote",
        "The chemical symbol for gold is",
        "Abraham Lincoln was",
        "The speed of light is approximately",
        "Mount Everest is the tallest",
        "The Mona Lisa was painted by",
        "The Earth revolves around",
        "The atomic number of carbon is",
        "The capital of Japan is",
        "DNA is made up of",
        "The Great Wall of China is",
        "The novel Moby Dick was written by",
        "The human brain contains approximately",
        "The Roman Empire fell in",
        "Photosynthesis converts sunlight into",
        "The Amazon rainforest is",
        "The periodic table was created by",
        "The first person to walk on the moon was",
        "Gravity is a force that",
        "The speed of sound is approximately",
        "The Statue of Liberty was a gift from",
        "The Industrial Revolution began in",
        "The boiling point of water is",
        "The Pythagorean theorem states that",
        "The solar system has",
        "The main language spoken in Brazil is",
        "The organ responsible for pumping blood is",
        "The discovery of penicillin is credited to",
        "The currency of the United States is",
        "The tallest building in the world is",
        "The process by which plants make food is called",
        "The smallest unit of matter is",
        "The chemical formula for table salt is",
        "The first president of the United States was",
        "The longest river in the world is",
        "The primary color of leaves is due to",
        "The force that pulls objects toward Earth is",
        "The scientist who developed the theory of evolution was",
        "The largest planet in our solar system is",
        "The closest star to Earth is",
        "The ancient civilization known for pyramids is",
        "The main component of the Sun is",
        "The number of bones in the adult human body is",
    ]

    def dataset_activation(
        self,
        prompts: list[str] | None = None,
        layer: int = 0,
        neuron_index: int = 0,
        top_k: int = 50,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        """Run a set of prompts through the model and record which ones
        most activate the given neuron.

        Uses batched forward passes for efficiency.  Padding tokens are
        excluded from activation measurements.
        """
        if not prompts:
            prompts = list(self.DEFAULT_PROBES)

        results: list[dict[str, Any]] = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]

            encoded = self._model.tokenizer(
                batch, padding=True, return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(self._device)
            attn_mask = encoded["attention_mask"].to(self._device)

            with torch.no_grad():
                _, cache = self._model.run_with_cache(input_ids)

            hook_name = f"blocks.{layer}.mlp.hook_post"
            acts_batch = cache[hook_name]  # (batch, seq, d_mlp)

            for b in range(len(batch)):
                seq_len = int(attn_mask[b].sum().item())
                neuron_acts = acts_batch[b, :seq_len, neuron_index].cpu().numpy()
                max_act = float(neuron_acts.max())
                max_tok = int(neuron_acts.argmax())
                prompt_tokens = self._model.to_str_tokens(batch[b])
                trigger = (
                    prompt_tokens[max_tok] if max_tok < len(prompt_tokens) else "?"
                )

                results.append(
                    {
                        "prompt": batch[b],
                        "max_activation": round(max_act, 6),
                        "trigger_token": trigger,
                        "trigger_token_index": max_tok,
                    }
                )

        results.sort(key=lambda x: -x["max_activation"])
        all_acts = [r["max_activation"] for r in results]

        return {
            "layer": layer,
            "neuron_index": neuron_index,
            "num_prompts": len(results),
            "top_k": top_k,
            "results": results[:top_k],
            "statistics": {
                "mean": round(float(np.mean(all_acts)), 6),
                "std": round(float(np.std(all_acts)), 6),
                "max": round(float(np.max(all_acts)), 6),
            },
        }

    # ------------------------------------------------------------------ #
    #  Prediction Trace — decompose a logit into layer/head contributions
    # ------------------------------------------------------------------ #

    def prediction_trace(
        self, prompt: str, target_token: str | None = None, layer_for_heads: int | None = None
    ) -> dict[str, Any]:
        """Trace which layers' attention and MLP outputs contribute to a
        target token's logit.

        For each layer, projects attn_out and mlp_out through
        ln_final + unembed to measure how much each component's output
        'points toward' the target token.

        If layer_for_heads is given, also decomposes attention heads
        within that layer using W_O projection of hook_z.
        """
        self.run(prompt)

        # Determine target token
        if target_token is not None:
            target_id = self._model.to_single_token(target_token)
            if target_id is None or target_id == -1:
                target_id = self._model.to_single_token(" " + target_token)
            if target_id is None or target_id == -1:
                target_id = self._model.tokenizer.encode(target_token)[0]
        else:
            top_logits = self.last_position_logits
            if top_logits is None:
                return {"error": "No logits available."}
            target_id = int(top_logits.argmax())

        target_token_str = self._model.to_string(int(target_id))

        # Final logit
        resid_final = self.get_cache("blocks.11.hook_resid_post")[-1, :]
        resid_t = torch.from_numpy(resid_final).to(self._device)
        final_logit = float(
            self._model.unembed(self._model.ln_final(resid_t))[int(target_id)]
            .detach()
            .cpu()
            .numpy()
        )

        # Per-layer contribution (logit lens on each component)
        layer_contributions: list[dict[str, Any]] = []
        all_head_contributions: list[dict[str, Any]] = []

        for l in range(self.num_layers):
            attn_out = self.get_cache(f"blocks.{l}.hook_attn_out")[-1, :]
            mlp_out = self.get_cache(f"blocks.{l}.hook_mlp_out")[-1, :]

            attn_t = torch.from_numpy(attn_out).to(self._device)
            mlp_t = torch.from_numpy(mlp_out).to(self._device)

            attn_logit = float(
                self._model.unembed(self._model.ln_final(attn_t))[int(target_id)]
                .detach()
                .cpu()
                .numpy()
            )
            mlp_logit = float(
                self._model.unembed(self._model.ln_final(mlp_t))[int(target_id)]
                .detach()
                .cpu()
                .numpy()
            )

            layer_contributions.append({
                "layer": l,
                "attention_logit": round(attn_logit, 4),
                "mlp_logit": round(mlp_logit, 4),
            })

            # Per-head contributions for the specified layer
            if layer_for_heads is not None and l == layer_for_heads:
                z = self._cache[f"blocks.{l}.attn.hook_z"]  # (1, seq, n_heads, d_head)
                z_last = z[0, -1, :, :]  # (n_heads, d_head)
                W_O = self._model.W_O[l]  # (n_heads, d_head, d_model)
                head_outputs = torch.einsum("hd,hde->he", z_last, W_O)  # (n_heads, d_model)

                for h in range(self.num_heads):
                    h_t = head_outputs[h].unsqueeze(0)  # (1, d_model)
                    h_logit = float(
                        self._model.unembed(self._model.ln_final(h_t))[int(target_id)]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    all_head_contributions.append({
                        "layer": l,
                        "head": h,
                        "logit": round(h_logit, 4),
                    })

                # Sort by abs logit descending
                all_head_contributions.sort(key=lambda x: -abs(x["logit"]))

        # Also compute per-head contributions for any layer that has
        # especially high attention_logit (top 3 by abs value)
        top_attn_layers = sorted(
            layer_contributions, key=lambda x: -abs(x["attention_logit"])
        )[:3]

        # Decompose heads for top attention layers if not already done
        for tal in top_attn_layers:
            l = tal["layer"]
            if layer_for_heads is not None and l == layer_for_heads:
                continue  # already done above
            z = self._cache.get(f"blocks.{l}.attn.hook_z")
            if z is None:
                continue
            z_last = z[0, -1, :, :]  # (n_heads, d_head)
            W_O = self._model.W_O[l]
            head_outputs = torch.einsum("hd,hde->he", z_last, W_O)
            for h in range(self.num_heads):
                h_t = head_outputs[h].unsqueeze(0)
                h_logit = float(
                    self._model.unembed(self._model.ln_final(h_t))[int(target_id)]
                    .detach()
                    .cpu()
                    .numpy()
                )
                all_head_contributions.append({
                    "layer": l,
                    "head": h,
                    "logit": round(h_logit, 4),
                })
        all_head_contributions.sort(key=lambda x: -abs(x["logit"]))

        return {
            "prompt": prompt,
            "target_token": target_token_str,
            "target_id": int(target_id),
            "final_logit": round(final_logit, 4),
            "d_model": self.hidden_dim,
            "layer_contributions": layer_contributions,
            "head_contributions": all_head_contributions[:20] if all_head_contributions else None,
        }

    # ------------------------------------------------------------------ #
    #  Circuit Trace — auto-chain: heads → neurons → residual → logit
    # ------------------------------------------------------------------ #

    def circuit_trace(
        self, prompt: str, target_token: str | None = None
    ) -> dict[str, Any]:
        """Automatically trace the circuit for a target token.

        1. Computes prediction trace (heads + layer contributions)
        2. For the top-K heads, finds important neurons in their layers
        3. Returns the full chain: heads → neurons → residual → final logit
        """
        # Step 1: Full prediction trace (heads for top attention layers)
        pt = self.prediction_trace(prompt, target_token)
        if "error" in pt:
            return pt

        target_tok = pt["target_token"]
        target_id = pt["target_id"]

        # Step 2: For top heads, find important neurons
        head_neurons: list[dict[str, Any]] = []
        head_contribs = pt.get("head_contributions", [])
        seen_layer_neurons: set[tuple[int, int]] = set()

        for hc in (head_contribs or [])[:8]:  # top 8 heads
            l = hc["layer"]
            h = hc["head"]

            # Get attention pattern for this head to find the token it attends to most
            pattern = self.get_cache(f"blocks.{l}.attn.hook_pattern")[0, h, -1, :]  # (seq,)
            most_attended_token = int(np.argmax(pattern))

            # Get top neurons in this layer for the most-attended token
            mlp_acts = self.get_cache(f"blocks.{l}.mlp.hook_post")  # (seq, d_mlp)
            neuron_acts = mlp_acts[most_attended_token, :]  # (d_mlp,)
            top_neuron_indices = np.argsort(neuron_acts)[::-1][:5]

            for ni in top_neuron_indices:
                key = (l, int(ni))
                if key in seen_layer_neurons:
                    continue
                seen_layer_neurons.add(key)
                head_neurons.append({
                    "layer": l,
                    "head": h,
                    "neuron_index": int(ni),
                    "activation": round(float(neuron_acts[int(ni)]), 4),
                    "trigger_token_index": int(most_attended_token),
                    "trigger_token_str": self.str_tokens[int(most_attended_token)] if int(most_attended_token) < len(self.str_tokens) else "?",
                    "head_logit": hc["logit"],
                })

        # Also find top neurons from the target token position directly
        target_token_idx = -1  # last position
        direct_neurons: list[dict[str, Any]] = []
        for l in range(self.num_layers):
            mlp_acts = self.get_cache(f"blocks.{l}.mlp.hook_post")
            neuron_acts = mlp_acts[target_token_idx, :]
            top_n = np.argsort(neuron_acts)[::-1][:3]
            for ni in top_n:
                key = (l, int(ni))
                if key in seen_layer_neurons:
                    continue
                seen_layer_neurons.add(key)
                direct_neurons.append({
                    "layer": l,
                    "neuron_index": int(ni),
                    "activation": round(float(neuron_acts[int(ni)]), 4),
                    "source": "target_position",
                })

        return {
            "prompt": prompt,
            "target_token": target_tok,
            "target_id": target_id,
            "final_logit": pt["final_logit"],
            "layer_contributions": pt["layer_contributions"],
            "head_contributions": (head_contribs or [])[:12],
            "head_neurons": head_neurons[:30],
            "direct_neurons": direct_neurons[:20],
            "circuit_summary": {
                "num_heads_examined": len(head_contribs or []),
                "num_neurons_found": len(head_neurons),
                "top_layers_by_attention": [
                    lc["layer"] for lc in sorted(
                        pt["layer_contributions"], key=lambda x: -abs(x["attention_logit"])
                    )[:5]
                ],
            },
        }

    # ------------------------------------------------------------------ #
    #  Prompt Comparison — run two prompts, compare across all features
    # ------------------------------------------------------------------ #

    def prompt_compare(
        self, prompt_a: str, prompt_b: str, top_k: int = 10
    ) -> dict[str, Any]:
        """Run two prompts and compare neuron activations, attention, residual, logits."""
        self.run(prompt_a)
        tokens_a = list(self.str_tokens)
        logits_a = self.last_position_logits

        def _top_k_preds(logits, k):
            if logits is None:
                return []
            top_idx = np.argsort(logits)[::-1][:k]
            stable = logits - np.max(logits)
            probs = np.exp(stable) / np.sum(np.exp(stable))
            return [{"token_str": self._model.to_string(int(idx)), "token_id": int(idx), "logit": round(float(logits[idx]), 4), "probability": round(float(probs[idx]), 6)} for idx in top_idx]

        preds_a = _top_k_preds(logits_a, top_k)
        top1_a = preds_a[0] if preds_a else None

        neuron_acts_a: list[dict[str, Any]] = []
        for l in range(self.num_layers):
            acts = self.get_cache(f"blocks.{l}.mlp.hook_post")[-1, :]
            for ni in np.argsort(acts)[::-1][:top_k]:
                neuron_acts_a.append({"layer": l, "neuron_index": int(ni), "activation": round(float(acts[int(ni)]), 4)})

        attn_a: list[dict[str, Any]] = []
        for l in range(self.num_layers):
            pat = self.get_cache(f"blocks.{l}.attn.hook_pattern")[0, :, -1, :]
            for h in range(self.num_heads):
                attn_a.append({"layer": l, "head": h, "pattern": pat[h].tolist()})

        def _logit_lens_for_prompt():
            return [{"layer": l, "predictions": _top_k_preds(self._model.unembed(self._model.ln_final(torch.from_numpy(self.get_cache(f"blocks.{l}.hook_resid_post")[-1, :]).to(self._device))).detach().cpu().numpy(), top_k)} for l in range(self.num_layers)]

        lens_a = _logit_lens_for_prompt()

        self.run(prompt_b)
        tokens_b = list(self.str_tokens)
        logits_b = self.last_position_logits
        preds_b = _top_k_preds(logits_b, top_k)
        top1_b = preds_b[0] if preds_b else None

        neuron_acts_b: list[dict[str, Any]] = []
        for l in range(self.num_layers):
            acts = self.get_cache(f"blocks.{l}.mlp.hook_post")[-1, :]
            for ni in np.argsort(acts)[::-1][:top_k]:
                neuron_acts_b.append({"layer": l, "neuron_index": int(ni), "activation": round(float(acts[int(ni)]), 4)})

        attn_b: list[dict[str, Any]] = []
        for l in range(self.num_layers):
            pat = self.get_cache(f"blocks.{l}.attn.hook_pattern")[0, :, -1, :]
            for h in range(self.num_heads):
                attn_b.append({"layer": l, "head": h, "pattern": pat[h].tolist()})

        lens_b = _logit_lens_for_prompt()

        # Top differing neurons
        a_dict = {(n["layer"], n["neuron_index"]): n["activation"] for n in neuron_acts_a}
        b_dict = {(n["layer"], n["neuron_index"]): n["activation"] for n in neuron_acts_b}
        diffs = [{"layer": k[0], "neuron_index": k[1], "activation_a": round(a_dict.get(k, 0.0), 4), "activation_b": round(b_dict.get(k, 0.0), 4), "diff": round(abs(a_dict.get(k, 0.0) - b_dict.get(k, 0.0)), 4)} for k in set(a_dict.keys()) | set(b_dict.keys())]
        diffs.sort(key=lambda x: -x["diff"])

        # Top differing attention heads
        attn_a_d = {(a["layer"], a["head"]): a["pattern"] for a in attn_a}
        attn_b_d = {(a["layer"], a["head"]): a["pattern"] for a in attn_b}
        attn_diffs: list[dict[str, Any]] = []
        for key, pat_a in attn_a_d.items():
            pat_b_val = attn_b_d.get(key)
            if pat_b_val is None:
                continue
            ml = min(len(pat_a), len(pat_b_val))
            attn_diffs.append({"layer": key[0], "head": key[1], "diff": round(sum(abs(pat_a[i] - pat_b_val[i]) for i in range(ml)) / max(ml, 1), 4)})
        attn_diffs.sort(key=lambda x: -x["diff"])

        # Residual norms
        rn_a = [round(float(np.linalg.norm(self.get_cache(f"blocks.{l}.hook_resid_post")[-1, :])), 4) for l in range(self.num_layers)]
        rn_b = [round(float(np.linalg.norm(self.get_cache(f"blocks.{l}.hook_resid_post")[-1, :])), 4) for l in range(self.num_layers)]

        return {
            "prompt_a": prompt_a, "prompt_b": prompt_b,
            "tokens_a": tokens_a, "tokens_b": tokens_b,
            "predictions_a": preds_a, "predictions_b": preds_b,
            "top1_a": top1_a, "top1_b": top1_b,
            "neuron_diffs": diffs[:top_k],
            "attention_diffs": attn_diffs[:top_k],
            "logit_lens_a": lens_a, "logit_lens_b": lens_b,
            "residual_norms_a": rn_a, "residual_norms_b": rn_b,
            "num_layers": self.num_layers, "num_heads": self.num_heads,
        }

    # ------------------------------------------------------------------ #
    #  Experiment Runner — batch prompts, aggregate results, export report
    # ------------------------------------------------------------------ #

    def run_experiment(
        self, prompts: list[str], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run multiple prompts through analyses and aggregate results.

        Collects head importance, neuron importance, category scores,
        and optionally patching matrices across all prompts.
        """
        cfg = config or {}
        run_pm = cfg.get("run_patching_matrix", False)
        top_k = int(cfg.get("top_k", 20))
        target_token = cfg.get("target_token", None)
        if target_token is not None:
            target_token = str(target_token)

        import time

        report: dict[str, Any] = {
            "num_prompts": len(prompts),
            "num_success": 0, "num_failed": 0,
            "duration_seconds": 0.0,
            "config": cfg,
            "errors": [],
        }
        pm_sum: np.ndarray | None = None
        pm_count = 0
        head_logits: dict[tuple[int, int], list[float]] = {}
        neuron_acts: dict[tuple[int, int], list[float]] = {}
        top1_tokens: list[dict[str, Any]] = []

        t0 = time.time()
        for prompt in prompts:
            try:
                self.run(prompt)

                if run_pm:
                    pm = self.compute_patching_matrix(prompt)
                    mat = np.array(pm["matrix"])
                    pm_sum = mat.copy() if pm_sum is None else pm_sum + mat
                    pm_count += 1

                pt = self.prediction_trace(prompt, target_token)
                for hc in (pt.get("head_contributions") or []):
                    head_logits.setdefault((hc["layer"], hc["head"]), []).append(hc["logit"])

                for l in range(self.num_layers):
                    acts = self.get_cache(f"blocks.{l}.mlp.hook_post")[-1, :]
                    for ni in np.argsort(acts)[::-1][:top_k]:
                        neuron_acts.setdefault((l, int(ni)), []).append(float(acts[int(ni)]))

                if self.last_position_logits is not None:
                    tid = int(self.last_position_logits.argmax())
                    top1_tokens.append({"prompt": prompt, "token_str": self._model.to_string(tid), "token_id": tid})

                report["num_success"] += 1
            except Exception as e:
                report["num_failed"] += 1
                report["errors"].append({"prompt": prompt, "error": str(e)})

        report["duration_seconds"] = round(time.time() - t0, 2)

        if pm_sum is not None and pm_count > 0:
            pm_avg = (pm_sum / pm_count).tolist()
            report["patching_matrix"] = {
                "matrix": pm_avg, "layers": len(pm_avg),
                "heads": len(pm_avg[0]) if pm_avg else 0,
                "count": pm_count,
                "vmin": round(float(np.min(pm_sum / pm_count)), 4),
                "vmax": round(float(np.max(pm_sum / pm_count)), 4),
            }

        avg_heads = sorted(
            [{"layer": l, "head": h, "mean_logit": round(sum(v)/len(v), 4), "std_logit": round(float(np.std(v)), 4), "count": len(v)} for (l, h), v in head_logits.items()],
            key=lambda x: -abs(x["mean_logit"])
        )
        report["head_importance"] = avg_heads[:top_k * 3]

        avg_neurons = sorted(
            [{"layer": l, "neuron_index": n, "mean_activation": round(sum(v)/len(v), 4), "std_activation": round(float(np.std(v)), 4), "count": len(v)} for (l, n), v in neuron_acts.items()],
            key=lambda x: -x["mean_activation"]
        )
        report["neuron_importance"] = avg_neurons[:top_k * 3]

        token_counts: dict[str, int] = {}
        for t in top1_tokens:
            token_counts[t["token_str"]] = token_counts.get(t["token_str"], 0) + 1
        report["top1_summary"] = {"total": len(top1_tokens), "distribution": dict(sorted(token_counts.items(), key=lambda x: -x[1])[:top_k])}

        cat_sum: dict[str, float] = {}
        cat_cnt = 0
        for i in range(min(50, len(prompts))):
            try:
                ll = self.logit_lens(prompts[i], 5, target_token)
                for cat, sc in ll.get("category_scores", {}).items():
                    cat_sum[cat] = cat_sum.get(cat, 0.0) + sc
                cat_cnt += 1
            except Exception:
                pass
        if cat_cnt > 0:
            report["category_scores"] = {k: round(v / cat_cnt, 4) for k, v in sorted(cat_sum.items(), key=lambda x: -x[1]) if v / cat_cnt > 0.01}

        return report
