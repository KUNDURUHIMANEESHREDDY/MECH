"""Tuned Lens: learned per-layer translators, trained against live weights.

What this is
------------
A plain logit lens unembeds an intermediate residual stream directly. It is known
to be badly miscalibrated at early layers, because the residual stream is not yet
in the space the unembedding expects. A tuned lens learns a small per-layer
affine map that corrects it, trained to minimise the divergence between the
early-layer projection and the model's own final-layer prediction.

What it used to do
------------------
It added a flat `+0.12` to the top-token probability and reported
`affine_translation_applied: True`. That inflated every confidence it produced
while claiming a learned transformation that never ran. An earlier fix made it
declare itself unavailable, which was honest but left the algorithm absent.

Now the translators are trained here, on hidden states from real GPT-2 weights,
and the improvement over the untuned lens is measured and reported.

Scope, stated plainly
---------------------
The default translator is **full affine** -- a per-layer ``d x d`` weight plus a
``d`` bias, ``d*d + d`` parameters (~590k per layer for GPT-2 small), which
matches the published modified architecture. It can rotate between features,
which a per-dimension map cannot.

Two caveats are reported rather than buried. The calibration corpus here is
orders of magnitude smaller than a published tuned lens uses, so the held-out
KL reduction is the only figure that means anything and it will be far below a
paper number. And an L2 penalty toward the identity map is applied, because an
unregularised 590k-parameter map will memorise a small calibration set; the
penalty strength is in the payload as ``identity_decay``.

The cheaper ``diagonal_affine`` family is still fittable (`kind=`), so the
shortfall of the restricted map against full rank is a measurement you can take
rather than an assertion. ``translator_kind`` says which one produced a payload.

Fail-closed
-----------
No translators are bundled, and there is no default weight file. Training must
run, against weights that must load. If either fails the class reports
`provenance: unavailable` and projects nothing. There is no path that returns a
plausible projection without having trained one.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .logit_lens import LogitLens, ProjectionModel

LIVE = "live"
UNAVAILABLE = "unavailable"

#: Translator families this module can fit. ``full_affine`` is the published
#: tuned lens -- a per-layer ``d x d`` weight plus a ``d`` bias, so the map can
#: rotate between features. ``diagonal_affine`` is the cheaper per-dimension
#: scale-and-shift, retained so its shortfall against full rank is measured
#: rather than asserted.
FULL_AFFINE = "full_affine"
DIAGONAL_AFFINE = "diagonal_affine"

#: Fitted by default. The published method is full affine, so the default is no
#: longer the restricted variant.
DEFAULT_TRANSLATOR_KIND = FULL_AFFINE

#: Backwards-compatible alias for callers that imported the old constant.
TRANSLATOR_KIND = DEFAULT_TRANSLATOR_KIND


class TunedLens(ProjectionModel):
    """Logit lens with per-layer translators trained on live GPT-2 weights.

    Two states, and they are not interchangeable:

    * untrained -- `project()` reports the plain logit lens with
      `provenance: unavailable`, because projecting through a lens with no
      trained translators is not a tuned-lens result.
    * trained   -- `train()` has run against loaded weights; `project()`
      applies the learned scale and shift and reports `provenance: live`
      alongside the measured improvement.
    """

    def __init__(self, engine: Optional[Any] = None) -> None:
        self.logit_lens = LogitLens(engine=engine)
        self.engine = self.logit_lens.engine
        # layer -> {"kind": ..., "weight"/"bias" (full affine) or
        #           "scale"/"shift" (diagonal)}
        self.translators: Dict[int, Dict[str, Any]] = {}
        self.training_report: Optional[Dict[str, Any]] = None

    # ── state ────────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return bool(self.translators) and self.training_report is not None

    # ── training ─────────────────────────────────────────────────────────

    def train(
        self,
        calibration_prompts: Sequence[str],
        *,
        steps: int = 200,
        learning_rate: float = 1e-3,
        seed: int = 42,
        holdout: bool = True,
        kind: str = DEFAULT_TRANSLATOR_KIND,
        identity_decay: float = 1e-3,
    ) -> Dict[str, Any]:
        """Fit the per-layer translators against the model's own predictions.

        The objective is the KL divergence between the tuned projection at layer
        L and the model's final-layer distribution, minimised with Adam. That is
        the tuned-lens objective: make the early projection say what the model
        eventually says.

        With `holdout=True` (the default) a fifth of the calibration prompts are
        excluded from fitting and used only to score the result. That
        distinction matters: the in-sample reduction is trivially large and a
        small diagonal map can drive it to near zero by memorising, which would
        make any reported improvement a measurement of the parameter count rather
        than of learning. The held-out figure is the one that means anything.

        Returns a report carrying both, per layer, so the gap between fitting and
        generalising is visible rather than asserted.
        """
        torch = self._torch()
        if torch is None:
            return self._unavailable_report(
                "Training the tuned lens requires torch, which is not "
                "importable. No translators were fitted.")

        if not calibration_prompts:
            return self._unavailable_report(
                "No calibration prompts were supplied, so there is nothing to "
                "fit the translators against. A lens trained on an empty "
                "calibration set is not a trained lens.")

        model, tokenizer = self._model_and_tokenizer()
        if model is None or tokenizer is None:
            return self._unavailable_report(
                "GPT-2 weights are not loaded. The tuned lens is defined as "
                "translators trained against a specific model's own "
                "predictions; without that model there is nothing to train "
                "against and nothing to report.")

        torch.manual_seed(seed)
        fit_prompts, held_prompts = self._split(
            list(calibration_prompts), holdout)

        hidden, final_logits = self._collect_hidden(fit_prompts)
        if hidden is None or not hidden:
            return self._unavailable_report(
                "No hidden states were captured from a forward pass, so no "
                "translators could be fitted.")

        n_layers = len(hidden) - 1
        d_model = hidden[0].shape[-1]
        device = hidden[0].device
        unembed = model.transformer.wte.weight  # tied

        # The translator objective is a matmul against the unembedding, not a
        # model forward pass, so it does not need to share the model's device.
        # When the engine holds GPT-2 on CPU this is the difference between a
        # minutes-long fit and an hour-long one; the collected tensors are
        # already detached, so moving them costs a copy and nothing else.
        compute_device = (torch.device("cuda") if torch.cuda.is_available()
                          else device)
        hidden = [h.to(compute_device) for h in hidden]
        final_logits = final_logits.to(compute_device)
        target_log_probs = torch.log_softmax(final_logits.float(), dim=-1)
        unembed = unembed.to(compute_device)
        device = compute_device
        layers = list(range(1, n_layers + 1))

        if kind not in (FULL_AFFINE, DIAGONAL_AFFINE):
            return self._unavailable_report(
                f"Unknown translator kind {kind!r}; expected {FULL_AFFINE!r} or "
                f"{DIAGONAL_AFFINE!r}.")

        # Both families start at the identity map -- I and 0 for full affine,
        # 1 and 0 for diagonal -- so the untuned and initial-tuned lenses are
        # the same function and the reported reduction is measured from a
        # neutral starting point rather than from a lucky initialisation.
        if kind == FULL_AFFINE:
            weight = {L: torch.eye(d_model, device=device, requires_grad=True)
                      for L in layers}
        else:
            weight = {L: torch.ones(d_model, device=device, requires_grad=True)
                      for L in layers}
        bias = {L: torch.zeros(d_model, device=device, requires_grad=True)
                for L in layers}
        params = list(weight.values()) + list(bias.values())
        optimiser = torch.optim.Adam(params, lr=learning_rate)

        def translate(L, h):
            if kind == FULL_AFFINE:
                return h @ weight[L].T + bias[L]
            return h * weight[L] + bias[L]

        def identity_penalty():
            """L2 distance from the identity map.

            A full-rank translator is ``d*d + d`` parameters per layer -- about
            590k for GPT-2 small. On a calibration set this small it can drive
            the in-sample KL toward zero by memorising, so the penalty keeps it
            near the identity unless the data supports moving away. It is
            reported, so the figure is not read as an unregularised fit.
            """
            if not identity_decay:
                return None
            total = None
            for L in layers:
                anchor = (torch.eye(d_model, device=device)
                          if kind == FULL_AFFINE else 1.0)
                pen = ((weight[L] - anchor) ** 2).sum()
                total = pen if total is None else total + pen
            return identity_decay * total

        # Measured with no translator applied, i.e. the untuned lens's KL.
        with torch.no_grad():
            before = {L: float(self._mean_kl(torch, hidden[L], target_log_probs,
                                             unembed).item())
                      for L in layers}

        for _ in range(int(steps)):
            optimiser.zero_grad()
            total = None
            for L in layers:
                kl = self._mean_kl(torch, hidden[L], target_log_probs, unembed,
                                   translated=translate(L, hidden[L].float()))
                total = kl if total is None else total + kl
            pen = identity_penalty()
            if pen is not None:
                total = total + pen
            total.backward()
            optimiser.step()

        after = {L: float(self._mean_kl(
                     torch, hidden[L], target_log_probs, unembed,
                     translated=translate(L, hidden[L].float())).item())
                 for L in layers}

        heldout = None
        if held_prompts:
            heldout = self._holdout_report(
                torch, held_prompts, unembed,
                lambda L, h: translate(L, h.float()), layers)

        self.translators = {
            L: ({"kind": kind,
                 "weight": weight[L].detach().clone(),
                 "bias": bias[L].detach().clone()}
                if kind == FULL_AFFINE else
                {"kind": kind,
                 "scale": weight[L].detach().clone(),
                 "shift": bias[L].detach().clone()})
            for L in layers
        }

        per_layer = []
        for L in layers:
            b, a = before[L], after[L]
            entry = {
                "layer": L,
                "kl_untuned": round(b, 6),
                "kl_tuned": round(a, 6),
                "kl_reduction": round(b - a, 6),
                # Relative reduction is reported alongside the absolute one
                # because an early layer's absolute KL is large and a large
                # absolute drop there is not the same as a large *fraction*
                # recovered.
                "kl_reduction_pct": (round(100.0 * (b - a) / b, 2)
                                     if b > 0 else None),
                "improved": a < b,
            }
            if heldout is not None:
                h = heldout["per_layer"][L]
                entry.update({
                    "holdout_kl_untuned": h["kl_untuned"],
                    "holdout_kl_tuned": h["kl_tuned"],
                    "holdout_kl_reduction": h["kl_reduction"],
                    "holdout_kl_reduction_pct": h["kl_reduction_pct"],
                    "holdout_improved": h["improved"],
                })
            per_layer.append(entry)

        usable = [p for p in per_layer if p["kl_tuned"] is not None]
        improved = [p for p in usable if p["improved"]]
        heldout_improved = [p for p in usable
                            if p.get("holdout_improved") is True]

        n_params_per_layer = (d_model * d_model + d_model
                              if kind == FULL_AFFINE else 2 * d_model)
        n_params = n_params_per_layer * len(layers)
        report = {
            "status": "completed",
            "provenance": LIVE,
            "translator_kind": kind,
            "parameters_per_layer": n_params_per_layer,
            "identity_decay": float(identity_decay),
            "model_id": getattr(self.engine, "MODEL_ID", None),
            "n_calibration_prompts": len(calibration_prompts),
            "n_fit_prompts": len(fit_prompts),
            "n_holdout_prompts": len(held_prompts),
            "holdout_prompts": held_prompts,
            "n_tokens_used": int(hidden[0].shape[0]),
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "layers_trained": sorted(layers),
            "parameters_fitted": n_params,
            "per_layer": per_layer,
            "layers_improved": len(improved),
            "layers_total": len(usable),
            "layers_holdout_improved": len(heldout_improved) if heldout else None,
            # Stated rather than implied: a translator that did not help is a
            # fitted-but-useless parameter set, and reporting it as trained
            # without this number would overstate what was achieved.
            "all_layers_improved": len(improved) == len(usable) and bool(usable),
            # The headline number is the held-out one. In-sample improvement on
            # a calibration set this small is close to unfalsifiable.
            "headline_metric": "holdout_kl_reduction_pct" if heldout
                               else "kl_reduction_pct",
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                f"Fitted {n_params} {kind} parameters "
                f"({n_params_per_layer} per layer) over {len(usable)} layers on "
                f"{len(fit_prompts)} calibration prompts "
                f"({int(hidden[0].shape[0])} tokens)"
                + (f", scored on {len(held_prompts)} held-out prompts"
                   if heldout else ", with no held-out split")
                + f". {len(improved)} of {len(usable)} layers improved in "
                  f"sample"
                + (f"; {len(heldout_improved)} of {len(usable)} improved "
                   f"out of sample." if heldout else ".")
                + (f" Full-affine translators match the published modified "
                   f"architecture, but the calibration set is orders of "
                   f"magnitude smaller than a published tuned lens uses, and "
                   f"the identity penalty ({identity_decay}) is this module's "
                   f"addition, so these figures are not a reproduction of the "
                   f"paper's numbers."
                   if kind == FULL_AFFINE else
                   f" Published-restricted: this is the {DIAGONAL_AFFINE} "
                   f"variant, not the full affine translator, so these figures "
                   f"must not be quoted as 'the tuned lens'.")
            ),
        }
        self.training_report = report
        return report

    @staticmethod
    def _split(prompts: List[str], holdout: bool):
        """Deterministic fit/holdout split.

        Every fifth prompt is held out. Deterministic rather than shuffled so a
        repeated run scores the same prompts, and so the split cannot silently
        change between the report and a reproduction of it.
        """
        if not holdout or len(prompts) < 2:
            return prompts, []
        fit = [p for i, p in enumerate(prompts) if i % 5 != 0]
        held = [p for i, p in enumerate(prompts) if i % 5 == 0]
        # Keep at least one prompt to fit on.
        if not fit:
            return prompts, []
        return fit, held

    def _holdout_report(self, torch, held_prompts, unembed, translate_fn, layers):
        """Score the fitted translators on prompts they never saw."""
        hidden, final_logits = self._collect_hidden(held_prompts)
        if hidden is None or not hidden:
            return None
        # Score on the same device the translators were fitted on.
        target_device = getattr(unembed, "device", None)
        if target_device is not None:
            hidden = [h.to(target_device) for h in hidden]
            final_logits = final_logits.to(target_device)
        target = torch.log_softmax(final_logits.float(), dim=-1)

        per_layer = {}
        for L in layers:
            with torch.no_grad():
                b = float(self._mean_kl(torch, hidden[L], target, unembed).item())
                a = float(self._mean_kl(
                    torch, hidden[L], target, unembed,
                    translated=translate_fn(L, hidden[L])).item())
            per_layer[L] = {
                "kl_untuned": round(b, 6),
                "kl_tuned": round(a, 6),
                "kl_reduction": round(b - a, 6),
                "kl_reduction_pct": (round(100.0 * (b - a) / b, 2)
                                     if b > 0 else None),
                "improved": a < b,
            }
        return {"per_layer": per_layer, "n_tokens": int(hidden[0].shape[0])}

    # ── projection ───────────────────────────────────────────────────────

    def project(self, prompt: str, layer: int, top_k: int = 5) -> Dict[str, Any]:
        """Project through the trained translator, then unembed.

        This used to delegate straight to `self.logit_lens.project` and return
        that payload with `affine_translation_applied: True` written into it.
        The flag described a transformation that never happened: the numbers
        were the plain logit lens, byte-identical to an untrained lens, while
        the payload also claimed `tuned_lens_available`, `trained` and
        `layer_trained`. A researcher comparing a tuned lens against a raw one
        would have measured no difference and been told the tuning was applied.

        The translator is now applied to the cached residual stream before the
        unembedding, which is the tuned-lens definition. When the hidden state is
        unavailable -- weights unloaded, no cached prompt, an untrained layer --
        that is reported rather than substituted with the untransformed value,
        because a fallback here would reinstate exactly the defect.
        """
        base = self.logit_lens.project(prompt, layer, top_k=top_k)

        if not self.is_trained:
            base.update({
                "method": "TunedLens",
                "affine_translation_applied": False,
                "tuned_lens_available": False,
                "trained": False,
                "translator_kind": TRANSLATOR_KIND,
                "provenance": UNAVAILABLE,
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": (
                    "No translators have been trained. The projection below is "
                    "the plain logit lens and no translation was applied. A "
                    "tuned lens without trained translators is a logit lens "
                    "with a misleading name."
                ),
            })
            return base

        trained = self.training_report or {}
        layer_trained = layer in self.translators

        translated = self._project_translated(prompt, layer, top_k)
        if translated is None:
            # Report honestly instead of returning the untransformed
            # projection under a flag saying it was transformed.
            base.update({
                "method": "TunedLens",
                "affine_translation_applied": False,
                "tuned_lens_available": True,
                "trained": True,
                "layer_trained": layer_trained,
                "translator_kind": trained.get("translator_kind",
                                                TRANSLATOR_KIND),
                "provenance": UNAVAILABLE,
                "validation_eligible": False,
                "publication_eligible": False,
                "top_token": None,
                "top_probability": None,
                "top_k_tokens": None,
                "reason": (
                    "The residual stream for this prompt and layer was not "
                    "available, so the trained translator could not be applied. "
                    "The projection reported here is therefore the plain logit "
                    "lens and no translation was applied. Nothing was measured "
                    "with the tuned translator."
                ),
            })
            return base

        base.update(translated)
        base.update({
            "method": "TunedLens",
            "affine_translation_applied": True,
            "tuned_lens_available": True,
            "trained": True,
            "layer_trained": layer_trained,
            "translator_kind": trained.get("translator_kind", TRANSLATOR_KIND),
            "provenance": LIVE,
            "training": trained,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "The layer's trained diagonal-affine translator was applied to "
                "the residual stream before the unembedding. Note this is one "
                "prompt's projection; it is not by itself evidence of a "
                "mechanism."
            ),
        })
        return base

    def _project_translated(self, prompt: str, layer: int,
                            top_k: int) -> Optional[Dict[str, Any]]:
        """Apply the translator and re-project, or ``None`` if not possible.

        Reads the residual stream the engine already cached for `prompt`, so
        this costs no extra forward pass. The final layer's cached entry is
        post-`ln_f` (see `docs/notes/tl_block11_anomaly.md`), so it is used as-is
        while earlier layers are normalised first -- the same convention
        `gpt2_engine.logit_lens` follows.
        """
        torch = self._torch()
        if torch is None:
            return None
        if layer not in self.translators:
            return None

        engine = self.engine
        snapshot = getattr(engine, "_snapshot", None)
        if snapshot is None:
            return None
        state = snapshot()
        if state.get("prompt") != prompt:
            # The cache describes a different prompt; reading it would report
            # another prompt's activations under this one.
            return None

        hidden = (state.get("hidden") or [])
        n_layers = int(getattr(engine, "_n_layers", lambda: 0)())
        index = layer + 1
        if index >= len(hidden):
            return None

        try:
            model = engine._model
            row = torch.tensor(hidden[index][-1], dtype=torch.float32)
            entry = self.translators[layer]
            if entry.get("kind") == FULL_AFFINE:
                w = entry["weight"].float().to(row.device)
                b = entry["bias"].float().to(row.device)
                translated = row @ w.T + b
            else:
                s = entry["scale"].float().to(row.device)
                sh = entry["shift"].float().to(row.device)
                translated = row * s + sh

            # Earlier entries are pre-ln_f; the last is already normalised.
            if layer != n_layers - 1:
                translated = model.transformer.ln_f(translated)

            logits = translated @ model.transformer.wte.weight.T.float()
            probs = torch.softmax(logits, dim=-1)
            k = max(1, min(int(top_k), probs.numel()))
            top = torch.topk(probs, k)
            tokens = [
                {
                    "token": engine._decode(int(i)),
                    "prob": round(float(probs[i]), 6),
                }
                for i in top.indices
            ]
        except Exception:
            # Any failure means no tuned projection; the caller reports that
            # rather than falling back to an untransformed number.
            return None

        return {
            "top_token": tokens[0]["token"] if tokens else None,
            "top_probability": tokens[0]["prob"] if tokens else None,
            "top_k_tokens": tokens,
        }

    def apply_translator(self, hidden: Any, layer: int) -> Any:
        """Apply a layer's learned affine map to a hidden-state tensor.

        Separate from `project()` so the arithmetic is testable without a
        forward pass, and so it is explicit that `project()` does *not* apply
        it -- the engine's cached projection is returned as-is.
        """
        if layer not in self.translators:
            raise KeyError(
                f"no trained translator for layer {layer}; trained layers are "
                f"{sorted(self.translators)}")
        entry = self.translators[layer]
        if entry.get("kind") == FULL_AFFINE:
            return (hidden @ entry["weight"].to(hidden.device).T
                    + entry["bias"].to(hidden.device))
        return hidden * entry["scale"] + entry["shift"]

    # ── internals ────────────────────────────────────────────────────────

    def _torch(self):
        try:
            import torch  # noqa: PLC0415
        except Exception:
            return None
        return torch

    def _model_and_tokenizer(self):
        loader = getattr(self.engine, "_ensure_loaded", None)
        if loader is not None:
            error = loader()
            if error:
                return None, None
        return (getattr(self.engine, "_model", None),
                getattr(self.engine, "_tokenizer", None))

    def _collect_hidden(self, prompts: Sequence[str]):
        """Hidden states and final logits for the calibration set."""
        torch = self._torch()
        model, tokenizer = self._model_and_tokenizer()
        if torch is None or model is None or tokenizer is None:
            return None, None

        device = next(model.parameters()).device
        collected: Optional[List[Any]] = None
        finals: List[Any] = []

        for text in prompts:
            encoded = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**encoded, output_hidden_states=True)
            states = out.hidden_states  # tuple(n_layer+1) of (1, T, d)
            if collected is None:
                # Flatten batch and time so every token contributes.
                collected = [s[0].detach() for s in states]
            else:
                for i, s in enumerate(states):
                    collected[i] = torch.cat([collected[i], s[0].detach()], dim=0)
            finals.append(out.logits[0].detach())

        if collected is None or not finals:
            return None, None
        return collected, torch.cat(finals, dim=0)

    @staticmethod
    def _mean_kl(torch, hidden, target_log_probs, unembed, *,
                 translated=None):
        """Mean KL(target || projection) over tokens, in nats.

        Returns a **tensor**, not a float. This runs inside the training loop,
        and converting to a Python scalar there severs the autograd graph --
        `total.backward()` then fails on a float. An earlier version returned
        `float(...)`, so `train()` raised AttributeError on the first step and
        the lens could never actually be trained.

        Convert with `.item()` at the reporting boundary, not here.
        """
        # `translated` is the already-translated hidden state. Taking it as an
        # argument keeps _mean_kl free of any knowledge of which translator
        # family produced it, so both share one KL implementation.
        h = (translated if translated is not None else hidden).float()
        logits = torch.nn.functional.layer_norm(
            h, (h.shape[-1],)) @ unembed.T.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        # target_log_probs may cover more tokens than h if shapes differ; take
        # the leading slice so the two always describe the same tokens.
        n = min(log_probs.shape[0], target_log_probs.shape[0])
        if n == 0:
            return torch.tensor(float("nan"))
        target = target_log_probs[:n]
        kl = (target.exp() * (target - log_probs[:n])).sum(dim=-1)
        return kl[:n].mean()

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> Dict[str, Any]:
        """Persist the fitted translators and their report to ``path``.

        This is the "pre-train, then reuse" seam: training is the expensive
        part, and a lens that has to be refitted for every process is not a
        vendored artifact. The file is written with ``torch.save`` and loaded
        with ``weights_only=True``, so it holds tensors and primitives only.
        """
        if not self.translators or self.training_report is None:
            return {"status": "unavailable", "provenance": UNAVAILABLE,
                    "reason": "No trained translators to save."}
        torch = self._torch()
        if torch is None:
            return {"status": "unavailable", "provenance": UNAVAILABLE,
                    "reason": "torch is not importable, so nothing can be saved."}
        import hashlib
        import os

        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "translator_kind": self.training_report.get("translator_kind"),
            "report": self.training_report,
            "translators": {int(L): dict(v) for L, v in self.translators.items()},
        }
        torch.save(payload, path)
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        return {"status": "saved", "provenance": LIVE, "path": path,
                "sha256": f"sha256:{digest}",
                "size_bytes": os.path.getsize(path),
                "translator_kind": payload["translator_kind"]}

    def load(self, path: str) -> Dict[str, Any]:
        """Load translators written by :meth:`save`, or report why not.

        Refuses an unknown translator kind and a missing file rather than
        returning an empty lens that would silently project untransformed.
        """
        import os

        torch = self._torch()
        if torch is None:
            return {"status": "unavailable", "provenance": UNAVAILABLE,
                    "reason": "torch is not importable, so nothing can be loaded."}
        if not os.path.isfile(path):
            return {"status": "unavailable", "provenance": UNAVAILABLE,
                    "reason": f"No translator file at {path!r}."}
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            return {"status": "unavailable", "provenance": UNAVAILABLE,
                    "reason": (f"{path!r} could not be loaded with "
                               f"weights_only=True and was refused rather than "
                               f"unpickled: {type(exc).__name__}: {exc}")}
        if not isinstance(payload, dict) or "translators" not in payload:
            return {"status": "unavailable", "provenance": UNAVAILABLE,
                    "reason": f"{path!r} is not a tuned-lens checkpoint."}
        kind = payload.get("translator_kind")
        if kind not in (FULL_AFFINE, DIAGONAL_AFFINE):
            return {"status": "unavailable", "provenance": UNAVAILABLE,
                    "reason": f"Unknown translator kind {kind!r} in {path!r}."}
        self.translators = {int(L): dict(v)
                            for L, v in payload["translators"].items()}
        self.training_report = payload.get("report")
        return {"status": "loaded", "provenance": LIVE, "path": path,
                "translator_kind": kind,
                "layers": sorted(self.translators)}

    def _unavailable_report(self, reason: str) -> Dict[str, Any]:
        self.translators = {}
        self.training_report = {
            "status": "unavailable",
            "provenance": UNAVAILABLE,
            "translator_kind": TRANSLATOR_KIND,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": reason,
        }
        return self.training_report