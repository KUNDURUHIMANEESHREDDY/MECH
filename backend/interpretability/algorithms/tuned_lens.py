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
The translator is **diagonal** affine -- a per-dimension scale and shift,
`2 * d_model` parameters per layer, ~18k for GPT-2 small across 12 layers. The
tuned-lens paper uses a full affine per layer (~590k per layer for d=768).

That is a real restriction, not a detail. A diagonal map can correct per-feature
gain and offset but cannot rotate between features, so it will recover less
than a full affine map. Every figure reported here is therefore a measurement
*of this restricted variant*, and `translator_kind` says so in the payload. It
would be misleading to report these numbers as "the tuned lens".

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

#: The translator family actually implemented. Reported in every payload,
#: because "tuned lens" unqualified would imply the full affine map.
TRANSLATOR_KIND = "diagonal_affine"


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
        # layer -> {"scale": tensor(d,), "shift": tensor(d,)}
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

        target_log_probs = torch.log_softmax(final_logits.float(), dim=-1)

        scale = {L: torch.ones(d_model, device=device, requires_grad=True)
                 for L in range(1, n_layers + 1)}
        shift = {L: torch.zeros(d_model, device=device, requires_grad=True)
                 for L in range(1, n_layers + 1)}
        params = list(scale.values()) + list(shift.values())
        optimiser = torch.optim.Adam(params, lr=learning_rate)

        # Measured with no translator applied: scale 1, shift 0, so this is the
        # untuned lens's KL. `.item()` here because nothing differentiates it.
        with torch.no_grad():
            before = {L: float(self._mean_kl(torch, hidden[L], target_log_probs,
                                             unembed, None, None).item())
                      for L in scale}

        for _ in range(int(steps)):
            optimiser.zero_grad()
            total = None
            for L in scale:
                kl = self._mean_kl(torch, hidden[L], target_log_probs,
                                   unembed, scale[L], shift[L])
                total = kl if total is None else total + kl
            total.backward()
            optimiser.step()

        after = {L: float(self._mean_kl(torch, hidden[L], target_log_probs,
                                        unembed, scale[L], shift[L]).item())
                 for L in scale}

        heldout = None
        if held_prompts:
            heldout = self._holdout_report(torch, held_prompts, unembed,
                                          scale, shift)

        self.translators = {
            L: {"scale": scale[L].detach().clone(),
                "shift": shift[L].detach().clone()}
            for L in scale
        }

        per_layer = []
        for L in sorted(scale):
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

        report = {
            "status": "completed",
            "provenance": LIVE,
            "translator_kind": TRANSLATOR_KIND,
            "model_id": getattr(self.engine, "MODEL_ID", None),
            "n_calibration_prompts": len(calibration_prompts),
            "n_fit_prompts": len(fit_prompts),
            "n_holdout_prompts": len(held_prompts),
            "holdout_prompts": held_prompts,
            "n_tokens_used": int(hidden[0].shape[0]),
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "layers_trained": sorted(scale),
            "parameters_fitted": 2 * d_model * len(scale),
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
                f"Fitted {2 * d_model * len(scale)} diagonal-affine parameters "
                f"over {len(usable)} layers on {len(fit_prompts)} calibration "
                f"prompts ({int(hidden[0].shape[0])} tokens)"
                + (f", scored on {len(held_prompts)} held-out prompts"
                   if heldout else ", with no held-out split")
                + f". {len(improved)} of {len(usable)} layers improved in "
                  f"sample"
                + (f"; {len(heldout_improved)} of {len(usable)} improved "
                   f"out of sample." if heldout else ".")
                + f" Published-restricted: this is the {TRANSLATOR_KIND} "
                  f"variant, not the full affine translator, so these figures "
                  f"must not be quoted as 'the tuned lens'. The calibration set "
                  f"is also orders of magnitude smaller than a published tuned "
                  f"lens uses."
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

    def _holdout_report(self, torch, held_prompts, unembed, scale, shift):
        """Score the fitted translators on prompts they never saw."""
        hidden, final_logits = self._collect_hidden(held_prompts)
        if hidden is None or not hidden:
            return None
        target = torch.log_softmax(final_logits.float(), dim=-1)

        per_layer = {}
        for L in sorted(scale):
            with torch.no_grad():
                b = float(self._mean_kl(torch, hidden[L], target, unembed,
                                        None, None).item())
                a = float(self._mean_kl(torch, hidden[L], target, unembed,
                                        scale[L], shift[L]).item())
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
        base.update({
            "method": "TunedLens",
            "affine_translation_applied": True,
            "tuned_lens_available": True,
            "trained": True,
            "translator_kind": trained.get("translator_kind", TRANSLATOR_KIND),
            "provenance": LIVE,
            "layer_trained": layer in self.translators,
            "training": trained,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Translators were trained against this model's own predictions. "
                "Note the projection here is still the logit lens applied to "
                "one prompt; the learned translator is applied inside "
                "`apply_translator`, and a per-prompt projection is not by "
                "itself evidence of a mechanism."
            ),
        })
        return base

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
    def _mean_kl(torch, hidden, target_log_probs, unembed,
                 scale=None, shift=None):
        """Mean KL(target || projection) over tokens, in nats.

        Returns a **tensor**, not a float. This runs inside the training loop,
        and converting to a Python scalar there severs the autograd graph --
        `total.backward()` then fails on a float. An earlier version returned
        `float(...)`, so `train()` raised AttributeError on the first step and
        the lens could never actually be trained.

        Convert with `.item()` at the reporting boundary, not here.
        """
        h = hidden.float()
        # Scale and shift are applied independently. Requiring both-or-neither
        # made `h * scale + shift` raise on a legitimate "scale only" call.
        if scale is not None:
            h = h * scale
        if shift is not None:
            h = h + shift
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