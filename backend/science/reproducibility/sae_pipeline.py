"""SAE Reproduction Pipeline.

Trains a sparse autoencoder on GPT-2's MLP post-activation residuals and reports
reconstruction error, L0 sparsity and per-feature max-activation attributions,
all measured.

What this used to be
--------------------
A seeded RNG. The feature bank was drawn from it::

    act_freq    = round(rng.betavariate(0.5, 5.0), 4)
    top_tokens  = rng.sample(token_pool, min(n_top, len(token_pool)))
    mono        = round(rng.betavariate(3.0, 1.5), 4)
    is_absorbed = rng.random() < 0.12

reported under field names that read as findings. `top_activating_tokens` was a
random sample from a 19-word list, `monosemanticity_score` was a betavariate
draw, and `feature_absorption_rate` was a coin flip at p=0.12.

Three things made it worse than the other simulated pipelines:

* **It ignored `mock_mode`.** A caller with real GPT-2 weights loaded still got
  simulated features. The constructor flag controlled nothing.
* **`reconstruction_mse` was not a reconstruction.** It was
  `0.03 + (1 - mean_monosemanticity) * 0.04` -- a function of the same
  simulated scores. No encoding or decoding ran.
* **The manifest claimed a corpus that was never read.**
  `dataset_name="OpenWebText Sample"`, so every report asserted a provenance this
  platform had no access to.

The corpus
----------
There is no natural-language corpus bundled with this repository and no dataset
cache on the machine, so the default corpus is **this repository's own
documentation text** -- `README.md` and the module docstrings under `backend/`.

That is a real corpus and it is pinned by SHA-256, but it is *technical
documentation*, not natural prose. A sparse autoencoder trained on it learns
features for technical vocabulary: "provenance", "seed", "measured". Every
attribution below should be read that way. This is not a claim about what GPT-2
represents in general, and `dataset_name` says so.

Pass `corpus=` to train on something else. Nothing is invented when the corpus is
absent -- the manifest records the files actually read, with their digests, and
`corpus_is_documentation_text` is `True` so a consumer can refuse to treat the
result as a monosemanticity finding.

The autoencoder
---------------
A **top-k** SAE: the encoder's `k` largest pre-activations are kept and the rest
thresholded to zero, so sparsity is structural rather than penalised. `k` is a
parameter, not a target -- L0 is whatever the fit produced, measured as the mean
number of active features, not set to the `k` that was requested.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..models.adapter_base import LiveUnavailable

LIVE = "live"
UNAVAILABLE = "unavailable"


def _default_corpus(root: Path, max_chars: int = 200_000) -> Tuple[str, Dict[str, Any]]:
    """This repository's documentation text, with digests.

    Returns the text and a manifest describing exactly what was read. The
    manifest is the point: a corpus nobody can name is how "OpenWebText Sample"
    ended up in a dataset field for a corpus that was never opened.
    """
    candidates = [root / "README.md"]
    backend = root / "backend"
    if backend.is_dir():
        for path in sorted(backend.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            candidates.append(path)

    chunks: List[str] = []
    files: List[Dict[str, Any]] = []
    total = 0
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        files.append({"path": path.relative_to(root).as_posix(),
                      "sha256": digest, "chars": len(text)})
        chunks.append(text)
        total += len(text)
        if total >= max_chars:
            break

    corpus = "\n\n".join(chunks)[:max_chars]
    manifest = {
        "dataset_name": (
            "MECH repository documentation text (README.md and backend module "
            "docstrings). NOT a natural-language corpus."
        ),
        "corpus_is_documentation_text": True,
        "corpus_licence": (
            "Same licence as this repository. It is project documentation, "
            "which is why it was available offline."
        ),
        "corpus_chars": len(corpus),
        "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "corpus_files": files,
        "corpus_file_count": len(files),
        "limitation": (
            "Technical documentation, not natural prose. Features and "
            "attributions learned here describe technical vocabulary and must "
            "not be presented as findings about what GPT-2 represents in "
            "general."
        ),
    }
    return corpus, manifest


def _sae_class():
    """Build the autoencoder class.

    A factory rather than a module-level `torch.nn.Module` subclass, so importing
    this module does not import torch -- the pipeline raises `LiveUnavailable`
    when torch is missing, which it cannot do if the import already failed.
    """
    import torch

    class _TopKSAE(torch.nn.Module):
        """Top-k sparse autoencoder. Decoder rows are kept at unit norm.

        `n_features` is the dictionary width; `k` is how many entries survive the
        top-k threshold. They are different quantities, and conflating them is
        how L0 ends up equal to a requested constant.

        Deliberately small: the point is a measured reconstruction, not a
        competitive sparse-autoencoder result.
        """

        def __init__(self, d_model: int, n_features: int, k: int) -> None:
            super().__init__()
            self.d_model = d_model
            self.n_features = n_features
            self.k = max(1, min(int(k), n_features))
            self.W_enc = torch.nn.Parameter(
                torch.randn(d_model, n_features) * 0.02)
            self.b_enc = torch.nn.Parameter(torch.zeros(n_features))
            self.W_dec = torch.nn.Parameter(
                torch.randn(n_features, d_model) * 0.02)
            self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
            with torch.no_grad():
                self.unit_normalise_decoders()

        def unit_normalise_decoders(self) -> None:
            norms = self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            with torch.no_grad():
                self.W_dec.data /= norms

        def encode(self, x):
            pre = (x - self.b_dec) @ self.W_enc + self.b_enc
            values, indices = torch.topk(pre, self.k, dim=-1)
            # Everything outside the top k is thresholded to exactly zero. That
            # is what makes the representation sparse: the zeros are structural,
            # not a small penalty term.
            sparse = torch.zeros_like(pre).scatter_(-1, indices, values)
            return torch.relu(sparse), pre

        def decode(self, sparse):
            return sparse @ self.W_dec + self.b_dec

        def forward(self, x):
            sparse, pre = self.encode(x)
            return self.decode(sparse), sparse, pre

        def l0(self, sparse):
            """Mean number of active features per token, measured."""
            return float((sparse != 0).sum(dim=-1).float().mean())

        def dead_fraction(self, sparse):
            """Fraction of dictionary entries never active on this data."""
            active = (sparse != 0).any(dim=0).float()
            return float((active == 0).float().mean())

    return _TopKSAE


class SAEReproductionPipeline:
    """Trains a top-k SAE on live GPT-2 MLP activations."""

    PAPER_ID = "sparse_autoencoders"

    def __init__(self, mock_mode: bool = False, root: Optional[Path] = None,
                 engine: Optional[Any] = None) -> None:
        self.mock_mode = mock_mode
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        self.engine = engine

    # ── public API ───────────────────────────────────────────────────────

    def run(self, n_features: int = 50, seed: int = 42, *,
            corpus: Optional[str] = None, n_tokens: int = 4096,
            k: Optional[int] = None, steps: int = 400,
            learning_rate: float = 3e-4, top_tokens: int = 8,
            checkpoint_dir: Optional[Any] = None) -> Dict[str, Any]:
        """Train and evaluate a top-k SAE. Raises rather than simulating."""
        torch = self._torch()
        if torch is None:
            raise LiveUnavailable(
                "Training a sparse autoencoder requires torch, which is not "
                "importable. No autoencoder was fitted and no reconstruction "
                "error is reported.")

        model, tokenizer = self._model_and_tokenizer()
        if model is None or tokenizer is None:
            raise LiveUnavailable(
                "SAEReproductionPipeline needs loaded GPT-2 weights. It trains "
                "an autoencoder on this model's own MLP activations, so without "
                "the model there is nothing to train on. No feature bank is "
                "returned.")

        if corpus is None:
            corpus, corpus_manifest = _default_corpus(self.root)
        else:
            corpus_manifest = {
                "dataset_name": "caller-supplied corpus",
                "corpus_is_documentation_text": False,
                "corpus_chars": len(corpus),
                "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
                "limitation": (
                    "Provenance of a caller-supplied corpus is the caller's "
                    "claim; this platform did not fetch or verify it."
                ),
            }

        if not corpus.strip():
            raise LiveUnavailable(
                "The corpus is empty, so no activations could be collected. No "
                "autoencoder was fitted.")

        # Determinism is load-bearing: same seed + params + corpus must pin
        # the artifact, or "seed" is decoration. Two sources break it:
        # (1) the global RNG (reset by manual_seed), and (2) multithreaded
        # CPU reductions, whose summation order varies run to run. The fit
        # therefore runs single-threaded, restoring the previous setting
        # afterwards so interactive use keeps its threads.
        torch.manual_seed(seed)
        acts = self._collect_activations(torch, model, tokenizer, corpus, n_tokens)
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            if acts is None or acts.shape[0] == 0:
                raise LiveUnavailable(
                    "No MLP activations were captured from a forward pass, so no "
                    "autoencoder could be fitted.")

            # Hold out tokens so the reported reconstruction is not the training fit.
            split = max(1, int(0.8 * acts.shape[0]))
            fit_x, held_x = acts[:split], acts[split:]
            if held_x.shape[0] == 0:
                held_x = fit_x

            d_model = acts.shape[1]
            n_features = max(2, min(int(n_features), 4 * d_model))
            top_k = int(k) if k else max(2, min(16, n_features // 4))

            sae = _sae_class()(d_model, n_features, top_k).to(acts.device)
            optimiser = torch.optim.Adam(sae.parameters(), lr=learning_rate)

            # Per-element variance, and the summed squared deviation it corresponds
            # to. The squared error below is summed over the feature dimension while
            # `var()` is per element, so normalising one against the other divides by
            # d_model too few -- it reported NMSE 303 for a fit whose true NMSE was
            # 0.395, and I nearly wrote off a working reconstruction as a failure.
            variance = float(fit_x.var())
            variance_sum = variance * d_model
            history: List[Dict[str, float]] = []
            for step in range(int(steps)):
                optimiser.zero_grad()
                recon, sparse, _ = sae(fit_x)
                loss = ((recon - fit_x) ** 2).sum(dim=-1).mean()
                loss.backward()
                optimiser.step()
                sae.unit_normalise_decoders()
                if step % max(1, int(steps) // 8) == 0 or step == int(steps) - 1:
                    history.append({"step": step,
                                    "train_mse": round(float(loss.detach()), 8)})

            with torch.no_grad():
                fit_recon, fit_sparse, _ = sae(fit_x)
                held_recon, held_sparse, _ = sae(held_x)

                fit_mse = float(((fit_recon - fit_x) ** 2).sum(-1).mean())
                held_mse = float(((held_recon - held_x) ** 2).sum(-1).mean())
                held_l0 = sae.l0(held_sparse)
                dead = sae.dead_fraction(held_sparse)

            # Fraction of activation variance left unexplained. Both terms are
            # summed over the feature dimension, so the ratio is dimensionless and
            # comparable across models and layers.
            nmse = held_mse / variance_sum if variance_sum > 0 else None

            # An NMSE at or above 1 means the reconstruction is worse than predicting
            # the mean, so it is not a reconstruction at all. Reported as a failure
            # rather than left for a reader to notice.
            reconstruction_useful = bool(nmse is not None and nmse < 1.0)

            attributions = self._attribute(torch, sae, held_x, held_sparse,
                                           tokenizer, top_tokens)

            # Persist the fitted weights so train -> inspect is a real chain, not
            # two endpoints that merely share a name. The file carries the encoder
            # the loader reads (W_enc, b_enc, b_pre) plus the decoder and the fit
            # metrics the checkpoint was saved with. A save failure is recorded,
            # not fatal: the metrics above were still measured.
            checkpoint_record = self._save_checkpoint(
                torch, sae, d_model, n_features, top_k, int(seed), int(steps),
                float(learning_rate), checkpoint_dir)

            metrics = {
                # The measured reconstruction error, from a real encode/decode.
                "reconstruction_mse": round(held_mse, 8),
                "train_reconstruction_mse": round(fit_mse, 8),
                "held_out_fraction": round(held_x.shape[0] / acts.shape[0], 4),
                # Measured, not the requested k.
                "l0_mean_active_features": round(held_l0, 4),
                "requested_k": top_k,
                "dictionary_size": n_features,
                "activation_variance": round(variance, 8),
                "activation_variance_summed": round(variance_sum, 6),
                "mse_units": "sum of squared error over the feature dimension",
                "reconstruction_useful": reconstruction_useful,
                "normalized_mse": (round(nmse, 6) if nmse is not None else None),
                "dead_feature_fraction": round(dead, 4),
                "n_activations": int(held_x.shape[0]),
                "n_features": n_features,
                "topk_k": top_k,
                "top_activating_tokens": attributions,
                "training_history": history,
                # Stated because it bounds what the attributions mean.
                "monosemanticity_scored": False,
                "monosemanticity_reason": (
                    "Not scored. A monosemanticity score needs a stated "
                    "interpretability criterion evaluated against an independent "
                    "judge; max-activation token attribution is not that, and "
                    "inventing a scalar from it would repeat the defect this "
                    "pipeline previously had. The per-feature tokens below are "
                    "measurements of which tokens maximise each feature, and "
                    "nothing more."
                ),
            }

            # The status reflects whether the fit produced a usable reconstruction.
            # Training without converging is not "completed": reporting a status that
            # reads as success alongside an NMSE above 1 would be the same defect as
            # the one this pipeline was rewritten to remove.
            if not reconstruction_useful:
                return {
                    "pipeline": "SAEReproductionPipeline",
                    "status": UNAVAILABLE,
                    "provenance": UNAVAILABLE,
                    "field_provenance": {name: UNAVAILABLE for name in metrics},
                    "mock_mode": False,
                    "validation_eligible": False,
                    "publication_eligible": False,
                    "observed_metrics": metrics,
                    "checkpoint": checkpoint_record,
                    "corpus": corpus_manifest,
                    "seed": int(seed),
                    "steps": int(steps),
                    "reason": (
                        f"An autoencoder was fitted but its reconstruction is not "
                        f"usable: normalised MSE {nmse:.4g} is at or above 1, so it "
                        f"explains less of the held-out activation variance than "
                        f"predicting the mean would. Reported rather than presented "
                        f"as a reconstruction. The training history is in "
                        f"`observed_metrics.training_history`; more steps or a higher "
                        f"learning rate may fix it, and a corpus with narrower "
                        f"activation outliers may help more."
                    ),
                }

            return {
                "pipeline": "SAEReproductionPipeline",
                "status": "completed",
                "provenance": LIVE,
                "field_provenance": {
                    name: LIVE for name in metrics
                },
                "mock_mode": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "model_id": getattr(self.engine, "MODEL_ID", None),
                "autoencoder": "top_k_sae",
                "observed_metrics": metrics,
                "checkpoint": checkpoint_record,
                "corpus": corpus_manifest,
                "seed": int(seed),
                "steps": int(steps),
                "learning_rate": float(learning_rate),
                "reason": (
                    f"Top-k SAE with {n_features} dictionary elements and k={top_k} "
                    f"trained for {int(steps)} steps on {int(acts.shape[0])} real "
                    f"GPT-2 MLP post-activation vectors "
                    f"({int(held_x.shape[0])} held out). Reconstruction MSE "
                    f"{held_mse:.6g} on held-out activations, normalised "
                    f"{nmse:.4g} of activation variance, mean L0 {held_l0:.1f}."
                    + (" Corpus is this repository's documentation text, so these "
                       "are features of technical vocabulary, not findings about "
                       "GPT-2 in general." if corpus_manifest.get(
                           "corpus_is_documentation_text") else "")
                ),
            }

        finally:
            torch.set_num_threads(prev_threads)

    # ── internals ────────────────────────────────────────────────────────

    @staticmethod
    def _default_checkpoint_dir(root: Path) -> Path:
        return Path(root) / "backend" / "storage" / "sae_checkpoints"

    def _save_checkpoint(self, torch, sae, d_model: int, n_features: int,
                         top_k: int, seed: int, steps: int,
                         learning_rate: float,
                         checkpoint_dir: Optional[Any]) -> Dict[str, Any]:
        """Persist fitted weights beside the metrics that describe them.

        Keys match what the checkpoint loader reads (`W_enc`, `b_enc`,
        `b_pre`; the loader centres with `b_pre`, which is this SAE's
        `b_dec`). A unique filename per run: overwriting a previous
        checkpoint with a new fit would silently re-point every inspection
        at different weights.
        """
        import uuid

        target = (Path(checkpoint_dir) if checkpoint_dir is not None
                  else self._default_checkpoint_dir(self.root))
        try:
            target.mkdir(parents=True, exist_ok=True)
            name = (f"sae_d{d_model}_n{n_features}_k{top_k}_s{seed}_"
                    f"{steps}st_{uuid.uuid4().hex[:8]}.pt")
            path = target / name
            with torch.no_grad():
                payload = {
                    "W_enc": sae.W_enc.detach().cpu().clone(),
                    "b_enc": sae.b_enc.detach().cpu().clone(),
                    "b_pre": sae.b_dec.detach().cpu().clone(),
                    "W_dec": sae.W_dec.detach().cpu().clone(),
                    "b_dec": sae.b_dec.detach().cpu().clone(),
                    "d_model": int(d_model),
                    "n_features": int(n_features),
                    "top_k": int(top_k),
                    "seed": int(seed),
                    "steps": int(steps),
                    "learning_rate": float(learning_rate),
                    "autoencoder": "top_k_sae",
                }
                torch.save(payload, str(path))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            # File bytes identify the artifact instance, but `torch.save`
            # output is not byte-stable across calls (zip container
            # metadata), so it cannot pin reproducibility. The content
            # digest covers dtype + shape + raw bytes of every weight tensor
            # in sorted key order: same seed must pin the learned weights,
            # whatever the container does.
            content = hashlib.sha256()
            for key in sorted(payload):
                value = payload[key]
                if torch.is_tensor(value):
                    content.update(str(value.dtype).encode("utf-8"))
                    content.update(str(tuple(value.shape)).encode("utf-8"))
                    content.update(value.detach().to("cpu").contiguous()
                                   .numpy().tobytes())
                else:
                    content.update(repr(value).encode("utf-8"))
            return {
                "checkpoint_saved": True,
                "checkpoint_path": str(path),
                "checkpoint_sha256": f"sha256:{digest}",
                "checkpoint_content_sha256": f"sha256:{content.hexdigest()}",
                "checkpoint_d_in": int(d_model),
                "checkpoint_d_sae": int(n_features),
            }
        except Exception as exc:
            return {
                "checkpoint_saved": False,
                "checkpoint_path": None,
                "checkpoint_sha256": None,
                "reason": (f"Checkpoint could not be written: "
                           f"{type(exc).__name__}: {exc}"),
            }

    def _attribute(self, torch, sae, acts, sparse, tokenizer,
                   top_tokens: int) -> List[Dict[str, Any]]:
        """Which tokens maximise each feature, measured on held-out activations.

        This replaces the old `rng.sample(token_pool, ...)`. The previous
        implementation drew tokens at random and reported them as "top
        activating"; these are ranked by each feature's actual activation over
        real token positions.

        The arg-max token is reported rather than excluded -- it *is* the
        definition of a feature's top activating token. What is deliberately not
        done is turning this into a scalar: see `monosemanticity_reason`.
        """
        with torch.no_grad():
            pre = (acts - sae.b_dec) @ sae.W_enc + sae.b_enc
            # Which token position maximises each feature.
            best_pos = pre.argmax(dim=0)
            best_val = pre.max(dim=0).values

        out: List[Dict[str, Any]] = []
        order = torch.argsort(best_val, descending=True).tolist()
        for feature in order[:max(1, top_tokens * 2)]:
            token_id = int(best_pos[feature])
            try:
                token = tokenizer.decode([token_id])
            except Exception:
                token = str(token_id)
            out.append({
                "feature": int(feature),
                "top_tokens": [{"token": token,
                                "activation": round(float(best_val[feature]), 4)}],
                # What this feature's decoder direction reconstructs is a real
                # measurement of the feature, not a property of the corpus.
                "is_dead_on_holdout": bool(
                    float(sparse[:, feature].abs().sum()) == 0.0),
            })
        return out

    def _collect_activations(self, torch, model, tokenizer, corpus: str,
                             n_tokens: int):
        """Real MLP post-activation vectors from a forward pass.

        GPT-2's activation function is `gelu_new`, so `mlp.act(...)` is the
        post-activation residual. Pre-activation would be a different quantity
        and the reconstruction numbers would not be comparable.

        The MLP's contribution to the residual stream is obtained by calling the
        block's own `c_proj`, rather than by hand-multiplying with
        `c_proj.weight`. HF stores `Conv1D.weight` as `(in, out)`, so for
        `c_proj` that is `(3072, 768)` and an explicit `.T` produces a
        `(768, 3072)` operand that cannot multiply a `(*, 3072)` activation --
        `RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x3072 and
        768x3072)`. Letting the module do its own matmul removes the transpose
        question entirely.
        """
        device = next(model.parameters()).device
        collected = []
        total = 0
        budget = max(64, int(n_tokens))

        # Tokenize in slices so no single call exceeds the tokenizer's window
        # and warns about indexing errors that never happen.
        step_chars = 2000
        block = 128
        for offset in range(0, len(corpus), step_chars):
            slice_text = corpus[offset:offset + step_chars]
            if not slice_text.strip():
                continue
            ids = tokenizer(slice_text, return_tensors="pt").input_ids
            for start in range(0, ids.shape[1] - 1, block):
                chunk = ids[:, start:start + block].to(device)
                if chunk.shape[1] < 2:
                    continue
                with torch.no_grad():
                    out = model(chunk, output_hidden_states=True)
                layer = 0
                hidden = out.hidden_states[layer + 1]
                residual = out.hidden_states[layer]
                block_obj = model.transformer.h[layer]
                normed = block_obj.ln_2(hidden)
                acts = block_obj.mlp.act(block_obj.mlp.c_fc(normed))
                contribution = block_obj.mlp.c_proj(acts)
                rows = (residual + contribution)[0]
                collected.append(rows.detach().float())
                total += rows.shape[0]
            if total >= budget:
                break

        if not collected:
            return None
        return torch.cat(collected, dim=0)[:budget]

    def _model_and_tokenizer(self):
        from backend.services import gpt2_engine

        self.engine = self.engine or gpt2_engine
        loader = getattr(self.engine, "_ensure_loaded", None)
        if loader is not None:
            error = loader()
            if error:
                return None, None
        return (getattr(self.engine, "_model", None),
                getattr(self.engine, "_tokenizer", None))

    @staticmethod
    def _torch():
        try:
            import torch  # noqa: PLC0415
        except Exception:
            return None
        return torch