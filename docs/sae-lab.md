# SAE Lab

Sparse-autoencoder support in MECH is real but narrow: it can load a genuine
checkpoint, encode one live hidden state, and train a new top-k dictionary. It
cannot label features, browse examples, or report reconstruction fidelity over
the endpoint. This page states which of those is which, and defines every metric
the way the code computes it.

## 1. Surfaces

| Surface | What it does |
| --- | --- |
| `GET /api/sae/status` | capability probe; **never** returns features |
| `POST /api/sae/inspect` | encode one live hidden state with a real checkpoint |
| `POST /api/sae/train` | start a top-k training job (one at a time) |
| `GET /api/sae/runs/{run_id}` | poll that training job |
| `mech sae list` | known checkpoints from the registry (reference metadata) |
| `mech sae inspect <layer>` | registry facts + configured-checkpoint state |

`/sae/status` reports booleans and provenance only: `torch_available`,
`gpt2_available`, `pipeline_importable`, `checkpoint_configured`,
`checkpoint_source`, `training_running`. An absent capability appears here as
`false`, not as an empty feature list.

## 2. Provenance: where a dictionary came from

A dictionary is only meaningful for **one hook point on one model**. The loader
therefore carries that identity as data and refuses a mismatch.

| Field | Source |
| --- | --- |
| `repo_id`, `version` | the identifier + requested version |
| `checkpoint_sha` | `sha256:` of the checkpoint **file bytes** actually read |
| `weights_sha256` | hash of the encoder weight tensor |
| `d_in`, `d_sae` | encoder weight shape |
| `hook_point`, `layer` | from the checkpoint `cfg` (`hook_name` also accepted) |
| `model_id`, `dataset`, `l1_coefficient`, `revision` | from the checkpoint `cfg` |
| `pre_encoder_center` | `"b_pre"`, `"b_dec"`, or `""` — which vector the input was centred on |

`SAEConfig.applies_to(layer, hook_point)` returns `False` for a dictionary that
does not describe that activation, and an empty `hook_point` means "unknown
provenance — caller decides, and is told so" (it does not claim a match).

### Known checkpoints

`backend/interpretability/sae/registry.py` carries real, public GPT-2-small
dictionaries: `jbloom/GPT2-Small-SAEs`, revision `3a66e186…`, layer `L` at hook
`blocks.L.hook_resid_pre`, `d_in` 768, `d_sae` 24576, trained on
`Skylion007/openwebtext`, `l1_coefficient` 8e-05. Those values were verified by
downloading the checkpoints and reading their `cfg`.

### Sources

| `source` | Behaviour |
| --- | --- |
| `local` | bare filename inside the checkpoint directory (`MECH_SAE_DIR`, default `backend/storage/sae_checkpoints`) |
| `hf` | `layer:<int>` registry lookup, or `repo_id:filename`; fetched with `huggingface_hub` |
| `saelens` | `SAELens.SAE.from_pretrained(...)`; raises `SAELoadError` if `sae_lens` is not installed |
| `anthropic` | **not supported** in this build — raises, reads no weights |

Local identifiers are request-supplied, so they are confined: absolute paths,
`..`, hidden names, separators and UNC shares are `400`; the resolved path must
stay inside the checkpoint directory after symlink resolution; a well-formed name
with no file behind it reports `unavailable` rather than a path error. The
`env:` indirection is honoured **only** from the server environment, never from a
request payload — otherwise a caller could aim the loader at an arbitrary
server-side path. Checkpoints over 512 MiB are refused before deserialisation.

## 3. Metric definitions, as implemented

### Encoding

```
latent  = ReLU((h − b_pre) @ W_enc + b_enc)
features = top-k(latent),  k = min(32, d_sae)
```

- `h` is the live hidden state chosen by dimension: residual stream if
  `d_in == d_model`, else MLP post-activation if `d_in == d_mlp`; anything else
  fails closed with both numbers in the reason.
- Rounding: activations to 6 dp.
- **`pre_encoder_center` decides whether this is a decomposition or noise.** Some
  checkpoints carry a dedicated `b_pre`; `mats_sae_training` / Cunningham format
  has none and centres on `b_dec` (the decoder bias, also the data mean). Only
  the first was read originally, so the second format was encoded uncentred.
  Measured on this host, the layer-8 GPT-2 dictionary reports **51 active
  features at relative L2 0.243 when centred on `b_dec`, and 1558 at 5.56 when it
  is not**. `encoder.bias` is deliberately *not* used as a fallback: in SAELens
  that bias **is** `b_enc`, which is added after the matmul, so subtracting it
  from the input applies it twice with the sign inverted.

### `sparsity`

`sparsity = k / d_sae` — the **reported** fraction of the dictionary returned,
not a measured activation sparsity. With `k = 32` and `d_sae = 24576` it is
`0.0013` for every input. Do not cite it as L0 or as a measured density.

### `n_active_features`

`int((latent > 0).sum())` over the **full** latent, computed in
`reconstruction_error`. This one is measured; it counts non-zero pre-top-k
activations.

### Reconstruction error

```
recon   = latent @ W_dec + b_dec
rel_l2  = ||h − recon|| / ||h||          (None if ||h|| == 0)
```

Returned as `relative_l2_error` with `n_active_features`, `d_sae`, `hook_point`
and `model_id`. It is computed from real tensors or refused — there is no
fallback constant.

### What the endpoint actually reports

`POST /sae/inspect` sets `reconstruction_measured: false` and says so in the
reason: the loader ships encoder weights only for many checkpoints, so encoding
is measured but reconstruction fidelity is not. `reconstruction_error()` is
available in-process when the checkpoint does carry `W_dec` and `b_dec`;
`feature_direction(index)` likewise refuses with "this checkpoint carries no
decoder weights (W_dec)" when it does not.

## 4. Response fields on `/sae/inspect`

`status` / `provenance` / `reason` (or `completed`), `feature_indices`,
`activations`, `sparsity`, `d_sae`, `d_in`, `prompt`, `layer`, `tensor_source`
(e.g. `"residual stream, last token, layer 8"`), `weights_sha256`,
`reconstruction_measured`, `evidence_level` (default `OBSERVATIONAL`),
`field_provenance`. Every field is marked live; a refusal marks every field
`unavailable` and carries no indices.

## 5. Training

`POST /api/sae/train` runs `SAEReproductionPipeline` in a background thread.

| Parameter | Default | Bound |
| --- | --- | --- |
| `n_features` | 64 | clamped to 2–512 |
| `n_tokens` | 1024 | clamped to 64–16384 |
| `steps` | 100 | clamped to 1–2000 |
| `k` | unset (top-k off) | 1–512 when set |
| `seed` | 42 | — |
| `corpus` | pipeline default | refused above 200 000 chars (not truncated) |

Clamped values are what the run record stores, so a request for a billion
features is visibly a 512-feature run. One training slot: a concurrent start
returns `status: busy` rather than a second fit on the shared model. Finished
runs are capped at 20 in memory (`_SAE_MAX_RUNS`), oldest finished first out.

## 6. What is NOT implemented

| Capability | Status |
| --- | --- |
| Feature labels / semantic interpretation | **not implemented.** `SAEInspector.inspect_feature` returns `status: unavailable`, `inspected: false`, empty `connected_neurons` and empty `top_positive_examples`, with the reason "no encoder or decoder weights are loaded, so this feature was not analysed" |
| Feature search over a dictionary | not implemented |
| Activation examples per feature (top-firing prompts) | not implemented |
| Feature–neuron connectivity | not implemented |
| Anthropic-format checkpoints | unsupported (`AnthropicProvider` raises) |
| Reconstruction report over HTTP | not reported (`reconstruction_measured: false`) |
| Multi-layer or whole-model SAE sweeps | not implemented |
| Feature steering from an SAE direction | not implemented as an SAE operation (engine steering exists separately, `gpt2_engine.steer`) |

The inspector's history is the reason for the blunt wording: it previously
returned the same two neuron weights (`0.85`, `0.62`) and the same two example
prompts with activations (`4.2`, `3.8`) **for every feature id**, so `max_act`
and `n_examples` computed from them looked like statistics. It now returns
nothing and says why.

## 7. Caveats for interpretation

1. **A dictionary describes one hook point.** Layer-8 `hook_resid_pre` features
   applied to layer-4 activations index nothing.
2. **Centring is a checkpoint property, not a constant.** Check
   `pre_encoder_center` before comparing two dictionaries' sparsity or L2.
3. **Top-32 is a display cut.** `feature_indices` is the top 32 of a possibly
   much larger active set; `n_active_features` (when reported) is the real count.
4. **Encoder-only checkpoints cannot be checked for fidelity.** An encoding
   without a decoder is unverified, and the response says
   `reconstruction_measured: false` rather than implying otherwise.
5. **`sparsity` is not L0.** See §3.
6. **Training is a small top-k fit, not a research-scale SAE.** Bounds are
   enforced; a 512-feature, 16k-token, 2000-step run is a demonstration, not a
   dictionary you should interpret features from.

## See also

- [supported-models.md](supported-models.md) — which adapters and packages are
  actually present.
- [experiments.md](experiments.md) — provenance and evidence vocabulary.
- [data-and-privacy.md](data-and-privacy.md) — where checkpoints are stored.
- `docs/FEATURE_ATLAS.md`, `docs/INTERPRETABILITY_ALGORITHMS.md` — the wider
  feature-analysis vocabulary.
