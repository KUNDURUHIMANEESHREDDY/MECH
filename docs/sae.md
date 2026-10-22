# SAE Lab

The SAE Lab encodes **one** live GPT-2 hidden state with **one** real SAE
checkpoint and shows the feature indices and activations that the
checkpoint's own encoder produced.

Everything on screen comes from an API response. When no checkpoint can be
reached the view says so and draws no features.

---

## Pointing the platform at a checkpoint

There are two ways, and they differ in what the `identifier` means.

### 1. Server environment (the deployment's default)

```bash
MECH_SAE_SOURCE=local
MECH_SAE_CHECKPOINT=C:\Users\himan\Downloads\app test\MECH\backend\storage\sae_checkpoints\sae_d768_n50_k12_s42_400st_f43a8d1d.pt
```

* `MECH_SAE_SOURCE` must be a source the loader accepts: `local`, `hf`,
  `saelens` or `anthropic` (`backend/interpretability/sae/loader.py`). For a
  file on this machine it is `local`.
* `MECH_SAE_CHECKPOINT` accepts either form:
  * a **bare file name** is resolved inside the checkpoint directory, the same
    place a request-supplied identifier is resolved, so
    `MECH_SAE_CHECKPOINT=sae_d768_n50_k12_s42_400st_f43a8d1d.pt` names the file
    in `MECH_SAE_DIR`;
  * an **absolute path** is honoured as given, for a checkpoint that lives
    outside the directory.

  Before this, a bare name was handed to the loader as a path and resolved
  against the server's working directory, so the variable name did not mean
  what it said.
* `MECH_SAE_DIR` moves the checkpoint directory. Default:
  `backend/storage/sae_checkpoints/`.
* Leave `MECH_API_TOKEN` unset for a local run: the API token is then read
  from `backend/storage/.mech_api_token`.

With the environment set, `GET /api/sae/status` reports
`checkpoint_configured: true` and `checkpoint_source: "local"`, and
`POST /api/sae/inspect` encodes with that checkpoint.

### 2. Per request (explicit source + identifier)

```json
POST /api/sae/inspect
{"prompt": "The capital of France is", "layer": 8,
 "source": "local", "identifier": "sae_d768_n50_k12_s42_400st_f43a8d1d.pt"}
```

* `identifier` must be a **bare file name inside the checkpoint directory**.
  Absolute paths, `..`, path separators, hidden names and the `env:`
  indirection are refused with HTTP 400 — on purpose: a request may not aim
  the loader at an arbitrary server path.
* `source` may be left out; the backend then resolves the identifier
  locally.
* The SAE Lab sends exactly this: choosing a file in the checkpoint select
  sends `source: "local"` plus that identifier.

### Training new checkpoints

`POST /api/sae/train` writes into the same directory with the same naming
(`sae_d768_n{features}_k{top_k}_s{seed}_{steps}st_{hash}.pt`), so a trained
dictionary shows up in the same select.

---

## Which checkpoints exist locally

`backend/storage/sae_checkpoints/` holds **118** real trained checkpoints
on this machine:

| file group | count | what it is |
| --- | --- | --- |
| `sae_d768_n50_k12_s42_400st_f43a8d1d.pt` | 1 | 50 features, top-k 12, 400 steps, seed 42 |
| `sae_d768_n64_k16_s42_100st_8688ade7.pt` | 1 | 64 features, top-k 16, 100 steps, seed 42 |
| `sae_d768_n8_k2_s42_2st_*.pt` | 116 | 8 features, top-k 2, 2 steps — short test runs |

Each file holds real trained tensors: `W_enc (768, n)`, `b_enc (n,)`,
`b_pre (768,)`, `W_dec (n, 768)`, `b_dec (768,)` plus the run's
hyper-parameters. `backend/interpretability/sae/local_checkpoints.py` lists
what is on disk (name, size, mtime; `.pt` only, newest first, at most 50)
and is what the status route's `available_checkpoints` and `checkpoint_dir`
come from. A missing directory is an empty list, never a placeholder entry.

---

## A real measurement (this machine)

Prompt `"The capital of France is"`, layer 8, checkpoint
`sae_d768_n50_k12_s42_400st_f43a8d1d.pt`:

* encoded vector: the residual stream after block 8, last token
  (`d_in = 768`, `d_sae = 50`)
* weights digest: `sha256:079468735db6bea1f50e249fac6a051c627135c26f5011774b273f89ea1c2fdd`
* feature indices: `46, 6, 3, 8, 38, 18, 16, 41, 43, 1, 35, 9, 31, 11, 20,
  40, 14, 39, 4, 12, 36, 0, 45, 32, 10, 15, 34, 27, 44, 5, 49, 25`
* activations: `14.680641, 12.702766, 6.246507, 5.888462, 5.52628,
  4.963767, 4.779645, 4.352975, 4.136837, 3.887381, 3.83858, 3.603892,
  3.339100, 2.830403, 2.784046, 2.412809, 2.326519, 2.196602, 1.984691,
  1.425005, 1.364386, 1.250852, 0.907753, 0.825550, 0.558083, 0.490343,
  0.488293, 0.300031, 0.231244, 0.165826, 0.059030, 0.000000`

The same numbers come out of the request payload path and of the
environment path, and they match an in-process run of the same loader.

The reconstruction number quoted below is *not* part of that response. It
is measured from the checkpoint's own decoder, in process, with the loader
the route uses:

```python
from backend.interpretability.sae.loader import SAELoader
from backend.services import gpt2_engine
import torch

detail = gpt2_engine.layer_activations(8, "The capital of France is")
state = torch.tensor(detail["resid_post"][-1], dtype=torch.float32)
sae = SAELoader().load_sae("local", "backend/storage/sae_checkpoints/"
                                   "sae_d768_n50_k12_s42_400st_f43a8d1d.pt", "latest")
encoded = sae.activate(state)
print(encoded["feature_indices"])
print(sae.reconstruction_error(state, encoded["feature_indices"]))
# -> relative_l2_error 0.969552, n_active_features 31
```

---

## What the view does when no checkpoint is available

* Step 1 shows `checkpoint_configured: false` and states that no server
  checkpoint is configured.
* The checkpoint facts show **not returned by this backend** for any field
  the backend does not report. A field that was not returned is unknown —
  it is never drawn as "none exist" and never as "features available".
* The checkpoint select is disabled with the reason shown next to it. A
  checkpoint can still be named by hand in the identifier field.
* Clicking **Inspect features** returns the backend's honest `unavailable`
  answer with its reason; the view shows that answer and renders **no**
  features. An empty feature list is never drawn as "no feature fires" —
  that is a measurement the backend did not make.

---

## What this feature does NOT claim

* **No feature names.** A feature index is a row of the dictionary, not a
  concept. Nothing here says what a feature "means".
* **No monosemanticity score, no interpretability score, no causal claim.**
  No number here says how interpretable, important or causal a feature is.
* **No reconstruction fidelity from the API.** `POST /api/sae/inspect`
  reports `reconstruction_measured: false` and returns no fidelity number.
  The local checkpoints *do* carry a decoder (`W_dec`, `b_dec`), and
  `SAE.reconstruction_error` measures one from them — for the example
  above, relative L2 **0.969552** over 31 active features — but the inspect
  route does not call it, so the API does not report it and the view shows
  `NOT AVAILABLE`. Turning that on is a dispatcher change, not a view change.
* **Not a corpus statistic.** The activations are the checkpoint's ReLU
  latents for one hidden state (the last token of one prompt at one layer),
  in raw activation units. They are not firing rates, not frequencies, and
  not comparable across prompts by construction.
* **The activation may not be the one the dictionary was trained on.** The
  local checkpoints were trained on GPT-2 small's block-0 MLP contribution
  added to the residual stream, while the inspect route encodes the residual
  stream at the requested layer (block 8 in the example above). These
  checkpoints record no hook point, so the loader cannot refuse the
  mismatch and the high reconstruction error above is the symptom, not a
  measurement of dictionary quality.
* **Not attested, not validation-eligible.** The response carries
  `evidence_level: OBSERVATIONAL` and `attested: false`; the checkpoint's
  own fit metrics are not re-verified here.
* **No comparison and no ranking.** No feature is scored against another,
  and no list is ordered by importance.

---

## Verifying it

```bash
cd frontend
node tests/e2e/live-sae-check.mjs
```

The harness reads the API token from `backend/storage/.mech_api_token`,
drives the SAE Lab, and compares the features on screen with the payload
`POST /api/sae/inspect` returned — every rendered index must be one the API
returned, every rendered activation must be the value the API returned, and
no feature may appear that the API did not return. Against a backend with
no checkpoint it asserts the honest `NOT AVAILABLE` state and zero rendered
features.
