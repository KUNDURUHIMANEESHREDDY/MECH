# Logit lens

What MECH computes when a surface says "logit lens", exactly, so a number can be
checked rather than believed. The implementation is
`backend/services/gpt2_engine.py::logit_lens` (single layer) and
`logit_lens_all` (every layer, one cached forward). `mech run logit-lens` and
`POST /api/gpt2/logit_lens_all` both end up there.

## 1. The projection, step by step

For layer index `L` on prompt `p`, over the cached forward pass for `p`:

1. **Take the residual stream at the last token position, output of block `L`.**
   `h = cache["hidden"][L + 1][-1]`, cast to `float32`. Indexing `L + 1` is
   deliberate: slot `0` is the token embedding and slot `k` is the output of
   block `k - 1`, so `L + 1` is *after* block `L`.
2. **Apply the final LayerNorm exactly once.**
   - If `L < n_layers - 1`: `normed = ln_f(h)`.
   - If `L == n_layers - 1`: `normed = h` unchanged.
   The second branch exists because HuggingFace's `output_hidden_states` puts the
   **already normalised** final residual in that last slot, in place of block
   `n-1`'s raw output. Applying `ln_f` again applies the learned scale and bias
   twice. That defect was real and measured: with the double normalisation the
   argmax still matched the model's, but `p(" the")` read 0.038 against the
   model's 0.085, and a token the model puts at 0.004 was reported at 0.019 and
   promoted into the top five. The token-identity test passed while the
   probabilities were wrong.
3. **Project through the tied unembedding.** `logits = normed @ W_wte^T`. The
   unembedding is GPT-2's token embedding matrix, tied, with no extra linear
   bias.
4. **Softmax over the whole vocabulary.** `probs = softmax(logits)` — 50257
   entries for GPT-2, not just the survivors.
5. **Take the top-k** (default 5 per layer, 3 per layer for the all-layers
   endpoint) and round probabilities to 6 dp.

There is no trained affine translator in this path. MECH calls it a raw logit
lens and means it.

### What counts as "layer"

`layer = L` means **after transformer block `L`**, i.e. the hidden state that
block `L+1` would read. `layer = 0` is after the first block, not after the
embedding. `logit_lens_all` therefore returns `n_layers` entries
(`L = 0 … n_layers-1`); the residual *stream* has one more element than that,
because of the `ln_f` slot described above.

## 2. Response shape

`logit_lens(layer, prompt, top_k=5)` returns:

| Field | Meaning |
| --- | --- |
| `status` | `ok`, or a refusal with the reason |
| `method` | `"LogitLens"` |
| `layer`, `prompt` | echo of the request |
| `top_token` | highest-probability token |
| `top_k_tokens` | `[{token, prob}, …]`, `prob` rounded to 6 dp |
| `probability_is_truncated` | `true` — always |
| `probabilities_reported` | `k` (usually 5) |
| `probabilities_total` | vocabulary size (50257 for GPT-2) |
| `probabilities_reported_mass` | sum of the returned probabilities |

`logit_lens_all(prompt, top_k=3)` returns
`{status, method, prompt, layers: [{layer, top_token, top_k_tokens}, …]}`.

## 3. Normalization and truncation conventions

- **LayerNorm is the model's own `ln_f`**, with its learned weight and bias. No
  z-scoring, no per-layer rescaling, no temperature.
- **Probabilities are a softmax over the full vocabulary**, so they are
  comparable across layers and to the model's own output distribution.
- **The returned list is a truncation of a distribution, not a distribution.**
  Nothing is renormalised over the top-k. `probabilities_reported_mass` states
  the mass that was returned — measured example: the final layer of
  "The capital of France is" returns five of 50257 entries summing to 0.243318.
  Summing `top_k_tokens` and reporting the result as the model's confidence is
  the failure mode this field exists to prevent.
- **No renormalisation on masked/forced tokens.** If you want a conditional
  distribution, compute it from the logits yourself; this endpoint does not
  offer one.

## 4. Caveats — read before interpreting

1. **It is a projection, not a computation the model performed.** A logit lens
   reads an intermediate residual stream through the *final* unembedding. The
   model never computes that distribution at layer `L`; it is a linear probe of
   a representation that the rest of the network would still transform. Early
   layers are therefore usually near-arbitrary (frequent-token priors), and a
   plausible token at layer 3 is not evidence that the model "decided" it there.
2. **Observational evidence, at most.** Nothing here intervenes. Per
   `backend/core/provenance.py`, this is `OBSERVATIONAL`; causal claims require
   patching/ablation results (`INTERVENTIONAL` or stronger), not a lens.
3. **The tuned lens is a different object.** `algorithms/tuned_lens.py` applies
   per-layer affine translators before unembedding. With no trained translators
   loaded it refuses: it returns the plain logit lens **labelled as such**, with
   `affine_translation_applied: false`, `tuned_lens_available: false`,
   `provenance: unavailable` and a reason. Do not compare a "tuned lens" number
   without checking `affine_translation_applied`.
4. **Architecture-specific normalisation.** The `ln_f`-in-the-last-slot rule is
   HuggingFace GPT-2 behaviour. Architectures without `ln_f` do not follow it
   (`gpt2_engine.architecture()` states this in `residual_convention`). The
   engine only loads GPT-2 today.
5. **Layer semantics differ across modules.** `ResidualInspector.get_residual_norms`
   walks `range(num_layers)` and returns `n_layers` entries with **no** final
   normalised slot — it is a different series, not a shorter one of the same.
   `LogitInspector` softmaxes logits it receives from a repository
   (`backend/interpretability/logit_inspector.py:88-100`); it does not apply
   `ln_f`, because its input is already projected. Same word, different stage.
6. **One token position.** The projection is the last token's residual, the
   standard next-token framing. It says nothing about earlier positions.
7. **`top_k` is clamped** to `[1, vocab_size]`; it cannot be used to request the
   full distribution through this path.
8. **A refusal is a refusal.** If the model is unavailable, `status` is
   `unavailable`/`error` with the reason, and no token list is attached. There is
   no "representative" fallback distribution.

## 5. Reproducing a number

```bash
mech run logit-lens --prompt "The capital of France is" --layer 11
```

```bash
curl -s -X POST http://127.0.0.1:8000/api/gpt2/logit_lens_all \
  -H "Authorization: Bearer $MECH_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is"}'
```

Check `method`, `probabilities_reported_mass` and `probability_is_truncated`
before quoting a probability; check `status` before quoting anything.

## See also

- [supported-models.md](supported-models.md) — the model and its residual-stream
  convention.
- [experiments.md](experiments.md) — evidence levels and provenance fields.
- `docs/INTERPRETABILITY_ALGORITHMS.md`, `docs/CIRCUIT_DISCOVERY.md` — where
  causal methods sit relative to this observational one.
