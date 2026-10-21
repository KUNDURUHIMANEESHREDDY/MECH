# Supported models

GPT-2 Small is the only model MECH actually loads and runs today. Everything
else in the catalogue is either a spec-table entry with a class behind it and no
verified run, or a registered gap that reports `NOT_EXECUTED`. This page states
which is which, because the distinction is the whole point of `AGENTS.md`'s
no-fabricated-results rule.

## 1. The live model

| Fact | Value | Source |
| --- | --- | --- |
| Model id | `gpt2` (GPT-2 Small, 124M) | `backend/services/gpt2_engine.py:36` `SUPPORTED_MODEL` |
| Loading | `AutoTokenizer` / `AutoModelForCausalLM.from_pretrained("gpt2")` | same file, `load()` |
| Cache behaviour | tries `local_files_only=True` first, then downloads | `gpt2_engine.py:117-127` |
| Attention impl | `attn_implementation="eager"` — attention weights must be materialised | `gpt2_engine.py:120` |
| Config forced | `output_attentions = True`, `output_hidden_states = True` | `gpt2_engine.py:128-129` |
| Architecture | 12 layers, 12 heads, `d_model` 768, `d_mlp` 3072, vocab 50257 | read from the loaded config at runtime |

No second model can be loaded by the engine: `load()` has the repo id hardcoded,
and `infer(prompt, model_name)` documents that it echoes only the one model it
loaded rather than accepting a different name (`gpt2_engine.py:1545-1550`).

### Operations the engine really performs

Every one of these runs a live forward pass over real weights and returns
`measured: true` with provenance via the `@_measured` decorator
(`gpt2_engine.py:292`):

`architecture`, `run_prompt`, `generate_text`, `layer_detail`, `layer_activations`,
`list_neurons`, `neuron_detail`, `head_detail`, `attention_head`, `activations`,
`patch_head`, `patch_neuron`, `ioi`, `logit_lens`, `logit_lens_all`,
`ablate_layer`, `steer`.

If torch/transformers are missing, or weights cannot be loaded, each returns an
`unavailable`/`error` envelope with the reason — never a placeholder.
`is_available()` reports whether the ML stack imports; it does not mean weights
are resident.

## 2. Adapter matrix — real availability

"Adapter" here means the code path that would talk to a given library. Presence
of the module and presence of the package are different facts, so both columns
are given. The package column was probed on this checkout with
`importlib.util.find_spec`; it is a property of the environment, not of the
repository, and `mech doctor` re-probes it live.

| Adapter | Module | Package importable here | What it can do today |
| --- | --- | --- | --- |
| Hugging Face (`transformers`) | `backend/science/models/hf_adapter.py` | **yes** (`transformers`, `torch`) | loads a real model, forward passes, activation capture; used by the GPT-2 engine |
| TransformerLens | `backend/science/models/transformer_lens_adapter.py`, `backend/interpretability/tl_compat.py` | **yes** (`transformer_lens`) | `HookedTransformer` resolution with version/env diagnosis; hook-based capture and patching |
| SAELens | `backend/interpretability/sae/loader.py::SAELensProvider` | **no** (`sae_lens`) | if installed: `SAE.from_pretrained`, encoder-side encoding; without it, raises `SAELoadError("sae_lens is not installed: …")` |
| NNsight | `backend/interpretability/nnsight_adapter.py` | **no** (`nnsight`) | request construction works; execution reports `NOT_EXECUTED` with the reason |
| Anthropic SAE format | `loader.py::AnthropicProvider` | n/a | **not supported** — raises `SAELoadError` and reads no weights |

### TransformerLens caveat

TransformerLens 4.0 removed `HookedTransformer`, which is the class this adapter
needs. `tl_compat.resolve_hooked_transformer()` therefore distinguishes three
outcomes rather than one: available, `TransformerLensVersionError` (installed but
too new), and `TransformerLensEnvironmentError` (import failed for another
reason). `check_compatibility()` returns the report plus a `reason` of `version`
or `environment`. The adapter does not silently fall back to a raw HF model — a
caller that asked for hooks gets an error, not different numbers.

### NNsight: implemented module, absent capability

`nnsight_adapter.py` is complete as a module and honest about what is missing:

| Surface | Behaviour with `nnsight` absent |
| --- | --- |
| `describe()` / `capabilities()` | `available: False`, `version: None`, `status: NOT_EXECUTED`, reason naming the package and `pip install nnsight` |
| `operations` | every entry `False`; `supported_operations` empty |
| `build_trace_request(...)` | **works** — data construction only, so a caller can inspect the request a run would send |
| `run_trace(...)` | `NOT_EXECUTED`, no data fields |

The execution path (`model.trace(...)`, `proxy.save()`, in-place proxy
assignment) follows NNsight's documented API but has **not** been exercised
against an installed `nnsight` in this environment, and the module says so in its
own docstring. Do not read "NNsight adapter present" as "NNsight runs work". It
is capability scaffolding with a fail-closed default, pending a real run.

## 3. The catalogue the UI lists

`ModelAdapterRegistry.list_adapters()` returns entries built from spec tables,
not from loaded weights (`backend/science/models/adapter_registry.py`):

| Group | Ids |
| --- | --- |
| GPT-2 | `gpt2`, `gpt2-small`, `gpt2-medium`, `gpt2-large` |
| TransformerLens | `tl-gpt2`, `tl-gpt2-small`, `tl-gemma-2b` |
| Gemma | per `GEMMA` spec table |
| Llama | per `LLAMA` spec table (incl. `tinyllama-1.1b`) |
| Qwen | per `QWEN` spec table |
| Mistral | per `MISTRAL` spec table |
| DeepSeek | per `DEEPSEEK` spec table |

Being in this list means an adapter class exists with architecture metadata. It
does **not** mean the weights are present or that a run was verified. Only `gpt2`
has a live engine behind it; `mech models inspect <id>` reports the architecture
of the weights actually in memory, which is why it will refuse or report
unavailable for ids that are not loaded.

Registered-but-unimplemented capabilities are declared explicitly:
`UNIMPLEMENTED_CAPABILITIES = {deepseek, mistral, cloud, cluster}`
(`backend/core/execution_status.py:117-124`). `CapabilityRegistry.supports()`
returns `False` for them and `status_for()` returns `NOT_EXECUTED`; unknown ids
also return `NOT_EXECUTED` rather than raising.

## 4. Limitations worth knowing before you cite anything

1. **One model, one hook point at a time.** The engine holds one loaded model
   under `backend/services/model_lock.py`; runs serialise.
2. **Token-position conventions.** `layer_activations` returns *all* positions
   (`resid_post` = `hidden[layer+1]`, `mlp_post` = post-GELU MLP), rounded to
   4 dp for JSON transport, plus three summary stats. The logit lens, ablation
   and steering paths score the **last token** only.
3. **Residual stream has n_layers + 1 entries for GPT-2.** `[0]` is the token
   embedding, `[1..11]` raw block outputs, `[12]` is the residual *after* `ln_f`
   — already normalised, not a raw block output. The engine states this in
   `architecture()["residual_convention"]`. It is GPT-2-specific and must not be
   assumed for other architectures.
4. **`layer_activations` payload size is unbounded by row count.** It serialises
   every position × every dimension as JSON, so a long prompt produces a large
   response. The 2 MiB request-body cap does not bound the *response*.
5. **No multi-model comparison in one process.** Comparing two models means two
   runs and a comparison of records, not two resident models.
6. **`gpt2-medium` / `gpt2-large` are catalogue entries through `GPT2Adapter`**,
   not engine targets. Loading them is not wired to `gpt2_engine`.
7. **Third-party package absence is not a MECH defect.** `sae_lens` and `nnsight`
   are optional; their absence is reported by name, and installing them changes
   what `mech doctor` says without a code change.

## 5. Diagnostics

- `GET /api/models` (dispatcher) — catalogue as the UI shows it.
- `GET /api/models/{name}` / `mech models inspect <id>` — architecture of the
  weights actually in memory.
- `GET /health/subsystems` — probes process, dependency, model, executor,
  storage and evidence health; the vocabulary deliberately has no "healthy"
  member, only what a probe reported.
- `mech doctor` — the CLI equivalent: Python version, ML-stack importability,
  which checkpoint the engine serves, storage writability, SAE loader
  importability and whether an SAE checkpoint is configured. It does **not**
  probe `nnsight`/`sae_lens` package presence today; use
  `python -c "import importlib.util as u; print(bool(u.find_spec('nnsight')))"`
  or the adapter's own `describe()` for that.
  Note: `doctor` prints "weights load on demand from local files (never
  downloaded)" for the engine checkpoint, but the engine's own `load()` falls
  back to a Hugging Face download when the local cache is empty
  (`gpt2_engine.py:122-127`). The `doctor` string is wrong; the engine's
  behaviour is what this page documents.

## See also

- [experiments.md](experiments.md) — provenance recorded per run.
- [sae-lab.md](sae-lab.md) — SAE sources, metrics and limits.
- [logit-lens.md](logit-lens.md) — the projection this model is read through.
