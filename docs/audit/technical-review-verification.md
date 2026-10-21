# Technical Review — Verification of the "What Is Genuinely Built" Claim Table

**Date:** 2026-10-20
**Scope:** the eight claims in the request, verified by execution against the working tree at
`C:\Users\himan\Downloads\app test\MECH`.
**Method:** six parallel audit tracks (model weights, ACDC, SAE, refusal semantics, security/API,
a11y/build) plus a Lead cross-cutting scan that stripped all comments and docstrings from
`backend/**.py` and hunted for live fabricated metrics.
**Repo state:** not modified by the audit team. All audit artefacts are under `_audit_scratch/`
(gitignored). This report is the only file added.

> ⚠️ **Concurrency caveat.** Another agent session was actively editing the same tree throughout
> the audit — ≥5 concurrent pytest runs, and live edits to `backend/main.py`,
> `backend/services/gpt2_engine.py`, `backend/api/project_export.py` and `backend/plugins/*`.
> Every line number in this report was **re-verified against the tree as it stands at the end of
> the audit**; where a finding was fixed mid-audit it is marked as such (§4 #23). One test run
> was invalidated by the source-stability guard as a result (§6).

---

## 1. Verdict at a glance

| # | Claim | Verdict |
|---|-------|---------|
| 1 | Real GPT-2 Small weights, not mocks | **CONFIRMED** for the engine path; **2 adjacent refutations** |
| 2 | Real ACDC, not a label | **PARTIALLY CONFIRMED** — real measurement, but not the Conmy et al. algorithm; FVE absent |
| 3 | SAE loader honest and correct | **CONFIRMED** — 51 vs 1558 reproduced exactly; the `sae_lens` path re-confirmed against the real package |
| 4 | Refusal semantics enforced | **PARTIALLY CONFIRMED** — vocabulary holds, enforcement leaks in 3 places |
| 5 | Security posture | **PARTIALLY CONFIRMED** — 4 controls real and **now observed running**, 6 gaps, 2 file misattributions |
| 6 | API breadth: 69 endpoints | **CONFIRMED** at 69 for the dispatcher; true total is 82 |
| 7 | Accessibility: 88/86/36 | **PARTIALLY CONFIRMED** — real work, numbers stale, scope misstated |
| 8 | Build artifacts | **PARTIALLY CONFIRMED** — `dist` exact; both `.exe` are 3 days stale |

Nothing in the claim table is a fabrication. The problem is not that the claims are false — it is
that three of them are **more flattering than the code**, and there are **59 catalogued defects**
underneath (11 HIGH, 48 MEDIUM/LOW), several of which produce results stamped
`provenance: "live"`.

**Second pass.** After the first report, the three open gaps were closed:
the full test suite was replaced by a **per-file matrix** over all 160 test files; the Electron
app was **launched and observed** rather than read; and the `sae_lens` path was re-tested against
the **real package** (6.54.12) instead of a stub. Three of this report's own conclusions
changed as a result — the token-in-URL finding, the SAELens provider, and the 51-vs-1558
premise — and one **new deterministic test failure** was found. §4.1 records the fix status of
every finding at the end of the audit: **39 of 40 re-checked findings are still open.**

---

## 2. What is genuinely real (verified, not inferred)

### 2.1 GPT-2 Small — the strongest evidence in the repo
- `backend/services/gpt2_engine.py` loads the **canonical published checkpoint**. 148/148
  `state_dict` tensors are **byte-identical** to `model.safetensors`
  (sha256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`); the one
  "mismatch" is `lm_head.weight`, which is tied to `wte` and byte-equal. Revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e` matches the live HuggingFace API.
- **The fingerprint is a hash of weight values**, not metadata:
  `backend/science/reproducibility/model_fingerprint.py:60-72` streams raw little-endian float32
  bytes of every tensor (name + dtype + shape prefixed, sorted by key).
  `sha256:113687a222f8cf98039222c27b39aaf716493e5e8c1db94ea4e6544e0814088c`
  **Perturbing a single element** (`transformer.h.5.mlp.c_fc.weight[0,0]`) flips it to
  `sha256:a37d1a917ac722ae1504c121d2c3f38cd5ccfcd841b46ab114ad8e1c8d69d153`; restoring
  reproduces the original. An independent re-derivation matched exactly.
- Param count is **124,439,808** — the published GPT-2 Small count, read from the weights.
- Behaviour is real GPT-2, not random: `"The Eiffel Tower is in the city of"` → ` Paris` rank 1;
  IOI on `Mary/John` gives clean `+2.9548` / corrupted `-3.6654` with the correct sign.
- The TransformerLens adapter loads the same weights: identical top-5 token ids, logits equal to
  a constant 116.45277 ± 2.9e-05 (TL's `center_unembed` shift).

### 2.2 SAE loader — honest, correct, and the 51/1558 figures are real
- `backend/interpretability/sae/loader.py` **refuses `weights=None`**; no caller catches the
  refusal and fabricates a result (dispatcher → `unavailable`, CLI → `NOT_EXECUTED`,
  `attested:False` blocks publication).
- `acts = ReLU((h − b_pre) @ W_enc + b_enc)` on real tensors.
- **Reproduced exactly with the real 24,576-wide jbloom checkpoint and real GPT-2 activations:**

| prompt | centred on `b_dec` (correct) | uncentred (the bug) |
|---|---|---|
| `The capital of France is` | **51 active, relL2 0.2428** | 1558 active, relL2 5.5594 |
| `The president of the United States is` | 53 active, relL2 0.2474 | 1459 active, relL2 5.5115 |
| `Water boils at one hundred` | 56 active, relL2 0.2981 | 1087 active, relL2 4.1563 |

  Mean-baseline relL2 is 0.6369 — so the centred reconstruction (0.2428) beats predicting the
  mean, while the uncentred one (5.56) is **5.6× the norm of the signal**, i.e. worse than
  predicting zero. **51 is the correct count; 1558 is the noise signature.**
  (Note: the original audit brief asserted the opposite. The repository is right and the brief
  was wrong — a calibration point worth recording.)
- `51` and `1558` appear **only in prose**. No test pins those numbers; the live test pins the
  *invariant* (`uncentered_error > error*4`), so the figures are host-specific measurements, not
  baked-in metrics.
- **Re-confirmed against the real `sae_lens` package** (6.54.12, installed for this audit):
  the SAELens code path was also exercised against genuine SAELens objects and files rather than
  a stub — which changed two findings for the worse (§4 #26, #27).

### 2.3 ACDC — real computation, genuinely not a label
- `backend/interpretability/discovery/algorithms/acdc.py` (442 lines) performs **144 real
  head-level interventions** on live GPT-2 Small (CUDA/fp16) in 20.8 s.
- Every reported number reproduces by hand: 4/4 sampled heads matched exactly
  (e.g. `L9H9` recovery −0.5, `L0H0` 0.3333, `L5H3` 0.3333, `L9H10` 8.0).
- The threshold is live (retained 144 → 138 → 91 → 59 → 8 as the threshold goes 0 → 0.05 → 0.5 →
  1 → 5), and the retained set genuinely depends on the prompt (138 heads on `ioi_0001`, 124 on
  `ioi_0002`).
- The prior fabrication pass is real and holds: no `L9H9`/`L10H0` backfill, no `confidence: 0.95`,
  no `0.90 + 0.09·(1−score)` fidelity formula, no random tensors, no `torch.no_grad` stub.
  `confidence` is conservatively `0.0`; the final edge's weight is `None` with a stated reason.
- Only **one** ACDC implementation exists in `backend/`, and it is the one the registry dispatches.

### 2.4 Refusal vocabulary is genuinely distinct
```
backend/core/execution_status.py:38  IMPLEMENTED  = "IMPLEMENTED"
                                     :42  NOT_EXECUTED = "NOT_EXECUTED"
                                     :46  UNAVAILABLE  = "UNAVAILABLE"
                                     :49  FAILED       = "FAILED"
```
Distinct constants, never merged. A `NOT_EXECUTED` payload carries **no data fields** (verified
against an allow-list). The CLI exits 3 for it. An AST scan of every `except` handler in
`backend/` found **0 sites** that convert an exception into a success status, **0** that zero a
metric, and **3** that empty a result (all benign). The laundering AGENTS.md forbids has already
been repaired across the tree, and the repair is documented in place.

### 2.5 Security controls are real and were probed live
| Control | Result |
|---|---|
| Loopback bind | `backend/main.py:429` binds `127.0.0.1`; a socket-bind test proved nothing listens on `0.0.0.0`; LAN-IP curl → refused |
| Bearer token | no token → **401**, wrong → **401**, correct → **200**; empty / prefix / suffix / Basic → **401** |
| Constant-time compare | `hmac.compare_digest` at `backend/core/auth.py:133` |
| No default token | `secrets.token_urlsafe(32)` at `auth.py:112` |
| CORS allow-list | explicit list, **no `"*"`**; evil origin gets no ACAO; preflight → **400 `Disallowed CORS origin`**; `localhost.evil.com` correctly rejected by the anchored regex |
| `contextIsolation: true` / `nodeIntegration: false` | `frontend/electron/main.js:54-55` |
| Docs/OpenAPI/MCP gated | `/docs`, `/openapi.json`, `/redoc`, `/mcp`, `/health/subsystems` all **401** without a token; only `/health` is public, by design |
| Path traversal | `/api/..%2f..%2fetc%2fpasswd` → 404; no traversal found in the StaticFiles mount |
| Stack-trace leaks | none — all probed error paths return structured JSON |

**All 82 logical endpoints were probed live** with a valid token: 70×200 plus domain-correct
400/404/409/422, **zero shadowed or unreachable routes, zero cross-module path collisions**.
The Electron-side controls were additionally **observed in a running app** — see §2.9.

### 2.6 Endpoint count — the number 69 is exactly right
`grep -cE '^@router\.(get|post|put|patch|delete)\(' backend/api/dispatcher.py` → **69**, confirmed
a second time by walking the live FastAPI route tree (69 operations over 63 distinct paths; six
paths serve two methods each). The `/api/v2/jobs` list/get/cancel routes exist and work
end-to-end, including a real 202 → 200 → 404 → 409 sequence.

### 2.7 Build artefacts — `frontend/dist` is exact and provably current
- **1,089,568 bytes = 1.09 MB / 1.039 MiB**, 77 files. The claim's "1.1 MB" is accurate.
  (`du -sh` reports 1.3M — that is Windows 4KB block allocation, not a discrepancy.)
- **77/77 files byte-identical (SHA-256)** to a fresh rebuild of the current source, built into a
  scratch directory. Zero missing chunk references, zero orphans, and **0 source files newer than
  `dist/index.html`** — the bundle is not stale.
- `frontend/release/MECH Platform Setup 2.0.0.exe` — 92,566,275 bytes, genuine NSIS PE
  (`MZ` / `PE\0\0` / `Nullsoft` strings), VersionInfo `MECH Platform` `2.0.0`.
- `frontend/release/win-unpacked/MECH Platform.exe` — 186,328,576 bytes, x86-64 PE.
- electron-builder config (`version 2.0.0`, `productName "MECH Platform"`) matches the artefact names.

### 2.8 Accessibility work is real, not cosmetic
375 `role=` attributes, 136 `aria-labelledby`, 24 `sr-only` occurrences, 11 `@keydown` handlers,
`useId()` for duplicate-prone ids, **0** `div`/`span` with `@click` lacking `role`+`tabindex`,
**0** icon-only buttons without an accessible name, **0** dangling ARIA refs (the `Settings.vue`
`aria-labelledby` values genuinely resolve). The distribution is flat — the top contributor holds
5.8% of all attributes, so the counts are not inflated by one file.

### 2.9 The Electron desktop app — now OBSERVED RUNNING, not just read

The first pass read the Electron controls from source. The app was subsequently **launched for
real** (Electron 32.3.3 driven by Playwright, serving the actual built `frontend/dist`), and
every control below was observed in the running process:

| Control | Static claim | **Runtime observation** |
|---|---|---|
| `contextIsolation: true` | `main.js:54` | ✅ **Confirmed.** In the page's main world `require`, `process`, `module`, `exports` are **all `undefined`**; the 36-method `contextBridge` surface (`appApi`) is present and functional |
| `nodeIntegration: false` | `main.js:55` | ✅ **Confirmed** — same observation |
| Preload bridge works | — | ✅ `window.appApi` exposes 36 methods; `getApiToken()` returns the live token |
| The app actually works | — | ✅ **The app boots and talks to the backend.** `main_window_ready` logged; the renderer issued `GET /api/models` → **200**, `POST /api/gpt2/fresh_prompt` → **200**, `GET /api/status` (with token) → **200**; `GET /health` → **200** |
| **CSP** | absent in `frontend/index.html` | ❌ **Confirmed absent at runtime.** The loaded document has **zero** `<meta>` tags, and **Electron itself logged**: *"Electron Security Warning (Insecure Content-Security-Policy) — This renderer process has either no Content Security Policy set or a policy with 'unsafe-eval' enabled."* |
| Token in the page main world | `main.js:65` | ❌ **Confirmed.** `window.__MECH_API_TOKEN__` is the 43-char live token, and it is **identical** to the copy in `localStorage['mech_api_token']` and to `appApi.getApiToken()` — all three same value, all reachable from renderer JS |
| `sandbox: false` | `main.js:56` | ❌ **Confirmed** (the app also had to be launched with `--no-sandbox` in this environment, where Electron aborts with `STATUS_BREAKPOINT` without it) |
| `will-navigate` blocking | `main.js:243-261` | ✅ **Confirmed.** A renderer-initiated navigation to `https://example.com/should-not-load` was **blocked**; the page stayed at `http://127.0.0.1:5173/` |
| `setWindowOpenHandler` | `main.js:84-87` | ✅ **Confirmed.** `window.open('https://example.com/')` returned `null` — no new window |
| **`shell.openExternal` has no validation** | `ipc/events.js:94-98` | ❌ **Confirmed at runtime.** The main process's `shell.openExternal` was patched to record instead of open, and **all four hostile URLs reached it unmodified**: `file:///C:/Windows/System32/drivers/etc/hosts`, `javascript:void(0)`, `\\evil-host\share\payload` (UNC → NTLM relay), and `mech-probe-scheme://x`. Every IPC call resolved `{ok: true}`. **Nothing was opened** — the interception made the test free of side effects |
| **The IPC bridge 401s** | `ipc/runtime.js:19-23` | ❌ **Confirmed at runtime.** `appApi.pythonPing()` → `{"ok":false,"error":"backend ping -> HTTP 401"}`; `appApi.getRuntimeStatus()` → `{"status":"error","error":"backend runtime/status -> HTTP 401"}`; `appApi.pythonCall('status',{})` → threw `backend status -> HTTP 401`. The backend access log shows `POST /api/ping 401`, `POST /api/runtime/status 401`, `POST /api/status 401`, and the app logged `runtime_status_error` |

Two further runtime facts fell out:

- **A 401 is invisible to the browser.** A no-token `fetch('/api/status')` from the renderer did
  not return 401 — it failed with *"Failed to fetch"*, because the CORS layer withholds
  `Access-Control-Allow-Origin` on an error response. The 200-with-token retry then succeeded.
  A caller therefore cannot distinguish "not authorised" from "backend down".
- **The token does NOT reach the app's SQLite database.** The app's in-memory logger captured 13
  `backend_stdout` entries (full request URLs) — the channel by which a `?token=` request would
  arrive — but the `logs` table in `app.db` was **empty (0 rows)** after the run, because
  `Logger.attachStorage()` is defined at `frontend/electron/logger.js:11` and **called from
  nowhere**. The disk leg of the token finding therefore stops at the uvicorn access log, not at
  the app's database. *(This corrects the first pass of this audit, which asserted the SQLite
  leg.)*

---

## 3. Where the claims overstate the code

### 3.1 "Real ACDC" is not ACDC
The claim cites *"Conmy et al. activation caching, greedy reverse-topological edge pruning,
logit-diff/FVE metric."* Two of those three do not exist in the code:

| Paper element | Reality |
|---|---|
| Computational graph / **edge** pruning | **Refuted.** Output is a linear chain (140 nodes / 139 edges) whose head-to-head edges were **never measured** and that jump over pruned heads (`H_L10_H7 → H_L10_H9`, skipping pruned L10H8) |
| **Reverse-topological** traversal | **Refuted — cosmetic.** `reversed(range(num_layers))` over a flat `(layer, head)` list. **Proof: shuffling the candidate order at runtime yields an identical retained set** (138 vs 138, symmetric difference `[]`) |
| Clean **and** corrupted activation caching | **Partial.** Only the clean prompt's head vectors are captured; the corrupted prompt's activations never are, and `cache.corrupted_logits` is an unused dict |
| Greedy threshold keep/drop | **Partial.** Per-head, non-sequential, and **direction-insensitive**: **61 of 138 retained heads have *negative* recovery** (worst −8.08) and are still promoted to circuit members with `confidence = -8.08` |
| Real logit metric | **Verified**, but it is a single-token clean-vs-corrupt gap, not the paper's IO-minus-subject logit difference — that branch is dead (bundled datasets lack `io_id`/`subject_id`) |
| **FVE** | **Refuted — not implemented anywhere in `backend/`.** `grep -i "fve\|variance_explained"` over `backend/**.py` returns 9 hits and **none of them is a computation**: the docstring at `acdc.py:8`, a *target* in `configs.py:81`, `"fve_variance_explained": None` placeholders in `transcoders.py:138`, and two **hardcoded demo fixtures** in `discovery_memory.py:172, 206` (`"fve": 0.94`, `"fve": 0.97`). No report contains an FVE key |

Also: the whole-circuit fidelity is **negative** (−3.0755 — injecting all 138 retained heads moves
the target logit *further* from clean than the corrupted run did) while the run still reports
`status=completed, measured=True`. And the "circuit" is 138 of 144 heads (96% retained).

### 3.2 Two of the three "weight fingerprints" are not fingerprints of the published weights
| Fingerprint | Value | What it actually hashes |
|---|---|---|
| Engine | `sha256:113687a2…` | ✅ the published checkpoint's raw tensor bytes |
| TransformerLens | `4c8d4ce7…` | ❌ **TL-reparameterised** weights — under MECH's defaults (`fold_ln`/`center_writing_weights`/`center_unembed`), **0 of 938 tensors** are byte-identical to the checkpoint; the LayerNorm parameters do not even exist (they are folded) |
| Discovery registry | `sha256:e9a4f81c9b201a34d` | ❌ **a literal string** — 17 hex chars, not 64 |

Three mutually incomparable "weight identities" for one model, none cross-comparable, and the
third is fabricated. `backend/interpretability/discovery/mechanism_registry.py:16` ships
`model_checkpoint_sha = "sha256:e9a4f81c9b201a34d"`, `commit_sha = "git:a4f81c9"`,
`environment_hash = "env_py311_torch2.2"` (torch here is 2.14.0) and
`cuda_driver = "CUDA 12.2 / Driver 535.12"` as the provenance of the seeded `mech_ioi`
mechanism, surfaced through `discovery_engine.py:62`. Executed: the seeded entry's
`provenance_record` carries all four. A SHA-256 is 64 hex characters; this one is 17.

### 3.3 "NOT_EXECUTED ≠ FAILED" is true as vocabulary, false as enforcement
`NOT_EXECUTED` and `FAILED` are distinct constants — but `backend/core/provenance.py:73-76`
builds `_FAILURE_STATUSES` from `{"error","failed","failure","unavailable","not_run","notrun",
"blocked","skipped","timeout","invalid","rejected","declined"}`. **`"not_executed"` is not in it,
and neither is `"running"`.** `is_failure_status` lowercases, so:

```
NOT_EXECUTED   is_failure_status -> False     <-- passes the gate
FAILED         is_failure_status -> True
UNAVAILABLE    is_failure_status -> True
RUNNING        is_failure_status -> False     <-- passes the gate
```

Consequently `attest_measurement` stamps a registered-but-unimplemented engine's refusal with
`provenance: "live"`, `attested: True`, and mints a **trusted-key-signed `measurement_record`** —
identical in strength to a real measurement. Only the separate `_is_completed` status check keeps
it out of validation and publication. `backend/agents/evidence_policy.py:25-69` describes itself
as "the single source of truth for scientific attestation" and returns True for it.

### 3.4 The legacy-evidence quarantine has no timestamp criterion — and three gaps
The claim cites "which schema version, which timestamp". There is **no timestamp criterion**: the
quarantine reads only `schema_version == 1` and `record is a dict`
(`backend/core/evidence_graph.py:492-495`). It **is** genuinely wired and called at startup
(`backend/main.py:50`, inside the FastAPI lifespan) — that part of the claim holds. But:

- **GAP-A** — a legacy record written *after* the startup pass is never quarantined. It is then
  listed by `GET /api/society/runs` as `status: "completed"` and served by
  `GET /api/society/runs/{id}` with `result.provenance == "live"` and
  `validation_eligible/publication_eligible == true`. `envelope_status` does report
  `{"envelope":"legacy","attested":false}` — but that is advisory, and a consumer reading
  `body["result"]["provenance"]` sees `"live"`.
- **GAP-B** — the eligibility stamp is applied to the **copy**, not the source. A failed
  `os.unlink` leaves the legacy record live in the authoritative directory, unstamped. `main.py:51`
  only logs when `report.get("moved")` is truthy, and `main.py:54-55` swallows any exception —
  so the failure is silent.
- **GAP-C** — a **future** `schema_version` is treated as legacy and **destroyed**: moved, and
  eligibility flags forced False. An older build silently closes a newer build's signed records.

### 3.5 The API count and the security file attributions
- `69` is exactly right **for the dispatcher**, but the claim undercounts the rest:
  `runtime_api.py` has **four** `/jobs` routes (list / **create** / get / cancel), and
  `v1_contracts` (6) and `events` (3) go unmentioned. **True total: 82 logical endpoints /
  75 distinct paths / 164 mounted routes / 153 OpenAPI paths — +10 over the claim.**
- The loopback bind and the CORS allow-list live in **`backend/main.py`**, not
  `frontend/electron/main.js` as the claim's evidence column states.

### 3.6 The accessibility numbers are real but stale, and the scope is wrong
- The frontend is **Vue 3 SFC**, not React. The claim's wording (".jsx/.tsx/.js/.ts") yields
  **0** aria-label and **0** aria-live — those extensions hold only services/stores/utils.
  All a11y markup is in 44 `.vue` files.
- **88 and 86 are exactly the git-HEAD counts** (commit `a978853`) — a true measurement of the
  committed tree, now understated by 9 and 6 because four uncommitted views were added.
  Current source: **97 aria-label / 92 aria-live** (plus 136 `aria-labelledby`, 238 `aria-hidden`,
  275 `role=`).
- "36 component files instrumented" reproduces at HEAD only under a **`role=`-based** count. The
  honest current figure is **37 of 43 component `.vue` files (86.0%)**, or **33 of 35 routed
  views (94.3%)**.

### 3.7 The installers ship a stale renderer, but a current backend

`frontend/release/*` was built 2026-10-17; `frontend/dist` was rebuilt 2026-10-20. The packaged
`app.asar` references `index-1-Dt3nak.js` while current source references
`index-CYh5f3sH.js`, and it contains **no `ActivationBrowserView`, `DatasetsView`,
`LogitLensView` or `SAELabView` chunks at all** — the shipped app runs a renderer that predates
four entire views and the current a11y work. `MECH-standalone/frontend/dist` is stale the same
way. `favicon.ico` is a **0-byte stub** in `dist/`, `public/` and `MECH-standalone/`.

**The Python side is not stale, and that matters.** A file-by-file SHA-256 comparison of the
whole backend tree against both shipped copies:

| copy | `.py` files | byte-identical to source | differ | only in one side |
|---|---|---|---|---|
| `frontend/release/win-unpacked/resources/backend/` | 478 | **478 (100%)** | 0 | 0 |
| `MECH-standalone/backend/` | 478 | **478 (100%)** | 0 | 0 |

So the installers ship the **current backend** and a **three-day-old renderer**. Every backend
finding in this report applies to the shipped app as-is; the four missing views and the a11y
work are a renderer-only gap.

---

## 4. Material defects, by severity

### HIGH
1. **Cache tearing in the production GPT-2 API.**
   `backend/services/gpt2_engine.py:725 / 854 / 985 / 1204` (`layer_detail`, `list_neurons`,
   `neuron_detail`, `head_detail`) read `_cache["mlp_post"][layer]`, `_cache["str_tokens"]` and
   `_cache["prompt"]` as **separate unlocked reads**, while `_forward_locked()` rebinds the
   module-global `_cache` under `MODEL_LOCK`. Re-verified against the current tree: `with
   MODEL_LOCK:` appears only at `gpt2_engine.py:574, 691, 1272, 1296, 1333, 1438, 1495` — i.e.
   in `_forward_locked`, `generate_text`, `attention_head`, `activations`, `patch_head` and
   `patch_neuron` — and **not** in the four functions above. Measured over 400 concurrent
   iterations:
   **44/400** `list_neurons` and **53/400** `neuron_detail` responses carried a token string
   that does not occur in the prompt the same response names — while still stamped
   `provenance:"live"`, `attested:true`. These are exactly the handlers behind
   `/api/gpt2/{layer,neurons,neuron,head}`. `dispatcher.py:1205-1208` claims "readers now do
   ensure-then-read atomically under MODEL_LOCK" — that fix covers only `activations` and
   `attention_head`; the comment overstates it.
2. **A refusal is attested as a live, signed measurement.** `provenance.py:73-76` + `:449-453`
   (see §3.3). Fix direction: add `"not_executed"` and `"running"` to `_FAILURE_STATUSES`.
3. **ACDC measures the wrong token for every non-default prompt.** `acdc.py:166` reads
   `dataset.get("target_token", " Mary")`, but the bundled IOI dataset uses the key **`"target"`**.
   `ioi_0002` (answer `" Bob"`) was measured on `" Mary"`, producing `baseline_logit_diff =
   -0.375` and `fidelity = 13.1577`.
4. **ACDC's capture failure is silently reported as a successful 144-candidate live sweep.**
   `acdc.py:100-104` stores the exception text in `self._capture_error`, which is **never read**.
   Forcing the capture to raise yields `retained=0, pruned=144` with
   `provenance.source='live'`, `measured=True`, `candidates_evaluated=144` — and the report does
   not mention the failure at all.
5. **ACDC's fidelity is measured on a different model instance than its own sweep.**
   `acdc.py:340/357` → `live_measure.py:46-53` uses `backend.services.gpt2_engine`
   (CPU / fp32 / eager) while the sweep uses the adapter (CUDA / fp16 / sdpa). The same gap
   reads 0.75 from one and 0.7877 from the other — 5.03% apart.
6. **Legacy evidence that escapes quarantine is served as a current completed run**
   (`evidence_graph.py:750-755, 806-841, 921-938`; `main.py:48-55`; `dispatcher.py:2342-2375`).
7. **`GPT2Model` methods that crash on every call — 3 of 22 public methods.**
   `prediction_trace` (`gpt2_model.py:1361`) and `circuit_trace` (`:1406`) raise
   `AttributeError: 'ActivationCache' object has no attribute 'get'` (the only `.get()` on an
   object that supports `__getitem__` but not `.get`); `prompt_compare` (`:1516`) raises
   `IndexError: too many indices for array: array is 3-dimensional, but 4 were indexed`.
8. **The Electron IPC bridge sends no `Authorization` header — CONFIRMED AT RUNTIME.**
   `frontend/electron/ipc/runtime.js:19-23` sets only `Content-Type`. With the real app
   running: `appApi.pythonPing()` → `{"ok":false,"error":"backend ping -> HTTP 401"}`;
   `appApi.getRuntimeStatus()` → `{"status":"error","error":"backend runtime/status -> HTTP 401"}`;
   `appApi.pythonCall('status',{})` → threw `backend status -> HTTP 401`. The backend access log
   shows the three 401s and the app logged `runtime_status_error`. *Severity qualifier found by
   the Lead:* **0 `invoke()` calls and only 2 `appApi.*` references exist in `frontend/src`**
   (`services/api.ts:34, 48` — both `getApiToken()`), so the renderer talks to the backend
   directly through `frontend/src/services/api.ts:53-65`, which *does* add the header. The
   desktop app's user-facing path works — **confirmed at runtime** (`/api/models` → 200,
   `/api/gpt2/fresh_prompt` → 200). The bridge is exposed, reachable, and broken, but the app
   does not currently use it: **latent**, not a live breakage.
9. **ACDC's "node patch" overwrites the head at *every* position**, not at the node's position
   (`gpt2_adapter.py:1071-1074`, `position=None` → `x[..., start:end] = v`). All-positions vs
   last-position-only differ on **41 of 48** heads, max |Δlogit| **4.9375** — the "node" effect
   is conflated with the effect of rewriting the head's output at every earlier position.
10. **The token is accepted in the URL query string and written to disk.**
    `backend/main.py:183-184` accepts `?token=` / `?bearer_token=`; uvicorn logs the full URL,
    and `frontend/electron/main.js:168` pipes backend stdout into the app's logger. Executed
    proof of the disk leg is in the captured server log.
    **Corrected at runtime:** the log does **not** reach the app's SQLite `logs` table. The
    app's in-memory logger captured 13 `backend_stdout` entries (the channel a `?token=` request
    would arrive on), but `app.db`'s `logs` table was **empty (0 rows)** after a full app
    session, because `Logger.attachStorage()` is defined at `frontend/electron/logger.js:11` and
    **called from nowhere** — so `logger.storage` is always `null` and `storage.appendLog` is
    never invoked. The persistence path exists (`storage.js:270-273`) but is unreachable.
    **The disk leg stops at the uvicorn access log, not at the app's database.** (The first pass
    of this audit asserted the SQLite leg; the runtime probe refuted it.)
11. **The repo's own test suite has a deterministic failure — `GET /api/models/gpt2-small`
    takes ~27-31 s against a 10 s budget.** `tests/pytest/test_audit_all_features.py` builds its
    client as `TestClient(app, raise_server_exceptions=False)` (line 23) **without a `with`
    block**, so Starlette never runs the application's lifespan — and the lifespan is what
    pre-loads GPT-2 (`backend/main.py:64`). The first request that touches the model therefore
    absorbs the whole load inline. Reproduced **3/3**:
    `Latency 26285.71ms`, `31148.32ms`, `27417.83ms`, `27402.80ms` — all against
    `assert elapsed < 10000` (line 131). The same file's other 42 tests pass.
    **Mechanism isolated:** with the lifespan running, the preload completes in 34.8 s and the
    endpoint then answers in **124 ms** (HTTP 200) — comfortably inside the budget. The test
    is measuring a cold path the production app never takes (`main.js` waits for `/health`
    before showing the window), so the defect is in the **test harness**, not the endpoint:
    the client bypasses the startup path, and the 10 s budget is applied to a request that
    includes a model load. `test_audit_all_features.py` and `backend/main.py` are **unmodified
    vs HEAD**, so this is pre-existing, not a regression from concurrent work.

### MEDIUM
12. `pruning_ratio` is defined **twice** in `acdc.py:382,390`; the surviving literal reports the
    **retention** fraction, so `pruning_ratio = 0.958` sits next to `pruned_components = 6`.
13. ACDC's metric is **fp16-quantised**: a baseline gap of 0.75 against a 0.0625 quantum gives
    12 distinguishable levels, and the default threshold 0.05 sits *below* one quantum.
14. ACDC's retention criterion ignores `config.metric` and `config.search_space`; MLPs and
    neurons are never searched despite the registry advertising them.
15. `live_discovery.py:658-665, 724` turns an **undefined** recovery ratio into a literal `0.0`
    inside a live, attested result — violating "Empty outputs never become zeros".
16. `live_discovery.py:684-686` raises the evidence level to **`CAUSALLY_VALIDATED`** on a
    10-prompt sample-size threshold alone, with no held-out validation run.
17. **No CSP in the shipped renderer — CONFIRMED AT RUNTIME.** The Vite entry
    `frontend/index.html` has none; the CSP meta in `frontend/src/index.html:5-6` is a **dead
    file** — and its `connect-src` omits `http://127.0.0.1:8000`, the packaged app's own API
    base, so enabling it verbatim would break the app. `frontend/dist/index.html` (freshly
    rebuilt) has no CSP. With the real app running, the loaded document had **zero** `<meta>`
    tags and **Electron itself logged**: *"Electron Security Warning (Insecure
    Content-Security-Policy) — This renderer process has either no Content Security Policy set
    or a policy with 'unsafe-eval' enabled."*
18. Electron `sandbox: false` (`main.js:56`) and `webSecurity` never set — the preload runs with
    full Node privileges. (Confirmed at runtime; the app also required `--no-sandbox` to launch
    at all in this environment, where Electron aborts with `STATUS_BREAKPOINT` without it.)
19. `shell.openExternal` is called with **no scheme or origin allow-list** (`main.js:85`,
    `ipc/events.js:94-98`) — **CONFIRMED AT RUNTIME.** The main process's `shell.openExternal`
    was patched to record rather than open, and all four hostile URLs reached it **unmodified**:
    `file:///C:/Windows/System32/drivers/etc/hosts`, `javascript:void(0)`,
    `\\evil-host\share\payload` (UNC → NTLM relay) and `mech-probe-scheme://x`. Every IPC call
    resolved `{ok: true}`. Nothing was opened — the interception made the test side-effect free.
20. The token is exposed to the **page main world** and persisted in `localStorage`
    (`main.js:64-68`, `preload.js:31-32, 42-43`), so any renderer XSS exfiltrates it.
    **Confirmed at runtime**: `window.__MECH_API_TOKEN__`, `localStorage['mech_api_token']` and
    `appApi.getApiToken()` all returned the **same 43-character live token**.
21. The token file's `os.chmod(path, 0o600)` is a **no-op on Windows** — `icacls` shows the file
    group-readable. The docstring's "not world-readable" guarantee holds on POSIX only.
22. `gpt2_model.py:1564-1565` computes "prompt A" residual norms **after** running prompt B, so
    `residual_norms_a` and `residual_norms_b` are the same numbers. (Currently masked by the
    `prompt_compare` crash.)
23. `gpt2_model.py:753-786` `_compute_category_scores` returns **all ten categories = 1.0** from a
    single token: the reverse-substring test `if kw in token_lower or token_lower in kw` lets the
    1-char token `" i"` match every keyword containing "i", and `probability * 2.0` capped at 1.0
    saturates. It is returned to callers as "confidence scores (0-1)".
24. ~~`gpt2_engine.py` — an **unseeded** `random.choice` fallback prompt becomes the measured
    input (6 identical calls → 5 distinct prompts), and the payload carries no seed.~~
    **FIXED IN THE WORKING TREE DURING THIS AUDIT.** The finding was real when executed
    (`gpt2_engine.py:1430-1444` at the time). The tree now uses a fixed
    `_DEFAULT_MEASURE_PROMPT = "The capital of France is"` (`gpt2_engine.py:40`) and returns a
    `prompt_defaulted` flag (`gpt2_engine.py:1445, 1463`) recording that the engine chose the
    prompt rather than the caller naming it. The uncommitted diff states the reason explicitly:
    *"The previous `random.choice` made that choice unreproducible as well — two identical
    requests measured different prompts."* **A missing `seed` field remains open** — the
    response still carries no seed, so the fix addresses determinism but not full provenance.
25. `gpt2_engine.py:702` — `torch.multinomial` with no generator: every `temp > 0`
    generation is irreproducible and no seed is recorded. (`temp = 0` is deterministic.)
26. `gpt2_engine.py:92` — `weights_fingerprint()` swallows every exception and returns
    `None` without logging, silently downgrading the attestation to the name-derived fallback
    (`provenance.py:214-223`). Its cache (`gpt2_engine.py:59`) is never invalidated.
    `gpt2_engine.py:289` (`_measure_done`, defined at `:275`) uses the same `is_failure_status`
    as the provenance gate, so it inherits the `NOT_EXECUTED` hole in §3.3.
27. `sae/loader.py:474-485` — the loader is **half** SAELens-aware, and the missing half is the
    dangerous one. `_load_tensors` finds the encoder **weights** under any of
    `("W_enc", "encoder.weight", "encoder", "weight")` (line 447), but reads the **biases and the
    decoder only from top-level keys**: `state.get("b_pre")`, `state.get("b_enc")`,
    `state.get("W_dec")`, `state.get("b_dec")` (lines 474-485). A SAELens-format file therefore
    loads with the **right weights and no centring, no `b_enc`, and no decoder** — the exact
    uncentred bug the fix targets, re-entered through the front door, with no error and
    `pre_encoder_center == ""`.
    **CONFIRMED WITH THE REAL PACKAGE** (`sae_lens` 6.54.12, installed for the audit): a real
    SAELens-format `safetensors` file (`encoder.weight`, `encoder.b_enc`, `encoder.b_pre`,
    `decoder.weight`, `decoder.b_dec`) goes through `LocalCheckpointProvider` and comes back
    `loaded: True` with **no error and no warning**, but
    `pre_encoder_center: ''`, `b_pre: ABSENT`, `b_enc: ABSENT`, `decoder: ABSENT`,
    `bias_dec: ABSENT` — and `activate()` then returns `status: completed`,
    `measured: True`, `provenance: 'live'`. The same tensors in the flat layout the provider
    *does* understand give `pre_encoder_center: 'b_pre'` with the decoder and `b_dec` present.
    **So the same weights, in SAELens' own file format, are silently loaded uncentred and
    reported as a live measurement.** This is the worst form of the finding: not a refusal, a
    silently wrong result.
28. `sae/loader.py:597-620` — `SAELensProvider` loads no decoder and has no `b_dec` fallback.
    **CONFIRMED, AND WORSE THAN REPORTED:** with the real `sae_lens` package the provider
    **cannot load anything at all**. It calls `SaelensSAE.from_pretrained(identifier,
    device="cpu")`, but the current release requires a positional `sae_id`
    (`SAE.from_pretrained() missing 1 required positional argument: 'sae_id'`), and the SAE
    object has **no `encoder` submodule at all** in 6.54.12 — the tensors live directly on the
    SAE as `W_enc`, `b_enc`, `W_dec`, `b_dec`, so `sae.encoder.weight` would raise
    `AttributeError` even with the argument fixed. The provider was written against an older
    `sae_lens` API. `SAELoader.load_sae('saelens', ...)` therefore raises `SAELoadError`
    ("sae_lens could not load …") — so it **fails closed and fabricates nothing**, which is the
    right behaviour, but the provider is dead code against the current library.
29. `dispatcher.py:1825-1832` feeds the SAE `resid_post`/`mlp_post` while the dictionary's hook
    point is `blocks.{L}.hook_resid_pre`; on the real layer-8 dictionary, `resid_pre` gives
    relL2 0.2428 / 51 active vs `resid_post` 0.3098 / 126 active. The loader's own
    `assert_applies_to` guard is never called in production.
30. `sae/loader.py:260-313` — `reconstruction_error()` ignores the `feature_indices` it is
    handed; the error with 16 indices equals the error with 1.
31. `sae/loader.py:200-208` — `sparsity` is a constant `32/24576` regardless of input (true L0
    was 51), shipped to clients under the name `sparsity`.
32. `sae/loader.py` — **no activation threshold** (features with activation `0.0` are reported as
    fired) and **no NaN/Inf guard** (a NaN input yields `status: completed`, `measured: true`,
    and `json.dumps` emits a non-standard `NaN` literal).
33. `sae/cache.py:24-27` — the activation-cache key omits checkpoint identity, so two different
    checkpoints both at version `"latest"` collide and stale activations are served — directly
    against "stale results can never be presented as current".
34. `dispatcher.py:1276-1286` — `/gpt2/fresh_prompt` promises "never from a pool" and "no seeded
    or pooled text is ever substituted", but `_generate_fresh_prompt` falls back to
    `_random_prompt()` on any exception and `_default_prompt` returns a pool prompt when the
    engine is unavailable. The returned prompt then looks real to the caller.
35. `dispatcher.py:2296-2303` — the durable evidence write is inside `except Exception: pass`; a
    failed write loses the run's evidence while the API still reports it completed, silently.
36. `gpt2_engine.py:1614` (docstring), `:908`, `:1101` — `resid_post` is documented as
    `hidden[layer+1]`, which for the last layer is the post-`ln_f` final residual, not a block
    output.
37. `gpt2_model.py:1287` hardcodes `blocks.11.hook_resid_post`; `gpt2_model.py:649` hardcodes a
    `r > 40000` rank threshold; `:665` prints a hardcoded "of 50,257" vocab size.
38. `gpt2_model.py:76` calls `.numpy()` unconditionally — bfloat16 raises `TypeError`, so the
    TransformerLens-side hash is fp32-only.
39. Device split: `gpt2_engine` loads on **CPU** even with CUDA available, while `gpt2_model`
   uses cuda.

### LOW / TEST-QUALITY
40. `tests/pytest/test_evidence_quarantine.py:100-104` asserts only that the **source text** of
   `backend/main.py` contains the string `quarantine_legacy_records()`. It never runs the
    lifespan — the shape AGENTS.md explicitly forbids ("a test must never pass because a payload
    contained plausible-looking numbers"). The other 9 tests in that file are genuinely good.
41. `dispatcher.py:1387-1388` — the unavailable branch clamps `layer % 12` / `head % 12`, so a
    request for layer 15 is echoed as layer 3, the exact clamping the engine refuses to do.
42. `backend/ui/circuit_explorer.py:34-37` — `run_live_patch` returns `value * 0.1`; the method
    name claims a live causal patch. No importer anywhere; dead code.
43. `backend/interpretability/discovery/automated_regression_suite.py:12-23` — a regression
    verdict computed against a **hardcoded baseline** (`{"IOI":0.94, …}`), unlabelled. No caller.
44. `backend/api/dispatcher.py:234-236` — `/api/research_catalog` serves hardcoded catalogue
    entries with **no provenance label**, while the neighbouring `/benchmarks` endpoint does
    label them `"reference"`.
45. `backend/services/gpt2_engine.py:347`, `logit_lens_pipeline.py:182,185`,
    `model_fingerprint.py:171` write `attested: True` / `provenance: "live"` **outside** the
    single designated issuer (`provenance.py:479-482`). Bounded impact — no `measurement_record`
    is present, so the evidence gate still rejects them — but the payload asserts an attestation
    nobody issued.
46. `backend/mcp_server/server.py:51-52` hardcodes `{"status":"ok","provenance":"live"}` for
    every MCP tool response.
47. `frontend/src/App.vue:195-206` — no focus management on route change; `App.vue` contains no
    `.focus()` call and no `<h1>`.
48. **8 dead live regions** — `aria-live="polite"` on always-rendered, never-changing content, so
    nothing can ever be announced: `BenchmarkDashboard.vue:25`, `BenchmarkSuiteView.vue:25`,
    `EvidenceFusionView.vue:24`, `KnowledgeGraphView.vue:24`, `PaperReproductionView.vue:25`,
    `ReasoningTraceView.vue:24`, `ResearchAnalyticsView.vue:24`, `ScientificHealthView.vue:24`.
49. **6 unlabeled form controls**: `NeuralExplorerView.vue:9, 69, 371`;
    `TransformerExplorer.vue:118, 123`; `Sidebar.vue:20` (dead code).
50. **2 routed views with zero aria attributes**: `NeuralExplorerView.vue`, `TransformerExplorer.vue`.
51. **3 role regions with no accessible name**: `CampaignWorkspaceView.vue:220`,
    `ResearchSocietyView.vue:149`, `NeuronUMAPWrapper.vue:14`.
52. CORS allows the opaque `null` origin **with credentials** and any loopback port
    (`main.py:103, 114, 121`).
53. `backend/main.py:271-273, 416` + `auth.py:153-166` — standalone browser mode is broken:
    `GET /` is shadowed by the JSON root route and `/index.html` is 401 because `.html` is absent
    from the static-extension allow-list, so the SPA entry document is unreachable.
54. `backend/api/dispatcher.py:1399-1400` stacks two decorators on one handler (a legacy alias),
    inflating a naive decorator count by 1.
55. `gpt2_model.py:98` prepends BOS; `gpt2_engine.py:576` does not — the two adapters tokenize
    the same prompt differently.
56. ~~`MECH-standalone/backend/interpretability/sae/loader.py:221-225` and
    `frontend/release/win-unpacked/resources/backend/interpretability/sae/loader.py:221-225` still
    carry the **pre-fix** loader with the old `encoder.bias` fallback — the shipped desktop
    bundle contains the bug the source fixed.~~
    **WITHDRAWN — the first pass was wrong.** A full SHA-256 comparison of the backend tree
    against both shipped copies shows **478/478 files byte-identical** in
    `frontend/release/win-unpacked/resources/backend/` and **478/478** in
    `MECH-standalone/backend/`, `sae/loader.py` included (all three `53ec36eedfed66e4…`).
    There is no `encoder.bias` fallback anywhere — the current source says so explicitly at
    `loader.py:471-473`. The three copies are in sync.
57. `backend/api/dispatcher.py:15-16` — a stale header comment says the engine "falls back to
    seeded stand-ins when absent"; that behaviour was removed and the comment now contradicts
    the code.
58. `frontend/electron/logger.js:11` — `Logger.attachStorage(storage)` is **defined and never
    called** anywhere in `frontend/electron/`. `logger.storage` is therefore always `null`, so
    `storage.appendLog` (`storage.js:270-273`) is never invoked and the app's `logs` table stays
    empty. Confirmed at runtime: 0 rows after a full app session, against 40 in-memory log
    entries. The app's persisted-log feature is dead code — and, as a side effect, the reason
    the token-in-URL finding stops at the access log rather than reaching the app's database.
59. `backend/interpretability/discovery/discovery_memory.py:150-210` — `DiscoveryMemory.search()`
    appends **hardcoded** result items for circuits and counterexamples: `"IOI Name Mover
    Subgraph"`, `"L9H9 + L10H0 Name Mover circuit"`, `"Gemma Induction Head L5H1"`,
    `relevance_score` 0.95/0.94/0.88 and `metadata={"fve": 0.94}` / `{"fve": 0.97}` — with **no
    provenance label of any kind** (the file contains no `provenance`, `seeded`, `reference` or
    `measured` token). The claims/concepts/features/neurons sections next to them *are* real
    (computed from the registries by cosine similarity), so the fabricated part is mixed in with
    the genuine part of the same result set. `DiscoveryMemory` is **never imported anywhere else
    in `backend/`**, so this is dead code — but it is the only place in the tree where a
    fabricated FVE number exists, and it is exactly the kind of item that becomes a finding the
    day someone wires it up.

### 4.1 Fix status at the end of the audit

A concurrent agent session was editing the repository throughout. Every finding was
**re-verified mechanically against the tree as it stands at the end of the audit**
(`_audit_scratch/lead/recheck_findings.py`):

| | count |
|---|---|
| Mechanically re-checked | 40 |
| **Still open** | **39** |
| Fixed during the audit | 1 |

The single fix is the one in §4 #23 — the unseeded random prompt in `patch_neuron` — replaced
by a fixed `_DEFAULT_MEASURE_PROMPT` plus a `prompt_defaulted` flag, with the reason written
into the diff. Everything else this report lists is still open, including the two the
concurrent session was clearly working on (a new `tests/pytest/test_shared_model_lock_scope.py`
appeared at 10:02 targeting the cache-tearing defect, and `backend/agents/executor.py` changed
to pass a prompt through) — but `backend/services/gpt2_engine.py` is **byte-identical** to its
state during the audit (`sha256 99493c03be34db86…`), and `with MODEL_LOCK:` still appears only
at `gpt2_engine.py:574, 691, 1272, 1296, 1333, 1438, 1495, 1594, 1602, 1691, 1759` — **not**
inside `layer_detail`, `list_neurons`, `neuron_detail` or `head_detail`.

**A note on how this table was built, because it nearly lied.** The first version of the
checker tested `_FAILURE_STATUSES` with a regex for the literal `"not_executed"`, and reported
that finding as **FIXED** — the string was never in the file, so its absence proved nothing. The
behavioural check (`is_failure_status("NOT_EXECUTED") is False`) shows it is still open. A
pattern check can only tell you that a thing is *present*; for "was this removed?" you have to
run the code.

---

## 5. Corrected claim table

| Claim | Correct wording |
|---|---|
| Real GPT-2 Small weights | ✅ `backend/services/gpt2_engine.py` loads the canonical published checkpoint (148/148 tensors byte-identical) and its SHA-256 is a hash of weight **values** (perturbation-verified). ⚠️ The TransformerLens hash is of *reparameterised* weights, and `mechanism_registry.py:16` ships a fabricated checkpoint SHA. |
| Real ACDC | ✅ Real, measurement-based head-level causal search on live GPT-2 (144 interventions, hand-recomputable numbers, prompt-dependent). ❌ **Not** Conmy et al.: no graph, no edge pruning, reverse-topological order provably has no effect, single-token logit gap rather than IO-minus-subject, and **FVE is not implemented at all**. |
| SAE loader honest and correct | ✅ Refuses `weights=None` with no laundering; real encoding; correct `b_pre`/`b_dec` usage; **51 vs 1558 reproduced exactly** (and 51 is the correct one). ⚠️ SAELens-format files still re-enter the uncentred bug; the API feeds the wrong hook point. |
| Refusal semantics | ✅ `NOT_EXECUTED` ≠ `FAILED` as vocabulary, with a data-free payload; ✅ quarantine is genuinely called at startup. ❌ `NOT_EXECUTED`/`RUNNING` are **not** in `_FAILURE_STATUSES`, so a refusal is attested `live` + signed; the quarantine has **no timestamp criterion** and three demonstrated gaps. |
| Security posture | ✅ Loopback bind, bearer token with constant-time compare, CORS allow-list, `contextIsolation`/`nodeIntegration` — all real and probe-verified. ⚠️ Two of the four live in `backend/main.py`, not `main.js`; plus `sandbox:false`, no CSP, unvalidated `openExternal`, token in `localStorage`, and the token accepted in a URL (→ logged to disk). |
| API breadth | ✅ 69 in `dispatcher.py` is exact. ⚠️ `runtime_api.py` has **4** `/jobs` routes (list/create/get/cancel) and two routers go unmentioned — **82 logical endpoints** in total. |
| Accessibility | ✅ Real work (375 `role=`, 136 `aria-labelledby`, keyboard handlers, 0 div-click-without-role). ⚠️ 88/86 are the **git-HEAD** counts; current source is **97/92**, and "36" only reproduces under a `role=`-based count (now 37 of 43). The stated `.jsx/.tsx` scope is wrong for a Vue codebase — 0 aria attributes there. |
| Build artifacts | ✅ `frontend/dist` is **1,089,568 bytes (1.09 MB)**, 77 files, byte-identical (77/77) to a fresh rebuild of current source. ✅ Both `.exe` exist and are genuine PE binaries. ❌ Both are **3 days stale** and ship a renderer missing 4 whole views; `favicon.ico` is a 0-byte stub. |

---

## 6. Test-suite evidence

- `MECH_REQUIRE_WEIGHTS=1 python -m pytest tests/pytest/test_sae_checkpoint_live.py
  tests/pytest/test_real_implementations.py -q` → **19 passed in 31.09s**, with the
  source-stability attestation passing (658 watched source files identical at session start/end).
- **Semantics / anti-fabrication lane — 179 passed, 0 failed, 0 errors, 0 skipped** (816.49 s,
  isolated `--basetemp`), across 11 files: `test_attestation_laundering` (17),
  `test_audit_adversarial_loopholes` (14), `test_confidence_defaults_are_not_fabricated`,
  `test_evidence_envelope` (13), `test_evidence_quarantine` (9),
  `test_experiment_not_executed` (9), `test_measurement_authenticity` (12),
  `test_measurement_record` (10), `test_no_fabricated_results` (27),
  `test_no_seeded_records_reach_analytics` (1), `test_provenance_origination` (37),
  `test_validation_status_contract` (30).
  ⚠️ **The run's exit code was 1, and that is the source-stability guard, not a test failure**:
  *"RUN NOT TRUSTWORTHY — 11 of 663 watched source file(s) changed"* (`backend/main.py`,
  `backend/cli/main.py`, `backend/api/project_export.py`, `backend/plugins/*`,
  `backend/services/gpt2_engine.py`, `scripts/packaged_backend_guard.py`,
  `tests/pytest/test_packaged_backend_drift.py`, and three others). **Another agent session was
  editing the same tree during the run.** Per AGENTS.md this is a stability report, not a
  failure — but it also means the run is **not a trustworthy attestation**, and it explains the
  line-number drift noted in §4. **Re-run on a quiet tree before treating it as a green gate.**
  This is the lane that directly exercises the refusal-semantics claim — and it is green, which
  is why the gaps in §3.3 and §3.4 are **not** covered by the suite rather than being
  contradicted by it.
- `python -m pytest tests/pytest/test_evidence_quarantine.py -q` → 9 passed.
- The **full** `python -m pytest tests/pytest -q` was **not** completed: ≥5 other pytest sessions
  were running concurrently on this machine, each loading GPT-2 weights (0.5–4.5 GB RSS), and a
  shared `--basetemp` (`.pytest-tmp`, `pytest.ini:4`) caused a run to stall at 42% with 0 failures
  before being killed. Treat the full suite as **unverified** — re-run it on a quiet machine with
  an isolated `--basetemp`.

---

## 6.1 AGENTS.md rule-by-rule verdict

The repo's own non-negotiable rules, each checked against the code and the executed evidence:

| AGENTS.md rule | Verdict | Evidence |
|---|---|---|
| "An exception is never converted into a successful result" | **HOLDS** | An AST scan of every `except` handler in `backend/` found **0** sites that set a success status, **0** that zero a metric, and 3 that empty a result (all non-measurements). 58 except-blocks set a failure status. The laundering has already been repaired across the tree, and the repair is documented in place (`executor.py:220-228`, `discoverer.py:35-79`, `critic.py:280-290`, `dispatcher.py:1211-1223`, `execution_backend.py:36-91`, `aggregates.py:13-19`, `logit_lens.py:5-32`). |
| "A registered-but-unimplemented engine reports `NOT_EXECUTED`, distinct from `FAILED`, and carries no data" | **HOLDS — with a hole** | Five distinct constants (`execution_status.py:38/42/46/49/52`); a `NOT_EXECUTED` payload carries 0 of 18 data fields; the CLI exits 3. **Hole:** `provenance.py:73-76` omits `"not_executed"`, so the refusal is later stamped live/attested (§3.3). |
| "Empty outputs never become zeros" | **VIOLATED** | `live_discovery.py:658-662, 665, 724` turn an undefined recovery ratio into a literal `0.0` inside a live, **attested** faithfulness/completeness (`:769`). Contrast the correct behaviour at `aggregates.py:97-106` and `logit_lens.py:83-86`. |
| "Hard-coded metrics never become measurements" | **HOLDS** on every reachable measurement path | The remaining fixed fixtures are all labelled `reference`/`seeded`/`unavailable` with eligibility False and `measured: False`. Exceptions are not measurements or are unreachable: `/api/research_catalog` (unlabelled catalogue, carries no numbers), `automated_regression_suite` (no caller), `backend/ui/circuit_explorer.py` (no importer in 1155 files), `evidence_ranker` (latent). |
| "Evidence and confidence levels are never raised above what the run supports" | **VIOLATED** | `live_discovery.py:684-686` raises `evidence_level` to **`CAUSALLY_VALIDATED`** — defined at `provenance.py:654` as "passed held-out validation checks" — on a 10-prompt sample-size threshold alone, with no held-out validation run. Plus the `NOT_EXECUTED` live stamp. |
| "Results carry provenance (model, revision, prompt, config, seed) and stale results can never be presented as current" | **VIOLATED** (staleness clause); provenance clause **HOLDS** | Provenance is real: weight-derived `content_hash` with an honestly-labelled name fallback that verification rejects, `config_sha256`, real seeds, and a signed + re-verified record with input/output commitments. **Staleness fails:** legacy records that escape quarantine are listed and served as current completed runs with `provenance:"live"` and both eligibility flags true. |

---

## 7. Coverage and limits

**Verified by execution:** all eight claims; every number quoted above; **and the Electron
desktop app itself, launched and observed** (§2.9).

**Not verified:**
- The full pytest suite (see §6) — now covered by a per-file matrix instead.
- The **packaged** artefacts were not launched: the runtime probe used the working tree and the
  freshly rebuilt `frontend/dist` served over `http://127.0.0.1:5173` (Electron loads
  `file://` URLs whose absolute `/assets/...` paths cannot resolve). The `app.asar` inside
  `MECH Platform.exe` was inspected statically, not executed. Its `main.js` is the 2026-10-17
  copy, so the runtime observations are for the **current source**; the shipped binary is
  3 days older.
- HuggingFace's published **LFS** sha256 for `model.safetensors` — the `paths-info` endpoint
  answers POST with HTTP 307 and urllib does not re-POST on 307. The *revision* hash
  `607a30d7…` does match the live HF API, and the byte comparison was against the local cache
  copy.
- The paper's exact IOI logit difference (`logit(IO) − logit(S)`) — unreachable with the bundled
  datasets, so it is unexercised dead code.
- Handler-level scientific logic for all 82 endpoints (reachability, auth gating, input
  validation shape, error responses and traversal were verified; per-handler science was not).
- Screen-reader behaviour; all a11y findings are static analysis.

**Unreachable-but-latent** (dead code that would fail if revived): `backend/ui/circuit_explorer.py`,
`automated_regression_suite.evaluate_regression`, `evidence_ranker` defaults,
`gpt2_adapter`'s mock `PatchResult` (`original_logit=0.82, patched_logit=0.45, delta=-0.37`).

---

## 8. What the audit found about the audit

The brief asked us to check whether the 51-vs-1558 SAE figures implied the *larger* count was
correct. It is not — **51 is correct**, because `b_dec` is the data mean and subtracting it puts
the input back into the distribution the encoder was trained on; leaving it in pushes every
pre-activation across the ReLU threshold. The repository was right and the brief was wrong. The
live test pins the *invariant* rather than the numbers, which is why the correction went
unnoticed until now.

The same pattern repeats across the codebase: the failures that remain are not the ones anyone
wrote, they are the ones **left over after a genuine remediation pass** — a gate that lists
`"not_run"` but not `"not_executed"`, a cache key that omits the checkpoint, a loader that only
understands one checkpoint layout, an installer that was never rebuilt. The honesty engineering
is real and unusually good; what it has not yet caught is the gap between a fix and the places
that fix never reached.

Two things happened during the audit that are worth recording. First, one of our own findings —
the unseeded random prompt fallback in `patch_neuron` — was **fixed in the working tree while
we were reporting it**, by a concurrent session that read the same problem and cited it in the
commit comment. That is the system working as intended. Second, the same concurrency is what
invalidated one test run via the source-stability guard, which is the guard working as intended
too. Both are the same fact seen twice: this repository is being actively worked on, and an
audit of it is a snapshot of a moving target. The line numbers in this report are re-verified
against the tree as it stands; the findings are not necessarily against the tree you are looking
at when you read this.
