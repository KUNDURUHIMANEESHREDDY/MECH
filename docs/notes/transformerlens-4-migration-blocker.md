# TransformerLens 4.x migration — deferred after security-compatible 3.9 pin

Status: **the legacy API is intentionally retained on TransformerLens 3.9.x while the dependency stack is upgraded to patched Transformers 5.x.** This avoids both problems seen in the previous pin: known advisories on Transformers 4.57.6 and removal of `HookedTransformer` in TransformerLens 4.0.

## Current compatibility contract

| Package | Minimum/upper bound | Reason |
|---|---|---|
| `transformers` | `>=5.20.2,<6.0` | Upstream security fixes reported by the strict audit; stays on the current 5.x API |
| `transformer-lens` | `>=3.9.0,<4.0` | 3.x retains `HookedTransformer`; 4.0 removed it |
| `huggingface-hub` | `>=1.31.0,<3.0` | Transformers 5.20.2 requires this Hub line |
| `setuptools` | `>=83.0.0` | Fixes PYSEC-2026-3447 detected in the hosted runner's preinstalled 79.0.1 |

The audited metadata for TransformerLens 3.9 declares `transformers>=5.9.0` and still exports
`HookedTransformer`. Transformers 5.20.2's dependency metadata requires
`huggingface-hub>=1.31.0,<3.0`. Thus the 3.9 + 5.20.2 combination avoids the
old TransformerLens 2.x / Transformers 5 incompatibility without prematurely
moving the repository's 33 legacy hook call sites onto the 4.x bridge surface.

This compatibility combination has been proposed in the security PR; it is **not
considered verified until the full hosted CI run succeeds**, including all backend
shards, live-weight evidence, dependency audit/SBOM/secrets scan, and E2E.

## Why the 4.x migration is still separate

TransformerLens 3.0 introduced `TransformerBridge` while retaining the legacy
`HookedTransformer` classes. TransformerLens 4.0 removed the `Hooked*` classes,
per the [upstream 4.0 release notes](https://transformerlensorg.github.io/TransformerLens/content/news/release-4.0.html).
A future migration must therefore be a measured refactor, not a version-only bump:

1. Replace `HookedTransformer.from_pretrained` with `TransformerBridge.boot_transformers`.
2. Re-map `run_with_cache` / `hook_z` conventions to the bridge API and confirm the tensor shapes consumed by attention and patching code.
3. Rewrite the `tl_compat.py` contract and update its negative control.
4. Re-run live-weight GPT-2 parity, attribution/patching tests, all backend shards and E2E.

Until those checks are independently passing, retaining the <4.0 cap is intentional.
