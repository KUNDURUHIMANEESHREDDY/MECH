"""
GPT-2 Small - 8 Steps
=====================
Step 1  Install  (done externally - torch, transformers, transformer_lens)
Step 2  Load GPT-2 Small
Step 3  Print num_layers / num_heads / d_model
Step 4  Run one prompt -> verify Paris generation
Step 5  Capture residual stream, attention, MLP activations
Step 6  Visualise one attention head (saves PNG)
Step 7  Patch one head, measure logit difference
Step 8  One IOI experiment
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows so special characters don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Resolve HookedTransformer through MECH's compat shim so an unsupported
# transformer-lens release reports the supported range instead of a bare
# AttributeError. See backend/interpretability/tl_compat.py.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.interpretability.tl_compat import resolve_hooked_transformer

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HookedTransformer = resolve_hooked_transformer()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------
# STEP 2 - Load GPT-2 Small
# ---------------------------------------------------------------
print()
print("=" * 60)
print("STEP 2 - Loading GPT-2 Small")
print("=" * 60)

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
print("Model loaded OK.")

# ---------------------------------------------------------------
# STEP 3 - Architecture
# ---------------------------------------------------------------
print()
print("=" * 60)
print("STEP 3 - Architecture")
print("=" * 60)
print(f"  Layers          : {model.cfg.n_layers}")
print(f"  Attention heads : {model.cfg.n_heads}")
print(f"  Embedding dim   : {model.cfg.d_model}")

# ---------------------------------------------------------------
# STEP 4 - Run one prompt, measure Paris
#
# GPT-2 Small is not instruction-tuned, so "The capital of France
# is" does NOT reliably produce "Paris" as top-1. The standard
# interpretability approach is to measure the logit for the known
# correct answer (" Paris") and compare it against distractors.
# We also run greedy generation for 5 tokens to show the model
# does eventually produce Paris in multi-token continuations.
# ---------------------------------------------------------------
print()
print("=" * 60)
print("STEP 4 - Prompt: 'The capital of France is'")
print("=" * 60)

PROMPT = "The capital of France is"
PARIS_ID  = model.to_single_token(" Paris")
LONDON_ID = model.to_single_token(" London")
BERLIN_ID = model.to_single_token(" Berlin")
MADRID_ID = model.to_single_token(" Madrid")

tokens = model.to_tokens(PROMPT)

with torch.no_grad():
    logits = model(tokens)

last_logits = logits[0, -1, :]
top1_tok    = model.to_string([int(last_logits.argmax())]).strip()
paris_rank  = int((last_logits > last_logits[PARIS_ID]).sum()) + 1
paris_logit = float(last_logits[PARIS_ID])

print(f"  Top-1 greedy token : '{top1_tok}'")
print(f"  Paris logit        : {paris_logit:.3f}")
print(f"  Paris rank         : {paris_rank}")
print()
print("  Logit comparison (Paris vs distractors):")
for name, tid in [("Paris", PARIS_ID), ("London", LONDON_ID),
                  ("Berlin", BERLIN_ID), ("Madrid", MADRID_ID)]:
    print(f"    {name:<8} logit = {float(last_logits[tid]):.3f}")

# Multi-token greedy generation - GPT-2 produces Paris in context
print()
print("  5-token greedy continuation:")
with torch.no_grad():
    gen = model.generate(PROMPT, max_new_tokens=5, do_sample=False)
print(f"  '{gen}'")

# ---------------------------------------------------------------
# STEP 5 - Capture residual stream, attention, MLP activations
# ---------------------------------------------------------------
print()
print("=" * 60)
print("STEP 5 - Capturing activations")
print("=" * 60)

with torch.no_grad():
    logits, cache = model.run_with_cache(tokens)

resid = cache["resid_post", 0]   # [1, seq, d_model]
attn  = cache["pattern",   0]   # [1, heads, seq, seq]
mlp   = cache["mlp_post",  0]   # [1, seq, d_mlp]

print(f"  resid_post[layer 0] : {list(resid.shape)}")
print(f"  attn_pattern[layer 0]: {list(attn.shape)}")
print(f"  mlp_post[layer 0]   : {list(mlp.shape)}")
print("  All three captured OK.")

# ---------------------------------------------------------------
# STEP 6 - Visualise attention head L10 H7
# ---------------------------------------------------------------
print()
print("=" * 60)
print("STEP 6 - Attention head L10 H7")
print("=" * 60)

VIZ_LAYER, VIZ_HEAD = 10, 7
attn_mat   = cache["pattern", VIZ_LAYER][0, VIZ_HEAD].cpu().numpy()
str_tokens = model.to_str_tokens(PROMPT)
seq_len    = len(str_tokens)

# ASCII table
print("  Attention weights (row=query, col=key):")
header = "  " + " " * 10 + "  ".join(f"{t[:5]:>5}" for t in str_tokens)
print(header)
for qi, qt in enumerate(str_tokens):
    row = "  ".join(f"{attn_mat[qi, ki]:.3f}" for ki in range(seq_len))
    print(f"  {qt[:9]:<9}  {row}")

# PNG
fig, ax = plt.subplots(figsize=(seq_len * 0.9 + 1, seq_len * 0.9 + 1))
im = ax.imshow(attn_mat, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(range(seq_len))
ax.set_xticklabels(str_tokens, rotation=45, ha="right")
ax.set_yticks(range(seq_len))
ax.set_yticklabels(str_tokens)
ax.set_title(f"Attention L{VIZ_LAYER} H{VIZ_HEAD}")
plt.colorbar(im, ax=ax)
plt.tight_layout()
PNG_PATH = "attn_L10_H7.png"
plt.savefig(PNG_PATH, dpi=120)
plt.close()
print(f"  Saved: {PNG_PATH}")

# ---------------------------------------------------------------
# STEP 7 - Patch ONE head (L10 H7), measure logit difference
# ---------------------------------------------------------------
print()
print("=" * 60)
print("STEP 7 - Patching head L10 H7, measuring logit difference")
print("=" * 60)

def paris_london_ld(lgts: torch.Tensor) -> float:
    return float(lgts[0, -1, PARIS_ID] - lgts[0, -1, LONDON_ID])

with torch.no_grad():
    clean_logits = model(tokens)
clean_ld = paris_london_ld(clean_logits)
print(f"  Clean  logit diff (Paris - London): {clean_ld:+.4f}")

PATCH_LAYER, PATCH_HEAD = 10, 7

def zero_hook(z: torch.Tensor, hook) -> torch.Tensor:
    """Zero out the value-weighted output of head PATCH_HEAD."""
    z[:, :, PATCH_HEAD, :] = 0.0
    return z

with torch.no_grad():
    patched_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{PATCH_LAYER}.attn.hook_z", zero_hook)],
    )
patched_ld = paris_london_ld(patched_logits)
delta      = patched_ld - clean_ld
print(f"  Patched logit diff (Paris - London): {patched_ld:+.4f}")
print(f"  Delta (patch effect)               : {delta:+.4f}")
print(f"  Interpretation: zeroing head L{PATCH_LAYER}H{PATCH_HEAD} "
      f"{'hurts' if delta < 0 else 'helps'} the Paris prediction")

# ---------------------------------------------------------------
# STEP 8 - ONE IOI experiment
# ---------------------------------------------------------------
print()
print("=" * 60)
print("STEP 8 - IOI experiment")
print("=" * 60)

# Standard IOI template from Wang et al. 2022
# IO = Mary, Subject = John
# Model should predict " Mary" (the indirect object)

IOI_CLEAN = "When Mary and John went to the store, John gave the bag to"
IOI_CORR  = "When John and Mary went to the store, Mary gave the bag to"

MARY_ID = model.to_single_token(" Mary")
JOHN_ID = model.to_single_token(" John")

def ioi_ld(prompt: str) -> tuple:
    toks = model.to_tokens(prompt)
    with torch.no_grad():
        lgts = model(toks)
    last = lgts[0, -1, :]
    ld   = float(last[MARY_ID] - last[JOHN_ID])
    top1 = model.to_string([int(last.argmax())]).strip()
    return ld, top1

clean_ld_ioi, clean_top = ioi_ld(IOI_CLEAN)
corr_ld_ioi,  corr_top  = ioi_ld(IOI_CORR)

print(f"  Clean  prompt: '...{IOI_CLEAN[-30:]}'")
print(f"  Clean  top-1 : '{clean_top}'")
print(f"  Clean  logit diff (Mary - John): {clean_ld_ioi:+.4f}")
print()
print(f"  Corrupted prompt: '...{IOI_CORR[-30:]}'")
print(f"  Corrupted top-1 : '{corr_top}'")
print(f"  Corrupted logit diff (Mary - John): {corr_ld_ioi:+.4f}")
print()

ioi_pass  = clean_ld_ioi > 0
corr_pass = corr_ld_ioi  < 0
print(f"  Clean Mary > John  : {'PASS' if ioi_pass  else 'FAIL'}")
print(f"  Corrupted John > Mary: {'PASS' if corr_pass else 'FAIL'}")

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print()
print("=" * 60)
print("ALL 8 STEPS COMPLETE")
print("=" * 60)
print(f"  Step 2  Model loaded           OK")
print(f"  Step 3  Layers={model.cfg.n_layers}, Heads={model.cfg.n_heads}, d_model={model.cfg.d_model}")
print(f"  Step 4  Paris rank={paris_rank}, logit={paris_logit:.3f}")
print(f"  Step 5  resid/attn/mlp captured")
print(f"  Step 6  {PNG_PATH} saved")
print(f"  Step 7  Patch delta = {delta:+.4f}")
print(f"  Step 8  IOI clean={'PASS' if ioi_pass else 'FAIL'}, corrupted={'PASS' if corr_pass else 'FAIL'}")
