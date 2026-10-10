"""Shared model-execution lock for GPT-2 Small.

All forward passes through the single loaded model — whether from
backend.services.gpt2_engine (prompt cache, ablations, steering) or
backend.interpretability.discovery.live_measure (zero-ablation screens,
injection, path patching) — install global forward hooks on the same
module objects. Two forwards with different hooks interleaved on two
threads corrupt each other's captures: hooks leak across prompts and the
prompt cache ends up describing a prompt it never ran.

The rule: hold MODEL_LOCK for the entire hook-install → forward →
hook-remove → cache-read sequence. It is an RLock so nested acquisition
by helpers on the same thread is safe. Lock ordering is trivially
consistent because there is only one lock.
"""

from __future__ import annotations

import threading

MODEL_LOCK = threading.RLock()
