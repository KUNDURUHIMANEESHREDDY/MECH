"""Concurrent experiment state isolation (#17).

The engine keeps one prompt's activations in a global cache and installs
global forward hooks per measurement. Two threads measuring different
prompts must never read each other's activations: every response must
name and describe its own requested prompt.

Fails before the fix: split prime-then-read (`run_prompt` then read) has
a race window — thread A primes prompt A, thread B primes prompt B, A
reads B's matrix. The atomic `prompt=` readers under the shared
MODEL_LOCK close it.
"""

from __future__ import annotations

import threading

import pytest


from _weight_guard import skip_reason, weights_available


needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)


def test_measure_lock_is_the_shared_model_lock():
    """live_measure and gpt2_engine must serialize on ONE lock.

    Two different RLocks still interleave: engine forwards (cache hooks)
    and discovery forwards (ablation hooks) would corrupt each other.
    """
    from backend.interpretability.discovery import live_measure as lm
    from backend.services.model_lock import MODEL_LOCK

    assert lm._MEASURE_LOCK is MODEL_LOCK


@needs_weights
def test_concurrent_attention_reads_stay_on_their_own_prompt():
    from backend.services import gpt2_engine as eng

    prompts = [
        "The capital of France is",
        "The quick brown fox jumps over",
        "Scientists recently discovered that",
        "Once upon a time there was a",
    ]
    results = {}
    errors = {}

    def worker(prompt: str) -> None:
        try:
            # Hammer the same head from every thread to maximize overlap.
            for _ in range(3):
                res = eng.attention_head(0, 0, prompt=prompt)
                assert res.get("status") == "ok", res
                assert res.get("prompt") == prompt, (
                    f"requested {prompt!r}, got cache for {res.get('prompt')!r}"
                )
            results[prompt] = True
        except Exception as exc:  # noqa: BLE001 — collected, asserted below
            errors[prompt] = exc

    threads = [threading.Thread(target=worker, args=(p,)) for p in prompts]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert not errors, f"worker failures: {errors}"
    assert set(results) == set(prompts)


@needs_weights
def test_concurrent_run_prompt_responses_match_their_prompts():
    from backend.services import gpt2_engine as eng

    prompts = [
        "Machine learning models can",
        "The future of technology looks like",
    ]
    results: dict = {}
    errors: dict = {}

    def worker(prompt: str) -> None:
        try:
            res = eng.run_prompt(prompt)
            assert res.get("status") == "ok", res
            assert res.get("prompt") == prompt, (
                f"requested {prompt!r}, got {res.get('prompt')!r}"
            )
            results[prompt] = res.get("str_tokens")
        except Exception as exc:  # noqa: BLE001
            errors[prompt] = exc

    threads = [threading.Thread(target=worker, args=(p,)) for p in prompts]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert not errors, f"worker failures: {errors}"
    # Distinct prompts must produce distinct tokenizations.
    assert results[prompts[0]] != results[prompts[1]]
