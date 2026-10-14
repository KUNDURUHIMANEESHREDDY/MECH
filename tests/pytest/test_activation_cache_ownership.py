"""Activation cache: ownership of returned data, and compound operations.

`test_activation_cache_seam.py` covers the CRUD/search/eviction surface
single-threaded, including the clone-on-put that stopped a caller rewriting a
cached measurement from outside. This file covers what the contract review
named as the next concerns -- gaps 1 through 5.

    1. Ownership: does mutating what get()/search() returns alter the cache?
    2. Compound concurrency: put / delete / clear_session / eviction
    3. Duplicate ids, malformed metadata, zero and negative capacities
    4. Retained memory bounded in bytes, not only in entries
    5. Cloning failure and unavailable devices

Gap 1 is the one with teeth. Clone-on-put is real and it is correctly placed
-- but it only closes the *producer's* side. `get()` and `search()` return the
stored `CachedActivation` by reference, so every other method is not the
"plain accessor" the comment at `put` claims it is. Measured on the current
code:

    after get()+mutate, cache value: [999.0, 5.0]

The cache reports a value that was never measured, under the activation_id and
timestamp that identified the original measurement. The same holds for
metadata: setting `entry.layer = 77` moves the entry in `search(layer=...)`,
which is a measurement index rather than a mutable field.
"""

from __future__ import annotations

import threading

import pytest
import torch

from backend.runtime.activation_cache import (
    ActivationCache,
    CachedActivation,
    HEADLESS,
)
from backend.runtime.errors import CacheError


def _entry(layer: int = 0, value: float = 1.0, **kwargs) -> CachedActivation:
    kwargs.setdefault("session_id", "s1")
    kwargs.setdefault("prompt_id", "p1")
    kwargs.setdefault("component", "attention")
    return CachedActivation(
        layer=layer,
        head=kwargs.pop("head", 0),
        shape=(2,),
        tensor=torch.tensor([value, value]),
        **kwargs,
    )


@pytest.fixture
def cache():
    return ActivationCache()


# ── Gap 1: ownership of returned data ─────────────────────────────────

def test_mutating_a_tensor_from_get_does_not_alter_the_cache(cache):
    """`get()` must not hand out the cached tensor itself."""
    aid = cache.put(_entry(value=5.0))

    retrieved = cache.get(aid)
    retrieved.tensor[0] = 999.0

    stored = cache.get(aid).tensor
    assert stored[0].item() == 5.0, (
        f"the cached value became {stored.tolist()}; a caller holding the "
        f"result of get() rewrote a measurement that was never made, under "
        f"the original activation_id and timestamp"
    )


def test_mutating_a_tensor_from_search_does_not_alter_the_cache(cache):
    """`search()` returns a list of the same live objects -- same exposure."""
    aid = cache.put(_entry(value=1.0))

    cache.search()[0].tensor[0] = -1.0

    assert cache.get(aid).tensor[0].item() == 1.0, (
        "search() handed out the cached tensor; filtering is read-only access"
    )


def test_mutating_metadata_from_get_does_not_move_the_entry_in_search(cache):
    """Metadata is a measurement index, not a mutable field.

    `layer`, `session_id`, `component`, `head` and `prompt_id` are what
    `search` filters on. A caller that edits one has silently changed which
    measurements a later query returns.
    """
    aid = cache.put(_entry(layer=3))

    cache.get(aid).layer = 77
    cache.get(aid).session_id = "HIJACKED"

    assert cache.search(layer=3), "the entry left its layer index"
    assert cache.search(layer=77) == [], (
        "a caller reached through get() and created a phantom layer index"
    )
    assert cache.search(session_id="HIJACKED") == [], (
        "a caller reached through get() and rewrote the session index"
    )


def test_two_callers_holding_the_same_entry_cannot_see_each_others_writes(cache):
    """Independent retrievals must not alias each other."""
    aid = cache.put(_entry(value=1.0))

    first = cache.get(aid)
    second = cache.get(aid)
    first.tensor[0] = 42.0

    assert second.tensor[0].item() == 1.0, (
        "two independent get() calls returned the same tensor; a caller "
        "could corrupt another caller's view through a shared reference"
    )


def test_the_producer_still_cannot_rewrite_the_cache_after_put(cache):
    """Clone-on-put must survive the ownership fix.

    This already worked. Pinned because the obvious way to fix gap 1 -- return
    copies from the accessors -- does not affect `put`, and the reverse mistake
    (dropping the clone now that accessors copy) would reintroduce the original
    defect silently.
    """
    original = _entry(value=7.0)
    aid = cache.put(original)

    original.tensor[0] = -7.0
    original.layer = 99

    assert cache.get(aid).tensor[0].item() == 7.0
    assert cache.get(aid).layer == 0


# ── Gap 3: malformed input and degenerate capacity ────────────────────

@pytest.mark.parametrize("capacity", [0, -1, -100])
def test_a_non_positive_capacity_does_not_crash(capacity):
    """Capacity 0 must mean "hold nothing", not raise StopIteration.

    `put` evicted by calling `next(iter(self._entries))` whenever
    `len(self._entries) >= self._max_entries`. With a capacity of 0 that is
    true on an empty dict, so the very first `put` raised StopIteration --
    a bare, contextless exception from an iterator rather than a refusal.

        max_entries=0: RAISED StopIteration
        max_entries=-1: RAISED StopIteration
    """
    cache = ActivationCache(max_entries=capacity)

    cache.put(_entry())          # must not raise

    assert cache.count() == 0, (
        f"a cache with capacity {capacity} retained {cache.count()} entries; "
        f"it should hold nothing rather than crashing or over-filling"
    )


def test_overwriting_an_existing_id_does_not_evict_a_sibling(cache):
    """A duplicate id is an update, not an insertion.

    The eviction ran before the insert and was unconditional, so re-putting an
    id already present consumed an eviction slot and discarded a *different*
    entry:

        filled: [0, 1, 2]
        after overwriting an existing id: [1, 2, 99]   <- layer 0 gone

    A caller updating a measurement in place would silently lose unrelated
    measurements to make room.
    """
    cache = ActivationCache(max_entries=3)
    for layer in range(3):
        cache.put(_entry(layer=layer))

    existing = cache.search()[0].activation_id
    cache.put(_entry(layer=99, activation_id=existing))

    layers = sorted(a.layer for a in cache.search())
    assert layers == [1, 2, 99], (
        f"after an in-place update the cache holds layers {layers}; layer 0 "
        f"was evicted to make room for an entry that was already present"
    )
    assert cache.count() == 3


@pytest.mark.parametrize("bad", [
    {"shape": None},
    {"shape": "not a tuple"},
    {"metadata": None},
    {"metadata": "not a dict"},
    {"timestamp": float("nan")},
    {"timestamp": float("inf")},
    {"tensor": None},
    {"activation_id": ""},
])
def test_malformed_metadata_is_stored_without_corrupting_the_cache(cache, bad):
    """Bad metadata is a caller error, but must not corrupt the index.

    The cache is a dict keyed by `activation_id`; the metadata is only read by
    `search`. A malformed field should either be stored as given and ignored by
    filtering, or rejected cleanly -- what it must not do is take the cache
    down or make a later query raise.
    """
    entry = _entry(layer=1)
    for key, value in bad.items():
        setattr(entry, key, value)

    cache.put(entry)

    # The cache still answers questions.
    assert isinstance(cache.search(), list)
    assert isinstance(cache.count(), int)
    cache.put(_entry(layer=2, session_id="s2"))
    assert cache.count() == 2
    assert len(cache.search(session_id="s2")) == 1


def test_a_very_large_tensor_is_stored_and_searchable(cache):
    """Size is not a correctness problem; the cache must not silently drop it."""
    big = torch.zeros(512, 512, dtype=torch.float32)   # 1 MiB

    entry = _entry()
    entry.tensor = big
    aid = cache.put(entry)

    assert cache.get(aid).tensor.shape == (512, 512)
    assert len(cache.search()) == 1


# ── Gap 4: retained memory bounded in bytes ───────────────────────────

def test_the_cache_can_bound_retained_bytes_not_only_entries():
    """Entry count is not a memory bound.

    One entry holding a 1 MiB tensor and one holding a byte are both "one
    entry". A caller who needs the cache to fit a memory budget has no way to
    express it, so the cache grows until the process runs out of memory -- and
    the failure appears somewhere else entirely.
    """
    cache = ActivationCache(max_entries=10_000, max_bytes=2 * 1024 * 1024)

    cache.put(_entry(layer=0))            # 2 floats, ~8 bytes
    for layer in range(1, 8):
        entry = _entry(layer=layer)
        entry.tensor = torch.zeros(512, 512, dtype=torch.float32)   # 1 MiB
        cache.put(entry)

    assert cache.bytes_used() <= cache.max_bytes, (
        f"the cache holds {cache.bytes_used()} bytes against a bound of "
        f"{cache.max_bytes}; entry-count bounds do not bound memory"
    )


def test_byte_budget_evicts_oldest_first_and_stays_consistent():
    """Eviction under a byte budget must respect insertion order."""
    # Each entry is 256 KiB and the budget is 768 KiB, so any three fit and a
    # fourth forces eviction. An earlier draft used 256 KiB entries against a
    # 4 KiB budget, where every single entry exceeded the budget on its own --
    # so the cache was right to refuse them and the test was measuring the
    # refusal, not the eviction it meant to check.
    cache = ActivationCache(max_entries=100, max_bytes=768 * 1024)

    for layer in range(6):
        entry = _entry(layer=layer)
        entry.tensor = torch.zeros(256, 256, dtype=torch.float32)   # 256 KiB
        cache.put(entry)

    assert cache.bytes_used() <= 768 * 1024, (
        f"{cache.bytes_used()} bytes retained against a 768 KiB budget")
    assert cache.count() == 3, (
        f"{cache.count()} entries retained; three 256 KiB entries fit, a "
        f"fourth must evict the oldest"
    )
    remaining = [a.layer for a in cache.search()]
    assert remaining == sorted(remaining), remaining
    assert remaining[-1] == 5, (
        f"surviving layers {remaining}; the newest entry was evicted first")


def test_an_entry_larger_than_the_whole_budget_is_refused(cache):
    """A cache whose defining property is a memory bound cannot break it.

    Storing it anyway -- after evicting everything else, which would empty the
    cache and still not satisfy the bound -- would make `max_bytes` a
    suggestion.
    """
    bounded = ActivationCache(max_entries=100, max_bytes=4096)
    bounded.put(_entry(layer=0))                      # small, stored

    oversized = _entry(layer=1)
    oversized.tensor = torch.zeros(1024, 1024)       # 4 MiB

    with pytest.raises(CacheError):
        bounded.put(oversized)

    assert len(bounded.search(layer=0)) == 1, (
        "the refused entry evicted the small one that already fit")


def test_counting_bytes_does_not_include_python_object_overhead():
    """The bound is on tensor payload, which is what a caller is budgeting.

    Reporting a total that included dict and dataclass overhead would make the
    number depend on CPython's allocator, so two runs of the same workload
    would report different totals and the figure could not be checked against
    a budget.
    """
    cache = ActivationCache()
    entry = _entry()
    entry.tensor = torch.zeros(10, dtype=torch.float32)

    cache.put(entry)

    assert cache.bytes_used() == entry.tensor.numel() * entry.tensor.element_size(), (
        f"bytes_used() is {cache.bytes_used()}, expected "
        f"{entry.tensor.numel() * entry.tensor.element_size()} from the tensor"
    )


# ── Gap 2: compound concurrency ───────────────────────────────────────

def test_the_entry_bound_holds_under_concurrent_puts():
    """The hard entry-count bound must hold with several writers.

    Measured on the current code: six threads -- three putting, two deleting,
    one clearing a session -- ended with 22 entries against a cap of 20. The
    bound is checked and enforced in two separate steps with nothing between
    them, so threads interleave between the check and the insert.
    """
    cache = ActivationCache(max_entries=25)
    barrier = threading.Barrier(6)
    errors: list[BaseException] = []

    def put_many() -> None:
        try:
            barrier.wait()
            for i in range(500):
                cache.put(_entry(layer=i))
        except BaseException as exc:            # noqa: BLE001 - recorded
            errors.append(exc)

    threads = [threading.Thread(target=put_many) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"concurrent put raised: {errors[:3]}"
    assert cache.count() <= 25, (
        f"the cache holds {cache.count()} entries against a cap of 25; the "
        f"bound is checked and enforced in two steps with nothing between them"
    )


def test_concurrent_put_delete_and_clear_leave_a_consistent_cache():
    """No entry may be duplicated, lost silently, or left half-written."""
    cache = ActivationCache(max_entries=50)
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def putter() -> None:
        try:
            barrier.wait()
            for i in range(400):
                cache.put(_entry(layer=i, session_id="a"))
                cache.put(_entry(layer=i, session_id="b"))
        except BaseException as exc:            # noqa: BLE001 - recorded
            errors.append(exc)

    def deleter() -> None:
        try:
            barrier.wait()
            for _ in range(400):
                for entry in cache.search(session_id="a")[:4]:
                    cache.delete(entry.activation_id)
        except BaseException as exc:            # noqa: BLE001 - recorded
            errors.append(exc)

    def clearer() -> None:
        try:
            barrier.wait()
            for _ in range(200):
                cache.clear_session("b")
        except BaseException as exc:            # noqa: BLE001 - recorded
            errors.append(exc)

    threads = [threading.Thread(target=fn)
               for fn in (putter, putter, deleter, clearer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"concurrent mutation raised: {errors[:3]}"
    assert cache.count() <= 50, (
        f"{cache.count()} entries against a cap of 50 after concurrent writes")
    # Every surviving entry must still be internally consistent.
    for entry in cache.search():
        assert entry.tensor is not None
        assert entry.tensor.shape == entry.shape, (
            "an entry survived with a tensor whose shape disagrees with the "
            "shape recorded at insert time")


def test_clear_session_during_eviction_does_not_resurrect_entries():
    """A cleared session must stay cleared."""
    cache = ActivationCache(max_entries=10)
    for layer in range(5):
        cache.put(_entry(layer=layer, session_id="doomed"))
    cache.put(_entry(layer=0, session_id="keeper"))

    removed = cache.clear_session("doomed")

    assert removed == 5
    assert cache.search(session_id="doomed") == []
    # And a later eviction must not bring them back.
    for layer in range(20):
        cache.put(_entry(layer=layer, session_id="fresh"))
    assert cache.search(session_id="doomed") == []


# ── Gap 5: cloning failure and devices ───────────────────────────────

def test_a_cloning_failure_leaves_the_cache_unchanged(cache):
    """A tensor that cannot be cloned must not leave a partial entry.

    `put` clones before inserting, so a failure should propagate with the
    cache untouched -- no entry, no eviction consumed, and nothing that later
    queries would trip over.
    """
    before = cache.count()
    entry = _entry()

    # A stand-in rather than a monkeypatch: torch.Tensor.clone is a C-level
    # method, so assigning `entry.tensor.clone = ...` silently does nothing and
    # the test's `pytest.raises` fails with DID NOT RAISE -- a test that looks
    # like it covers a failure path and covers nothing.
    class Unclonable:
        shape = (2,)

        def detach(self):
            return self

        def clone(self):
            raise RuntimeError("clone failed")

        def numel(self):
            return 2

        def element_size(self):
            return 4

    entry.tensor = Unclonable()      # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="clone failed"):
        cache.put(entry)

    assert cache.count() == before, (
        "a failed clone left an entry behind; a half-cached activation is "
        "worse than none, because it reads as a measurement that never "
        "completed"
    )
    assert cache.search() == []


def test_a_tensor_from_an_unavailable_device_is_refused_cleanly(cache):
    """A CUDA tensor on a CPU-only host must fail as a refusal, not a crash.

    The cache stores whatever it is given, so this is mostly a statement that
    the failure surfaces at `put` -- where the caller can see it -- rather
    than later during a search that happens to read the field.
    """
    entry = _entry()
    entry.tensor = torch.ones(2, device="meta")

    cache.put(entry)

    retrieved = cache.get(entry.activation_id)
    assert retrieved is not None
    assert retrieved.tensor.device.type == "meta", (
        "the cache altered the device of a tensor it was given; a meta tensor "
        "is a legitimate thing to stage and must not be silently moved"
    )


def test_the_cache_keeps_serving_after_a_refused_entry(cache):
    """One bad entry must not poison the cache for later callers."""
    cache.put(_entry(layer=1))

    bad = _entry(layer=2)
    bad.tensor = None                     # legitimate: a metadata-only entry
    cache.put(bad)

    good = _entry(layer=3)
    cache.put(good)

    assert cache.count() == 3
    assert len(cache.search(layer=3)) == 1
    # The tensor-less entry is still indexed by its metadata. `_entry` defaults
    # `head` to 0, so the HEADLESS query returns nothing here -- an earlier
    # draft asserted it returned 1, which would only have held if this entry
    # had no head at all. Query it the way it was actually written.
    headless = _entry(layer=4, head=None)
    headless.tensor = None
    cache.put(headless)
    assert len(cache.search(head=HEADLESS)) == 1, (
        "a metadata-only entry must still be reachable by head")