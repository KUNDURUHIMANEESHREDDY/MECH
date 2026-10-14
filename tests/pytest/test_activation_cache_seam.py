"""C7-C12: the activation cache, at its public seam.

`backend.runtime.activation_cache.ActivationCache` is a searchable store of
captured activations, keyed by session, prompt, layer, head and component.
It had **zero** test references in a 1824-test suite -- it was simply never
exercised.

Why it matters beyond tidiness: this cache is what a researcher reads back to
answer "what did the model do on *that* prompt, at *that* layer". A cache that
returns another session's activation, or one that quietly serves a tensor the
caller has since mutated, does not fail loudly -- it returns a plausible tensor
from the wrong place. That is the failure class this whole file exists to
catch, so every test here pins an identity (which entry, from which session)
rather than a shape.

Ordering follows the checklist: CRUD first (C7), then search (C8), then
eviction (C9), then the isolation properties (C10, C11), then the aliasing
hazard (C12).
"""

from __future__ import annotations

import threading

import pytest
import torch

from backend.runtime.activation_cache import (
    ActivationCache,
    CachedActivation,
)


def _entry(
    session_id: str = "s1",
    prompt_id: str = "p1",
    layer: int = 0,
    head: int | None = None,
    component: str = "residual",
    tensor: torch.Tensor | None = None,
) -> CachedActivation:
    """A cache entry with a distinguishable tensor.

    The default tensor encodes its own identity so a test that receives the
    wrong entry can say *which* one it got, rather than only that something is
    off.
    """
    if tensor is None:
        tensor = torch.full((2,), float(len(session_id) * 100 + layer))
    return CachedActivation(
        session_id=session_id,
        prompt_id=prompt_id,
        layer=layer,
        head=head,
        component=component,
        shape=tuple(tensor.shape),
        tensor=tensor,
    )


@pytest.fixture
def cache():
    return ActivationCache()


# ── C7: put / get / delete ────────────────────────────────────────────

def test_a_stored_activation_comes_back_identical(cache):
    """`put` then `get` must return the same tensor, not a copy of a copy.

    Pinned by value, not by shape: a copy that lost or transposed values would
    still satisfy a shape check, and the tensor *is* the measurement.
    """
    tensor = torch.tensor([1.5, -2.25, 3.0])
    entry = _entry(session_id="alpha", layer=7, tensor=tensor)

    returned_id = cache.put(entry)
    assert returned_id == entry.activation_id, (
        "put() returned an id the entry does not carry; the caller could not "
        "retrieve what it just stored"
    )

    fetched = cache.get(returned_id)
    assert fetched is not None, "get() could not find what put() just stored"
    torch.testing.assert_close(fetched.tensor, tensor)
    assert fetched.session_id == "alpha"
    assert fetched.layer == 7


def test_delete_removes_exactly_one_entry_and_reports_truthfully(cache):
    """`delete` must remove what was named and say whether it did.

    Returning True for an entry that is still there -- or False for one that
    was removed -- would both be lies, and callers use the return value to
    decide whether to re-capture.
    """
    keep = cache.put(_entry(session_id="keep"))
    drop = cache.put(_entry(session_id="drop"))

    assert cache.delete(drop) is True
    assert cache.get(drop) is None, "delete() reported success but the entry is still readable"
    assert cache.get(keep) is not None, "delete() removed an entry it was not asked to remove"

    # And a second delete of the same id must not claim to have worked again.
    assert cache.delete(drop) is False, (
        "delete() returned True for an id that was already absent"
    )


def test_get_returns_none_for_an_id_that_was_never_stored(cache):
    """A miss is `None`, not a fabricated empty entry.

    A cache that invented a default entry on miss would hand the caller a
    zero tensor wearing a real activation's identity.
    """
    assert cache.get("00000000-0000-0000-0000-000000000000") is None
    assert cache.delete("00000000-0000-0000-0000-000000000000") is False


# ── C8: search ────────────────────────────────────────────────────────

def test_search_filters_on_each_declared_key(cache):
    """Every documented search key must actually filter.

    Each key is checked on its own and then combined, because a filter that
    works alone but is dropped when combined with another is a real failure
    mode -- the AND chain is exactly where that tends to hide.
    """
    cache.put(_entry(session_id="s1", prompt_id="pa", layer=3, head=2, component="attention"))
    cache.put(_entry(session_id="s1", prompt_id="pb", layer=4, head=2, component="attention"))
    cache.put(_entry(session_id="s2", prompt_id="pa", layer=3, head=2, component="attention"))
    cache.put(_entry(session_id="s1", prompt_id="pa", layer=3, head=7, component="mlp"))

    assert len(cache.search(session_id="s1")) == 3
    assert len(cache.search(prompt_id="pa")) == 3
    assert len(cache.search(layer=3)) == 3
    assert len(cache.search(head=2)) == 3
    assert len(cache.search(component="attention")) == 3

    assert len(cache.search(session_id="s1", layer=3)) == 2
    assert len(cache.search(session_id="s1", prompt_id="pa", component="attention")) == 1

    # A filter that matches nothing must return nothing, not everything.
    assert cache.search(session_id="does-not-exist") == []
    assert cache.search(layer=999) == []
    assert cache.search(session_id="s1", component="mlp", layer=4) == []


def test_search_returns_the_entries_it_matched_not_just_the_count(cache):
    """Search results must be the requested entries, checked by identity.

    The obvious assertion -- `len(...) == 3` -- passes even when the search
    returns three of the *wrong* entries. So the returned set is compared by
    its identity fields, which is the only way to tell the difference.
    """
    cache.put(_entry(session_id="s1", layer=3, head=2, component="attention"))
    cache.put(_entry(session_id="s2", layer=3, head=2, component="attention"))
    cache.put(_entry(session_id="s1", layer=5, head=2, component="attention"))

    found = cache.search(session_id="s1")
    assert len(found) == 2, f"expected 2 entries, got {len(found)}"
    for entry in found:
        assert entry.session_id == "s1", (
            f"search(session_id='s1') returned an entry from {entry.session_id!r}"
        )
    assert sorted(e.layer for e in found) == [3, 5], (
        f"search returned layers {[e.layer for e in found]}, expected [3, 5]"
    )


# ── C9: eviction ──────────────────────────────────────────────────────

def test_eviction_drops_the_oldest_entry_first():
    """At capacity, the cache must evict in insertion order.

    The cache bounds itself at `max_entries`, so eviction has to happen. Which
    entry goes matters: a research session that captures a prompt, does other
    work, and comes back for the first prompt's activations should find them
    gone and say so -- not find a *different* session's activations under the
    same key.

    Insertion order is asserted directly by position, since "the right count
    of survivors" is also true of several wrong eviction policies.
    """
    cache = ActivationCache(max_entries=3)
    for i in range(5):
        cache.put(_entry(session_id=f"s{i}"))

    assert cache.count() == 3, (
        f"cache holds {cache.count()} entries, max_entries is 3"
    )
    survivors = [a.session_id for a in cache.search()]
    assert survivors == ["s2", "s3", "s4"], (
        f"FIFO eviction should leave ['s2', 's3', 's4']; got {survivors}"
    )


def test_the_cache_never_exceeds_its_declared_bound():
    """The bound is a hard ceiling, not an approximate target.

    Checked across the boundary rather than at one point above it, because an
    off-by-one here reads as a slowly growing memory profile instead of an
    obvious failure.
    """
    for max_entries in (1, 2, 5):
        cache = ActivationCache(max_entries=max_entries)
        for i in range(max_entries * 3):
            cache.put(_entry(session_id=f"s{i}"))
            assert cache.count() <= max_entries, (
                f"with max_entries={max_entries} the cache reached "
                f"{cache.count()} entries after the {i + 1}th put"
            )
        assert cache.count() == max_entries


def test_an_evicted_entry_reports_as_absent_not_as_another_entry():
    """After eviction, the evicted id must be a miss -- never another entry.

    This is the identity check that a count-based eviction test cannot make.
    If eviction ever deleted the wrong key, the count would still be right and
    the caller would be handed a live activation belonging to a different
    prompt, which is the specific corruption this cache must not cause.
    """
    cache = ActivationCache(max_entries=2)
    first = cache.put(_entry(session_id="first"))
    second = cache.put(_entry(session_id="second"))
    third = cache.put(_entry(session_id="third"))

    assert cache.get(first) is None, (
        "the oldest entry should have been evicted, but its id still resolves"
    )
    assert cache.get(third) is not None, "the newest entry was evicted"
    # Whatever survived must be the right entries, by their own identity.
    assert sorted(a.session_id for a in cache.search()) == ["second", "third"]
    assert cache.get(second).session_id == "second"


# ── C10: session isolation ────────────────────────────────────────────

def test_clearing_one_session_leaves_every_other_session_intact(cache):
    """`clear_session` must remove exactly the named session and nothing else.

    Session teardown is how MECH scopes a run's activations. Clearing session
    A must not disturb session B, because a researcher with two sessions open
    who tears one down still expects the other's activations to be readable.

    The return value is asserted too: it is the count the caller reports as
    "released N activations", so an off-by-one there misreports how much was
    actually removed.
    """
    for _ in range(2):
        cache.put(_entry(session_id="alpha"))
    cache.put(_entry(session_id="beta"))
    cache.put(_entry(session_id="gamma"))

    removed = cache.clear_session("alpha")

    assert removed == 2, (
        f"clear_session reported {removed} entries removed, expected 2"
    )
    assert cache.search(session_id="alpha") == [], (
        "session alpha still has entries after being cleared"
    )
    assert sorted(a.session_id for a in cache.search()) == ["beta", "gamma"], (
        "clearing one session disturbed the others"
    )


def test_clearing_a_session_twice_is_harmless_and_reports_zero(cache):
    """A second teardown must be a no-op that says so.

    Retries and double-teardown are ordinary. Returning a nonzero count the
    second time would overstate how many activations were released; raising
    would turn routine cleanup into a failure.
    """
    cache.put(_entry(session_id="alpha"))
    cache.put(_entry(session_id="beta"))

    assert cache.clear_session("alpha") == 1
    assert cache.clear_session("alpha") == 0, (
        "clearing an already-empty session claimed to have removed entries"
    )
    assert cache.count() == 1


def test_a_cached_activation_is_never_visible_to_another_session(cache):
    """The core isolation property: no cross-session reads.

    This is T04's cache half, and it is checked as an identity rather than a
    count. `search()` returning an empty list is the safe outcome; returning
    another session's entries is the dangerous one, and only comparing the
    returned sessions catches it.

    Two sessions hold *identical* layer/component/head metadata -- the setup
    that makes a missing session filter invisible to any check that only
    compares shapes or counts.
    """
    cache.put(_entry(session_id="alpha", layer=3, head=1, component="attention"))
    cache.put(_entry(session_id="beta", layer=3, head=1, component="attention"))

    found = cache.search(session_id="alpha")
    assert len(found) == 1, f"expected exactly 1 entry, got {len(found)}"
    assert found[0].session_id == "alpha", (
        f"search(session_id='alpha') returned an entry belonging to "
        f"{found[0].session_id!r}"
    )

    # And a session that was never written to must see nothing at all, rather
    # than everything.
    assert cache.search(session_id="gamma") == []


# ── C11: concurrency ──────────────────────────────────────────────────

def test_concurrent_writers_neither_lose_nor_corrupt_an_entry(cache):
    """Every entry a writer stored must come back with its own data intact.

    Activation capture is called from request handlers, so several sessions
    can be recording at once. The property that has to hold is not "the cache
    takes a lock" -- that would be asserting an implementation detail, and the
    GIL may satisfy it without any lock at all. The property is that nothing a
    writer stored is lost, and nothing comes back holding another entry's
    bytes.

    Each thread writes a tensor whose value encodes its own layer, so a
    returned entry carrying the wrong number is detected as corruption rather
    than passing as "some entry was found".

    Worker failures are collected, not asserted in-thread.

    This is the harness defect a review caught, and it is worth stating
    because the test was green while unable to fail: `worker` asserted, an
    uncaught `AssertionError` in a thread prints a traceback and ends the
    thread, `join()` returns normally, `is_alive()` is False -- and the `put`
    had already happened, so `count()` was still correct. A worker that read
    back another layer's tensor and died on that assertion left the suite
    green. The only reason it looked covered was the count check beneath it,
    which sees lost writes and cannot see corrupted reads.

    Fixed by collecting exceptions into a list guarded by the same barrier
    structure, asserting the collection is empty, and then re-verifying every
    stored id against its expected value after every worker has finished.
    """
    threads, per_thread = 8, 50
    start = threading.Barrier(threads)
    failures: list[str] = []
    expected: dict[str, float] = {}
    lock = threading.Lock()

    def worker(index: int) -> None:
        start.wait(timeout=30)          # every writer begins together
        try:
            for layer in range(per_thread):
                entry = _entry(
                    session_id=f"s{index}",
                    layer=layer,
                    tensor=torch.full((2,), float(layer)),
                )
                stored_id = cache.put(entry)
                with lock:
                    expected[stored_id] = float(layer)

                fetched = cache.get(stored_id)
                if fetched is None:
                    with lock:
                        failures.append(
                            f"thread {index}: entry stored immediately gone")
                    continue
                if float(fetched.tensor[0]) != float(layer):
                    with lock:
                        failures.append(
                            f"thread {index}: read back layer "
                            f"{float(fetched.tensor[0])} after storing "
                            f"{layer}")
        except BaseException as exc:            # noqa: BLE001 - recorded
            with lock:
                failures.append(f"thread {index}: {type(exc).__name__}: {exc}")

    workers = [
        threading.Thread(target=worker, args=(i,)) for i in range(threads)
    ]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=60)

    assert not any(t.is_alive() for t in workers), "a writer thread hung"

    # Every worker must have run to completion. This is the assertion whose
    # absence made the corruption case invisible.
    assert not failures, (
        f"{len(failures)} of {threads * per_thread} concurrent writes failed "
        f"or corrupted; first: {failures[:3]}"
    )

    assert cache.count() == threads * per_thread, (
        f"cache holds {cache.count()} entries after "
        f"{threads * per_thread} concurrent puts -- entries were lost"
    )

    # And re-verify independently, once everything has settled, so a corrupted
    # read is caught even if the worker's own check somehow passed.
    mismatched = []
    for stored_id, layer in expected.items():
        found = cache.get(stored_id)
        if found is None:
            mismatched.append(f"{stored_id} (layer {layer}) missing")
        elif float(found.tensor[0]) != layer:
            mismatched.append(
                f"{stored_id} holds layer {float(found.tensor[0])}, "
                f"expected {layer}")
    assert not mismatched, (
        f"{len(mismatched)} entries hold the wrong tensor after concurrent "
        f"writes: {mismatched[:3]}"
    )


def test_a_concurrent_read_never_sees_another_session_s_entry():
    """Reading session A while session B writes must never cross over.

    The identity check, under concurrency. Each thread pins one session and
    asserts that every entry it is handed belongs to that session -- the
    failure being guarded is a result that looks fine but belongs to somebody
    else's run, which no count-based assertion would catch.

    Uses `threading.Barrier` so the threads genuinely overlap rather than
    running one after another, which is the case that already works.
    """
    sessions = [f"s{i}" for i in range(6)]
    cache = ActivationCache()
    barrier = threading.Barrier(len(sessions))
    violations: list[tuple[str, str]] = []

    def worker(session_id: str) -> None:
        for layer in range(40):
            cache.put(_entry(session_id=session_id, layer=layer))
        barrier.wait(timeout=30)
        for _ in range(40):
            for found in cache.search(session_id=session_id):
                if found.session_id != session_id:
                    violations.append((session_id, found.session_id))

    workers = [
        threading.Thread(target=worker, args=(s,)) for s in sessions
    ]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=60)

    assert not any(t.is_alive() for t in workers), "a reader thread hung"
    assert not violations, (
        "concurrent reads returned entries from another session: %s"
        % violations[:5]
    )


# ── C12: the cached tensor must not alias the caller's ────────────────

def test_the_cache_stores_a_snapshot_not_a_live_view_of_the_caller_tensor(
    cache,
):
    """A later change by the caller must not rewrite what was cached.

    The hazard is aliasing. `put()` stored the caller's tensor object by
    reference, so any later in-place write by the caller silently changed the
    cached measurement -- the cache would then report a value that was never
    measured at capture time, under an activation_id and timestamp that claim
    otherwise.

    That is the worst failure mode for this subsystem: nothing raises, the
    provenance fields look right, and the number is simply not the one that
    was recorded. A cache is supposed to be a memory of a measurement, so it
    has to hold the measurement, not a reference to something that can change.

    `run_inference` is exactly the caller that triggers this: it captures an
    activation into a store, then the same buffer is reused or modified by
    subsequent work.
    """
    captured = torch.tensor([1.0, 2.0, 3.0])
    cache.put(_entry(session_id="s1", tensor=captured))

    # The caller does what real code does: reuses the buffer.
    captured[0] = 99.0

    stored = cache.search(session_id="s1")[0].tensor
    assert float(stored[0]) == 1.0, (
        "the cached tensor changed to %r when the caller's tensor was "
        "modified; the cache holds a live reference rather than a snapshot "
        "of what was measured" % float(stored[0])
    )


def test_the_cached_tensor_does_not_alias_another_cached_entry(cache):
    """Two entries must not share one tensor.

    Distinct activations are distinct measurements. If the cache handed the
    same object to two entries, then overwriting one -- which every caller
    eventually does, since buffers get reused -- would rewrite the other,
    producing two identical-looking entries that were never equal at capture
    time.

    Checked by identity (`is`) on the returned tensors, since equal-valued
    tensors are indistinguishable by value and that is precisely the point.
    """
    cache.put(_entry(session_id="s1", layer=0, tensor=torch.tensor([1.0, 2.0])))
    cache.put(_entry(session_id="s1", layer=1, tensor=torch.tensor([1.0, 2.0])))

    found = cache.search(session_id="s1")
    assert len(found) == 2
    assert found[0].tensor is not found[1].tensor, (
        "two distinct cache entries share one tensor object; writing to one "
        "would rewrite the other"
    )


# ── C8 (cont): distinguishing "no head" from "do not filter" ───────────

def test_headless_entries_are_reachable_and_distinct_from_no_filter(cache):
    """A caller must be able to ask for "the entries with no head".

    MLP and residual captures legitimately have `head=None`, and
    `run_inference` writes exactly those. But `search(head=None)` already
    meant "do not filter", so that population was unreachable -- the only way
    to get it was to filter everything else out by hand, and
    `search(head=None)` returned attention and MLP rows mixed together.

    Both readings are pinned here, because the fix must not quietly change the
    meaning of the existing call: `head=None` still means no-filter, and only
    the explicit sentinel means head-is-None.
    """
    from backend.runtime.activation_cache import HEADLESS

    cache.put(_entry(session_id="s1", component="mlp", head=None))
    cache.put(_entry(session_id="s1", component="attention", head=0))
    cache.put(_entry(session_id="s1", component="attention", head=3))

    headless = cache.search(session_id="s1", head=HEADLESS)
    assert len(headless) == 1, (
        f"expected the 1 headless entry, got {len(headless)}"
    )
    assert headless[0].head is None
    assert headless[0].component == "mlp", (
        "HEADLESS returned a headed entry: %r" % headless[0].component
    )

    # Unchanged meaning: None is still "no filter".
    assert len(cache.search(session_id="s1")) == 3, (
        "search(head=None) must keep meaning 'no filter' so existing callers "
        "are unaffected"
    )