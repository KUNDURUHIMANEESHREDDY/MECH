"""Disjoint discovery and validation panels.

The held-out validation must use prompts that were NOT used in discovery.
This test enforces that the intersection of discovery and validation
prompt identifiers is empty.
"""
import os
import sys
import tempfile

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Generate a test signing key for attestation. The key FILE is created
# once at import, but the environment points at it per-test: a
# process-global `os.environ` assignment here would leak into every other
# test module and break whichever ran second.
_key = Ed25519PrivateKey.generate()
_seed = _key.private_bytes(
    serialization.Encoding.Raw,
    serialization.PrivateFormat.Raw,
    serialization.NoEncryption(),
)
_KEY_PATH = os.path.join(tempfile.gettempdir(), f"mech_test_key_{os.getpid()}.key")
with open(_KEY_PATH, "wb") as f:
    f.write(_seed)

from backend.core.provenance import reset_trusted_key_ids_cache  # noqa: E402


@pytest.fixture(autouse=True)
def _test_keys(monkeypatch):
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", _KEY_PATH)
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _KEY_PATH)
    reset_trusted_key_ids_cache()
    yield
    reset_trusted_key_ids_cache()

from backend.interpretability.discovery.live_discovery import (  # noqa: E402
    LiveIOIDiscovery,
    prompt_panel,
    HELD_OUT_PROMPTS,
    MIN_PROMPTS_FOR_VALIDATION,
    MIN_PROMPTS_FOR_INTERACTION,
    PAIR_POOL,
    _allocate_panel,
    _allocate_validation_panel,
)
from backend.validation.live_validation import (  # noqa: E402
    LiveValidationExecutor,
)


def test_prompt_panel_returns_pool_indices():
    """prompt_panel returns dicts with subject, io, and pool_index."""
    panel = prompt_panel(5, offset=0)
    assert len(panel) == 5
    for entry in panel:
        assert "subject" in entry
        assert "io" in entry
        assert "pool_index" in entry
        assert isinstance(entry["pool_index"], int)


def test_prompt_panel_offset_wraparound():
    """prompt_panel wraps around the PAIR_POOL correctly."""
    from backend.interpretability.discovery.live_discovery import PAIR_POOL
    pool_size = len(PAIR_POOL)
    panel = prompt_panel(5, offset=pool_size - 2)
    assert len(panel) == 5
    # Last two entries should be from the end of the pool
    assert panel[0]["pool_index"] == pool_size - 2
    assert panel[1]["pool_index"] == pool_size - 1
    # Then wraps around
    assert panel[2]["pool_index"] == 0
    assert panel[3]["pool_index"] == 1
    assert panel[4]["pool_index"] == 2


def test_discovery_and_validation_panels_are_disjoint():
    """Discovery panel (offset 0, n prompts) and validation panel (after discovery) must not overlap."""
    # Discovery uses 10 prompts
    n_prompts = 10
    discovery_panel = prompt_panel(n_prompts, offset=0)
    validation_panel = prompt_panel(HELD_OUT_PROMPTS, offset=n_prompts)

    discovery_indices = {p["pool_index"] for p in discovery_panel}
    validation_indices = {p["pool_index"] for p in validation_panel}

    # No overlap
    overlap = discovery_indices & validation_indices
    assert not overlap, f"Discovery and validation panels overlap at indices: {overlap}"


def test_discovery_and_validation_panels_disjoint_with_interaction():
    """Discovery, validation, and interaction panels must all be disjoint.

    Uses the real allocator — not bare offsets — so this pins the rule
    the discovery run itself follows.
    """
    n_prompts = MIN_PROMPTS_FOR_VALIDATION  # 10
    n_interaction = MIN_PROMPTS_FOR_INTERACTION  # 5

    discovery_panel = prompt_panel(n_prompts, offset=0)
    validation_panel = _allocate_validation_panel(discovery_panel)
    assert validation_panel is not None
    used = ({p["pool_index"] for p in discovery_panel}
            | {p["pool_index"] for p in validation_panel})
    interaction_panel = _allocate_panel(used, n_interaction)
    assert interaction_panel is not None

    discovery_indices = {p["pool_index"] for p in discovery_panel}
    validation_indices = {p["pool_index"] for p in validation_panel}
    interaction_indices = {p["pool_index"] for p in interaction_panel}

    # All pairwise disjoint
    assert not (discovery_indices & validation_indices), "Discovery and validation overlap"
    assert not (discovery_indices & interaction_indices), "Discovery and interaction overlap"
    assert not (validation_indices & interaction_indices), "Validation and interaction overlap"


def test_validation_rejects_overlapping_panel():
    """LiveValidationExecutor rejects a discovery with overlapping validation panel."""
    from backend.interpretability.discovery.live_discovery import prompt_panel

    # Create a discovery with a validation panel that overlaps discovery
    discovery = {
        "discovery_id": "test_discovery",
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "heads": ["L5H1", "L5H5", "L7H3", "L8H6", "L9H9"],
        "head_effects": [
            {"head": "L5H1", "mean_delta": 0.5},
            {"head": "L5H5", "mean_delta": 0.4},
            {"head": "L7H3", "mean_delta": 0.3},
            {"head": "L8H6", "mean_delta": 0.2},
            {"head": "L9H9", "mean_delta": 0.1},
        ],
        "n_prompts": 10,
        "discovery_panel": prompt_panel(10, offset=0),
        # Malicious: validation panel overlaps with discovery (offset 4 instead of 10)
        "validation_panel": prompt_panel(3, offset=4),
    }

    executor = LiveValidationExecutor()
    result = executor.validate(discovery)

    # Should be rejected with unavailable status
    assert result.get("status") == "unavailable"
    assert "overlap" in result.get("reason", "").lower()


def test_validation_accepts_disjoint_panel():
    """LiveValidationExecutor accepts a discovery with properly disjoint validation panel."""
    # This test requires live weights, so skip if unavailable
    if not LiveIOIDiscovery.available():
        pytest.skip("live GPT-2 weights unavailable")

    # Run actual discovery
    discovery = LiveIOIDiscovery().run(
        hypothesis_statement="Identify the minimal set of heads sufficient for IOI",
        n_prompts=MIN_PROMPTS_FOR_VALIDATION,
        n_interaction_prompts=MIN_PROMPTS_FOR_INTERACTION,
    )

    assert discovery.get("status") == "completed"
    assert "discovery_panel" in discovery
    assert "validation_panel" in discovery

    # Verify disjointness in the discovery result itself
    discovery_indices = {p["pool_index"] for p in discovery["discovery_panel"]}
    validation_indices = {p["pool_index"] for p in discovery["validation_panel"]}
    assert not (discovery_indices & validation_indices)

    # Now validate
    executor = LiveValidationExecutor()
    result = executor.validate(discovery)

    assert result.get("status") == "completed", result.get("reason")
    assert "validation_panel" in result


def test_validation_stores_validation_panel():
    """Validation result includes the validation_panel for audit trail."""
    if not LiveIOIDiscovery.available():
        pytest.skip("live GPT-2 weights unavailable")

    discovery = LiveIOIDiscovery().run(
        hypothesis_statement="Identify the minimal set of heads sufficient for IOI",
        n_prompts=MIN_PROMPTS_FOR_VALIDATION,
        n_interaction_prompts=MIN_PROMPTS_FOR_INTERACTION,
    )

    executor = LiveValidationExecutor()
    result = executor.validate(discovery)

    assert result.get("status") == "completed"
    assert "validation_panel" in result
    assert len(result["validation_panel"]) == HELD_OUT_PROMPTS
    for entry in result["validation_panel"]:
        assert "subject" in entry
        assert "io" in entry
        assert "pool_index" in entry


def test_negative_overlapping_panel_rejected():
    """A discovery with an overlapping validation_panel is rejected."""
    from backend.interpretability.discovery.live_discovery import prompt_panel

    # Discovery with 10 prompts, but validation_panel overlaps at indices 4,5,6
    discovery = {
        "discovery_id": "test_overlap",
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "heads": ["L5H1"],
        "head_effects": [{"head": "L5H1", "mean_delta": 0.5}],
        "n_prompts": 10,
        "discovery_panel": prompt_panel(10, offset=0),
        # Overlapping validation panel - starts at offset 4 instead of 10
        "validation_panel": prompt_panel(3, offset=4),
    }

    executor = LiveValidationExecutor()
    result = executor.validate(discovery)

    assert result.get("status") == "unavailable"
    assert "overlap" in result.get("reason", "").lower()
    assert "held-out requirement" in result.get("reason", "").lower()


def test_held_out_prompts_constant():
    """HELD_OUT_PROMPTS constant is used consistently."""
    assert HELD_OUT_PROMPTS == 3
    from backend.validation.live_validation import LiveValidationExecutor
    # The executor should use the same constant
    assert hasattr(LiveValidationExecutor, "method")


def test_legacy_record_without_panels_is_rejected():
    """A discovery without recorded panels cannot prove held-out status.

    The old fallback guessed a validation offset from `n_prompts`
    (defaulting to 10). A record that used a different count — or no
    panels at all — would then validate on overlapping prompts while
    looking held-out. Fail closed instead.
    """
    discovery = {
        "discovery_id": "test_legacy",
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "heads": ["L5H1"],
        "head_effects": [{"head": "L5H1", "mean_delta": 0.5}],
        "n_prompts": 10,
        # No discovery_panel / validation_panel keys at all.
    }

    result = LiveValidationExecutor().validate(discovery)

    assert result.get("status") == "unavailable"
    assert "panel" in result.get("reason", "").lower()


def test_record_missing_only_discovery_panel_is_rejected():
    """A validation panel without a discovery panel proves nothing: there
    is no composition to check it against."""
    discovery = {
        "discovery_id": "test_no_discovery_panel",
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "heads": ["L5H1"],
        "head_effects": [{"head": "L5H1", "mean_delta": 0.5}],
        "n_prompts": 10,
        "validation_panel": prompt_panel(3, offset=10),
    }

    result = LiveValidationExecutor().validate(discovery)

    assert result.get("status") == "unavailable"
    assert "panel" in result.get("reason", "").lower()


def test_panel_entries_without_pool_index_are_rejected():
    """Pool indices are the disjointness evidence. Entries without one
    cannot be checked and must not pass by default."""
    discovery = {
        "discovery_id": "test_no_indices",
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "heads": ["L5H1"],
        "head_effects": [{"head": "L5H1", "mean_delta": 0.5}],
        "n_prompts": 2,
        "discovery_panel": [
            {"subject": "Alice", "io": "Bob"},
            {"subject": "Alice", "io": "Charlie"},
        ],
        "validation_panel": [
            {"subject": "David", "io": "Eve"},
        ],
    }

    result = LiveValidationExecutor().validate(discovery)

    assert result.get("status") == "unavailable"
    assert "pool_index" in result.get("reason", "").lower()


def test_wraparound_validation_panel_is_rejected():
    """The pool is cyclic: at offset 28 a 3-prompt panel wraps to index 0,
    which discovery (28 prompts from 0) already used. Proximity in offset
    is not disjointness; only the index sets decide."""
    discovery = {
        "discovery_id": "test_wraparound",
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "heads": ["L5H1"],
        "head_effects": [{"head": "L5H1", "mean_delta": 0.5}],
        "n_prompts": 28,
        "discovery_panel": prompt_panel(28, offset=0),
        "validation_panel": prompt_panel(3, offset=28),
    }
    # Sanity: this fixture really does wrap into the discovery range.
    assert {p["pool_index"] for p in discovery["validation_panel"]} == {28, 29, 0}

    result = LiveValidationExecutor().validate(discovery)

    assert result.get("status") == "unavailable"
    assert "overlap" in result.get("reason", "").lower()


@pytest.mark.parametrize("n_prompts", [1, 2, 4, 5, 9, 10, 11, 15, 26, 27])
def test_all_feasible_prompt_counts_stay_disjoint(n_prompts):
    """Pin the allocation rule across every discovery size that leaves
    room for a held-out panel, so a future change to the default counts
    fails here instead of silently reintroducing overlap."""
    from backend.interpretability.discovery.live_discovery import (
        PAIR_POOL,
        prompt_panel,
    )

    assert n_prompts + HELD_OUT_PROMPTS <= len(PAIR_POOL)
    discovery_panel = prompt_panel(n_prompts, offset=0)
    validation_panel = prompt_panel(HELD_OUT_PROMPTS, offset=n_prompts)

    overlap = ({p["pool_index"] for p in discovery_panel}
               & {p["pool_index"] for p in validation_panel})
    assert not overlap, f"n_prompts={n_prompts} overlaps at {overlap}"


@pytest.mark.parametrize("n_prompts", [1, 2, 4, 5, 9, 10, 11, 15, 26, 27])
def test_all_feasible_prompt_counts_stay_disjoint(n_prompts):
    """Pin the allocation rule across every discovery size that leaves
    room for a held-out panel, so a future change to the default counts
    fails here instead of silently reintroducing overlap."""
    assert n_prompts + HELD_OUT_PROMPTS <= len(PAIR_POOL)
    discovery_panel = prompt_panel(n_prompts, offset=0)
    validation_panel = _allocate_validation_panel(discovery_panel)

    assert validation_panel is not None
    assert len(validation_panel) == HELD_OUT_PROMPTS
    overlap = ({p["pool_index"] for p in discovery_panel}
               & {p["pool_index"] for p in validation_panel})
    assert not overlap, f"n_prompts={n_prompts} overlaps at {overlap}"


def test_oversized_discovery_emits_no_validation_panel():
    """At 28 discovery prompts only indices 28-29 are free — no disjoint
    3-prompt panel fits. Discovery must emit no panel (with a reason),
    not a self-overlapping one."""
    discovery_panel = prompt_panel(28, offset=0)
    assert _allocate_validation_panel(discovery_panel) is None

    fitting = _allocate_validation_panel(prompt_panel(10, offset=0))
    assert fitting is not None
    assert {p["pool_index"] for p in fitting} == {10, 11, 12}


def test_allocate_panel_unit_cases():
    """Pure allocation-table tests: fragmented, wrapped, and exhausted pools."""
    # Empty pool usage allocates from the start.
    first = _allocate_panel(set(), 3)
    assert [p["pool_index"] for p in first] == [0, 1, 2]
    # Fragmented usage still yields the first consecutive free run.
    frag = _allocate_panel({0, 2, 3, 4}, 3)
    assert [p["pool_index"] for p in frag] == [5, 6, 7]
    # Wraparound: only the tail is free, but the run does not fit
    # without wrapping into used indices.
    assert _allocate_panel(set(range(28)), 3) is None
    # Single free slot is not a 3-prompt panel.
    assert _allocate_panel(set(range(29)), 3) is None
    assert _allocate_panel(set(range(29)), 1) is not None
    # Fully covered pool allocates nothing.
    assert _allocate_panel(set(range(len(PAIR_POOL))), 3) is None
    # Non-positive requests allocate nothing.
    assert _allocate_panel(set(), 0) is None
    # Entries carry subject/io bound to their index.
    panel = _allocate_panel({0, 1}, 2)
    assert panel is not None
    for entry in panel:
        subj, io = PAIR_POOL[entry["pool_index"]]
        assert entry["subject"] == subj
        assert entry["io"] == io


def test_interaction_overlap_is_rejected():
    """The tuning (interaction) panel must be disjoint too: edges
    estimated on discovery prompts would carry the same selection bias
    the held-out requirement exists to prevent."""
    discovery = {
        "discovery_id": "test_interaction_overlap",
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "heads": ["L5H1"],
        "head_effects": [{"head": "L5H1", "mean_delta": 0.5}],
        "n_prompts": 10,
        "discovery_panel": prompt_panel(10, offset=0),
        "validation_panel": prompt_panel(3, offset=10),
        # Interaction reuses discovery indices 0-1: overlaps discovery.
        "interaction_panel": prompt_panel(2, offset=0),
    }

    result = LiveValidationExecutor().validate(discovery)

    assert result.get("status") == "unavailable"
    assert "overlap" in result.get("reason", "").lower()


def test_relabeled_panel_entry_is_rejected():
    """Disjoint indices with swapped-in names still remeasure discovery
    prompts. Each entry's names must match its claimed pool index."""
    from backend.interpretability.discovery.live_discovery import PAIR_POOL

    overlapping_names = PAIR_POOL[0]  # a discovery pair...
    relabeled = [
        {"subject": overlapping_names[0], "io": overlapping_names[1],
         "pool_index": 10},  # ...wearing a disjoint index
        {"subject": PAIR_POOL[11][0], "io": PAIR_POOL[11][1],
         "pool_index": 11},
        {"subject": PAIR_POOL[12][0], "io": PAIR_POOL[12][1],
         "pool_index": 12},
    ]
    discovery = {
        "discovery_id": "test_relabeled",
        "status": "completed",
        "provenance": "live",
        "attested": True,
        "heads": ["L5H1"],
        "head_effects": [{"head": "L5H1", "mean_delta": 0.5}],
        "n_prompts": 10,
        "discovery_panel": prompt_panel(10, offset=0),
        "validation_panel": relabeled,
    }

    result = LiveValidationExecutor().validate(discovery)

    assert result.get("status") == "unavailable"
    assert "pool index" in result.get("reason", "").lower()


def test_pair_for_index_binds_content():
    """The index-to-content map is total over the pool and rejects
    everything else."""
    from backend.interpretability.discovery.live_discovery import (
        pair_for_index,
    )

    for idx, (subj, io) in enumerate(PAIR_POOL):
        assert pair_for_index(idx) == (subj, io)
    assert pair_for_index(-1) is None
    assert pair_for_index(len(PAIR_POOL)) is None
    assert pair_for_index("10") is None
    assert pair_for_index(True) is None
    assert pair_for_index(None) is None


def test_live_run_emits_three_disjoint_panels():
    """End to end: a live discovery run emits discovery, validation, and
    interaction panels that are pairwise disjoint."""
    if not LiveIOIDiscovery.available():
        pytest.skip("live GPT-2 weights unavailable")

    discovery = LiveIOIDiscovery().run(
        hypothesis_statement="Identify the minimal set of heads sufficient for IOI",
        n_prompts=MIN_PROMPTS_FOR_VALIDATION,
        n_interaction_prompts=MIN_PROMPTS_FOR_INTERACTION,
    )

    assert discovery.get("status") == "completed"
    panels = {
        name: discovery.get(name)
        for name in ("discovery_panel", "validation_panel",
                     "interaction_panel")
    }
    for name, panel in panels.items():
        assert isinstance(panel, list) and panel, f"{name} missing"
    index_sets = [{p["pool_index"] for p in panel}
                  for panel in panels.values()]
    for i in range(len(index_sets)):
        for j in range(i + 1, len(index_sets)):
            assert not (index_sets[i] & index_sets[j]), (
                f"panels overlap: {index_sets[i] & index_sets[j]}")

    result = LiveValidationExecutor().validate(discovery)
    assert result.get("status") == "completed", result.get("reason")


def test_oversized_live_run_has_no_validation_panel():
    """End to end at 28 prompts: discovery completes (its own measurements
    are still real) but emits no validation panel and is not eligible,
    and validation fails closed."""
    if not LiveIOIDiscovery.available():
        pytest.skip("live GPT-2 weights unavailable")

    discovery = LiveIOIDiscovery().run(
        hypothesis_statement="Identify the minimal set of heads sufficient for IOI",
        n_prompts=28,
        n_interaction_prompts=2,
    )

    assert discovery.get("status") == "completed"
    assert discovery.get("validation_panel") is None
    assert discovery.get("validation_eligible") is False
    assert discovery.get("publication_eligible") is False

    result = LiveValidationExecutor().validate(discovery)
    assert result.get("status") == "unavailable"
    assert "panel" in result.get("reason", "").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
