"""Execution status vocabulary.

`AGENTS.md` requires:

    A registered-but-unimplemented engine reports `NOT_EXECUTED`, distinct
    from `FAILED`, and carries no data.

None of the three clauses existed. `NOT_EXECUTED` appeared in no production
source file, and the vocabulary actually in use collapsed the distinction the
rule exists to protect:

    "status": "unavailable"   (70 sites)
    "status": "error"         (55 sites)

into which three different situations fell -- an engine that was never
written, an engine whose weights are not loaded, and an engine that ran and
failed. They are not the same kind of fact. The first is a permanent property
of the build; the third is a transient fault with a log, a retry and a fix.
Reporting them identically lets a permanent gap disappear into the same
channel as an outage.

This module introduces the value and the two properties the rule attaches to
it -- distinct from failure, and carrying no data. It does not relabel
existing transient `unavailable` payloads: an engine that is present and
runnable but has no weights today is *unavailable*, not *not executed*. That
distinction is the point, and collapsing it would re-create the bug in the
opposite direction.
"""

from __future__ import annotations

from typing import FrozenSet

# ── the status values ─────────────────────────────────────────────────

#: The engine is registered and reachable, ran to completion, and produced
#: data.
IMPLEMENTED = "IMPLEMENTED"

#: The engine is registered but not implemented. Nothing ran. This is a
#: property of the build, not of the run.
NOT_EXECUTED = "NOT_EXECUTED"

#: The engine exists and is implemented, but cannot run right now -- typically
#: no weights are loaded. Transient: retry later.
UNAVAILABLE = "UNAVAILABLE"

#: The engine ran and failed. Transient and diagnosable.
FAILED = "FAILED"

#: The engine is still running.
RUNNING = "RUNNING"

ALL_STATUSES: FrozenSet[str] = frozenset(
    {IMPLEMENTED, NOT_EXECUTED, UNAVAILABLE, FAILED, RUNNING}
)

# ── the payload rule ──────────────────────────────────────────────────

#: Fields whose presence on a NOT_EXECUTED payload would mean something was
#: measured. A NOT_EXECUTED response must contain none of them, because
#: nothing ran.
DATA_FIELDS: FrozenSet[str] = frozenset(
    {
        "top_token",
        "top_tokens",
        "logit",
        "logits",
        "logit_diff",
        "activation",
        "activations",
        "layer_effects",
        "max_causal_layer",
        "max_indirect_effect",
        "circuit_faithfulness",
        "patch_success_rate",
        "functional_recovery",
        "observed_metrics",
        "raw_traces",
        "isolated_logit_diff",
        "ablated_logit_diff",
        "full_logit_diff",
    }
)

#: The only keys a NOT_EXECUTED payload may carry. Everything here describes
#: the refusal; nothing here describes a measurement.
NOT_EXECUTED_ALLOWED_KEYS: FrozenSet[str] = frozenset(
    {
        "status",
        "reason",
        "engine",
        "measured",
        "validation_eligible",
        "publication_eligible",
        "rejected_heads",
        "requested_heads",
        "discovered_heads",
        "capability",
    }
)


# ── the registry of what is known not to be implemented ───────────────

#: Capabilities that are registered in `CapabilityRegistry` but have no
#: working engine behind them. Derived from an inventory check, not guessed:
#: every entry here is reconciled against `MODEL_CATALOGUE` by
#   `test_the_registry_inventory_matches_the_model_catalogue`.
#:
#: This is deliberately a small, verified list rather than the complement of
#: the catalogue. `sae`, `patching` and `validation` have no model spec but
#: are demonstrably real -- `sae_pipeline.py` fits a real encoder and measures
#: a real reconstruction error -- so "no model spec" does not imply "not
#: implemented". Inferring the list that way would mark working capabilities
#: as gaps.
UNIMPLEMENTED_CAPABILITIES: FrozenSet[str] = frozenset(
    {
        "deepseek",
        "mistral",
        "cloud",
        "cluster",
    }
)


def not_executed_payload(reason: str, *, engine: str | None = None,
                         capability: str | None = None) -> dict:
    """Build a NOT_EXECUTED response.

    Asserts that no data field leaked in. A refusal that carries a payload is
    a failure wearing a label, and the most likely way that happens is a
    caller assembling a response by copy-paste from a successful one.
    """
    payload = {
        "status": NOT_EXECUTED,
        "reason": reason,
        "measured": False,
        "validation_eligible": False,
        "publication_eligible": False,
    }
    if engine is not None:
        payload["engine"] = engine
    if capability is not None:
        payload["capability"] = capability

    leaked = sorted(set(payload) & DATA_FIELDS)
    if leaked:
        raise AssertionError(
            f"a NOT_EXECUTED payload may not carry data fields: {leaked}. "
            f"Nothing ran, so no measurement may be reported."
        )
    return payload


def is_terminal_failure(status: str) -> bool:
    """True for statuses that should be surfaced as a distinct failure."""
    return status in (NOT_EXECUTED, FAILED)