"""One recursive provenance validator, for nested result data.

The suite checks top-level provenance in three places -- `causal_tracing`,
the discoverer's spread order, and the execution-status surfaces -- and each
does it for itself. That leaves the next risk open: a value nested inside a
list or a dict inheriting trust it never earned.

Measured, on the causal-tracing refusal response:

    {"layer": 3, "indirect_effect": 0.12, "top_causal_component": "Attention"}

`indirect_effect` is a number a caller can quote and `top_causal_component`
is a label that reads exactly like a circuit finding. The response marks the
*list* as "reference" by index::

    "layer_effects_field_provenance": {"0": "reference", "1": "reference", ...}

which says "row 3 came from the reference curve" and says nothing about which
field inside row 3 is synthetic. An extracted `entry["indirect_effect"]` is a
bare float with nothing attached.

The rule this module encodes
-----------------------------
    If a derived value depends on an unmeasured input, it must not become a
    live scientific claim merely because the outer response says completed.

Trust flows **downward only**. A response may be marked "live" while
containing "reference" leaves, and that is fine -- it means some fields were
measured and others were not, which is the truth. What is forbidden is the
opposite: a "reference" or "unavailable" leaf reaching a caller unmarked, or
an unmarked leaf anywhere in a response whose provenance was never stated.

Design decisions
----------------
* Recursive by construction. A new nesting shape cannot escape the check,
  which is the whole point of having one validator rather than three.
* Path-addressed errors. A failure names the exact JSON path, so "this field
  is unmarked" is actionable rather than a message about the response.
* Strict by default. ``require_all=True`` means an unmarked numeric leaf is
  a failure, not a warning.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Provenance values that assert a measurement happened.
LIVE = "live"
#: Values that mean "not measured": synthesised, refused, unavailable.
NON_LIVE = ("reference", "synthetic", "unavailable", "not_executed", "error")

#: Keys that are structural rather than scientific. These are never required
#: to carry provenance: `status` is the marker itself, and a path is a
#: locator. Everything else in a result payload is a value someone could read.
STRUCTURAL_KEYS = frozenset({
    "status", "reason", "method", "prompt", "layer", "layer_index", "layerIdx",
    "token", "token_id", "ids", "id", "index", "component", "head", "head_index",
    "timestamp", "created_at", "schema_version", "field_provenance",
    "type", "version", "name", "label", "path", "category",
})

#: Suffix for a sibling key that holds provenance for the key it qualifies.
PROVENANCE_SUFFIX = "_field_provenance"
#: The canonical name, kept for callers that need to add a marker.
PROVENANCE_MAP_KEY = "field_provenance"


def _is_provenance_key(key: str, suffix: str = PROVENANCE_SUFFIX) -> bool:
    """Whether `key` names a provenance map rather than a value.

    The canonical name has no prefix, so it does not match the suffix rule:
    `"field_provenance".endswith("_field_provenance")` is False. Both forms
    are accepted, and this predicate is the single place that decides it --
    getting this wrong silently disables every marker in a payload, which is
    the failure this module exists to prevent.
    """
    return key == PROVENANCE_MAP_KEY or key.endswith(suffix)


def _provenance_maps(node: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """Every sibling provenance map in `node`, merged.

    A census of the codebase at the time of writing:

        130x  "field_provenance"
          1x  "headline_field_provenance"
          1x  "layer_effects_field_provenance"
          1x  "alignment_field_provenance"
          1x  "genealogy_field_provenance"
          1x  "top_attributed_nodes_field_provenance"

    Five bespoke names against one canonical one. A validator hardcoded to
    `field_provenance` would report 130 producers clean and silently ignore
    five -- which is the same shape as the defect it exists to catch, one
    level up. So the rule is by *suffix*: any sibling key ending in
    `_field_provenance` is a field -> provenance map for this level.

    A map keyed by list index rather than field name simply does not match
    any sibling field name, so it stays inert. That is correct and is what
    `test_index_based_marking_does_not_mark_the_fields_inside` pins.
    """
    merged: Dict[str, Any] = {}
    for key, value in node.items():
        if _is_provenance_key(key, suffix) and isinstance(value, dict):
            merged.update(value)
    return merged


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_live(provenance: Any) -> bool:
    """Whether a provenance value asserts a measurement."""
    return isinstance(provenance, str) and provenance.strip().lower() == LIVE


def collect_unmarked_numbers(
    payload: Any,
    *,
    require_all: bool = True,
    structural_keys: Iterable[str] = STRUCTURAL_KEYS,
    provenance_suffix: str = PROVENANCE_SUFFIX,
    provenance_map_key: str = PROVENANCE_MAP_KEY,
) -> List[Tuple[str, Any]]:
    """Every numeric leaf in `payload` that no provenance marker covers.

    Returns ``(json_path, value)`` pairs. An empty list means every numeric
    leaf is either marked or is structural.

    Recognised as marking a value, in order of precedence:

    * any sibling map whose key ends in ``_field_provenance``, canonical
      ``field_provenance`` included -- see ``_provenance_maps`` for why the
      rule is by suffix rather than by one hardcoded name;
    * a sibling key ``<name>_field_provenance`` holding a bare string, the
      single-field form;
    * a ``field_provenance`` map carried *inside* the same dict or list entry
      -- the per-entry form, which is what nested results need.

    The last is the one that closes the nested gap: marking a list by index
    does not mark the fields within its entries.
    """
    structural = set(structural_keys)
    found: List[Tuple[str, Any]] = []

    def walk(node: Any, path: str, inherited_map: Dict[str, Any]) -> None:
        local_map = inherited_map
        if isinstance(node, dict):
            local_map = {**inherited_map, **_provenance_maps(node, provenance_suffix)}

        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict) and _is_provenance_key(
                        key, provenance_suffix):
                    continue
                child_path = f"{path}.{key}" if path else str(key)

                sibling = node.get(f"{key}{provenance_suffix}")
                marked = key in local_map or (
                    isinstance(sibling, str) and bool(sibling.strip()))

                if isinstance(value, dict):
                    walk(value, child_path, local_map)
                elif isinstance(value, (list, tuple)):
                    for i, item in enumerate(value):
                        item_path = f"{child_path}[{i}]"
                        if isinstance(item, (dict, list, tuple)):
                            walk(item, item_path, local_map)
                        elif _is_number(item) and not marked \
                                and key not in structural:
                            found.append((item_path, item))
                elif _is_number(value):
                    if marked or key in structural or not require_all:
                        continue
                    found.append((child_path, value))

    walk(payload, "", {})
    return found


def unmarked_leaves(payload: Any, *, require_all: bool = True) -> List[str]:
    """Just the paths, for assertion messages."""
    return [p for p, _ in collect_unmarked_numbers(
        payload, require_all=require_all)]


def assert_all_numbers_marked(payload: Any, *, require_all: bool = True,
                              what: str = "response") -> None:
    """Raise ``AssertionError`` naming every unmarked numeric leaf."""
    unmarked = collect_unmarked_numbers(payload, require_all=require_all)
    if unmarked:
        paths = ", ".join(p for p, _ in unmarked[:12])
        more = "" if len(unmarked) <= 12 else f" (+{len(unmarked) - 12} more)"
        raise AssertionError(
            f"{len(unmarked)} numeric value(s) in this {what} carry no "
            f"provenance: {paths}{more}. A number a caller can quote must say "
            f"whether it was measured. Mark it with a sibling "
            f"`field_provenance` entry, a `<key>_field_provenance` sibling, "
            f"or a `field_provenance` map inside the entry that contains it."
        )


def is_derived_from_unmeasured(payload: Any) -> List[str]:
    """Paths where a non-live marker sits next to a value that reads as live.

    Catches the specific shape the review describes: a derived score whose
    inputs were refused, but which sits in a response whose outer status says
    `completed`. The outer status is not consulted -- this walks markers and
    values together and reports where a value marked non-live is being used
    as though it were measured.

    Kept separate from `collect_unmarked_numbers` because it answers a
    different question: not "is this unmarked" but "is this marked *and the
    marker says it is not a measurement* while being presented as one".
    """
    offending: List[str] = []

    def walk(node: Any, path: str, inherited: Dict[str, Any]) -> None:
        local = inherited
        if isinstance(node, dict):
            local = {**inherited, **_provenance_maps(node, PROVENANCE_SUFFIX)}
        if isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                if isinstance(item, (dict, list, tuple)):
                    walk(item, f"{path}[{i}]", local)
            return
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(value, dict) and _is_provenance_key(key):
                continue
            child = f"{path}.{key}" if path else str(key)
            if isinstance(value, (dict, list, tuple)):
                walk(value, child, local)
                continue
            mark = local.get(key)
            if not isinstance(mark, str):
                mark = node.get(f"{key}{PROVENANCE_SUFFIX}")
            if isinstance(mark, str) and mark.lower() in NON_LIVE \
                    and key not in STRUCTURAL_KEYS:
                offending.append(child)

    walk(payload, "", {})
    return offending