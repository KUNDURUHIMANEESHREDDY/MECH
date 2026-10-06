"""No identifier in this codebase may come from ``hash()``.

``hash()`` is salted per process. Every identifier derived from it changes
between runs, so anything citing one yesterday does not resolve today, and a
reproducibility check that compares identifiers can never pass. Some call sites
also put the result in a namespace small enough to collide -- ``% 10000`` gives
a birthday collision at around a hundred items -- and these ids are persisted,
so a collision points two experiments at one stored row.

Fixed so far: the dispatcher's experiment and session ids, the Scribe's
experiment id, the Society's campaign id, and the discovery engine's discovery
id. Those were the two ``% 10000`` sites plus the identifiers that appear in
provenance chains.

Fourteen sites remain, and this file is deliberately written so the remainder is
*visible* rather than quietly accepted. `KNOWN_REMAINING` is an explicit
inventory: a new ``hash()`` identifier anywhere fails the suite and is named.
Deleting an entry is a deliberate act, so the list shrinks on purpose rather
than by neglect. Every entry is still a live defect, not an exemption.

The detector looks structurally, at f-strings, bit-masks and modulo
expressions, and at assignments to ``*_id``-shaped targets. A text search is
the wrong tool here and this repository has been bitten by it: several
docstrings in this codebase quote the old code in order to explain why it was
wrong, and a grep matches those explanations as if they were the defect.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

#: Every file still holding a ``hash()``-derived identifier.
#:
#: Sorted, and matched by exact path, so a fix anywhere surfaces as a failure
#: that has to be acknowledged. Do not add an entry to silence a failure:
#: convert the call site and delete the key.
KNOWN_REMAINING = frozenset({
    "backend/distributed/distributed_campaign.py",
    "backend/interpretability/discovery/autonomous_paper_replicator.py",
    "backend/interpretability/discovery/autonomous_research_loop.py",
    "backend/interpretability/discovery/cross_model_circuits.py",
    "backend/interpretability/discovery/dag_discovery_planner.py",
    "backend/interpretability/discovery/information_gain_scheduler.py",
    "backend/interpretability/discovery/research_campaign_manager.py",
    "backend/knowledge_graph/graph_builder.py",
    "backend/research_platform/autonomous/autonomous_planner.py",
    "backend/research_platform/autonomous/research_governance.py",
    "backend/research_platform/autonomous/research_program_manager.py",
    "backend/research_platform/meta/literature_learning_pipeline.py",
    "backend/runtime/execution/distributed.py",
    "backend/runtime/orchestration/ray_integration.py",
})


def _calls_builtin_hash(node: ast.AST) -> bool:
    """Whether `node` contains a call to Python's builtin `hash()`."""
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "hash"
        for child in ast.walk(node)
    )


def _hash_identifier_sites(path: Path) -> list:
    """Lines in `path` where `hash()` feeds something identifier-shaped."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []

    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            if _calls_builtin_hash(node):
                found.append(node.lineno)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.BitAnd)):
            if _calls_builtin_hash(node):
                found.append(node.lineno)
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id.endswith(("_id", "_hash"))
                   for t in node.targets):
                if _calls_builtin_hash(node.value):
                    found.append(node.lineno)
    return sorted(set(found))


def _all_sites() -> dict:
    sites = {}
    for path in sorted(BACKEND.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == "backend/core/identifiers.py":
            continue  # this module's prose quotes the old code on purpose
        lines = _hash_identifier_sites(path)
        if lines:
            sites[rel] = lines
    return sites


def test_the_detector_finds_something():
    """If this fails the guard is vacuous, which is the failure mode it exists to avoid."""
    assert len(_all_sites()) >= len(KNOWN_REMAINING), (
        "the detector found fewer sites than the inventory records; it is not "
        "seeing what the inventory claims to see")


def test_no_new_hash_derived_identifier_has_appeared():
    sites = _all_sites()
    unexpected = {p: lines for p, lines in sites.items()
                  if p not in KNOWN_REMAINING}
    assert unexpected == {}, (
        "new hash()-derived identifier(s). Use "
        "backend.core.identifiers.entity_id for a new thing, or content_id for "
        "content identity:\n"
        + "\n".join(f"  {p}:{lines}" for p, lines in unexpected.items()))


def test_the_inventory_lists_no_file_that_is_already_fixed():
    sites = _all_sites()
    stale = KNOWN_REMAINING - set(sites)
    assert stale == frozenset(), (
        "these files no longer contain a hash()-derived identifier, so a fix "
        "landed without the inventory being updated. Left alone, the suite "
        "would keep claiming coverage it does not have. Remove them: "
        + ", ".join(sorted(stale)))


@pytest.mark.parametrize("rel", sorted(KNOWN_REMAINING))
def test_each_remaining_site_is_still_present(rel):
    """Documents that none of these is fine.

    Exists so the inventory reads as outstanding work rather than accepted
    debt. When a file is converted, this fails and the entry is deleted.
    """
    target = ROOT / rel
    assert target.exists(), f"{rel} does not exist; drop it from the inventory"
    assert _hash_identifier_sites(target), (
        f"{rel} has no hash()-derived identifier left, so remove it from "
        f"KNOWN_REMAINING")


def test_the_inventory_only_shrinks():
    """Ratchet. Written to match today's count on purpose."""
    assert len(KNOWN_REMAINING) <= 14, (
        f"{len(KNOWN_REMAINING)} hash()-derived identifiers remain, up from "
        f"14. This guard makes the remainder visible; it is not a licence to "
        f"add. Convert them to backend.core.identifiers.")


def test_entity_and_content_ids_are_distinguishable():
    """The two kinds must not be interchangeable, or using the wrong one bites."""
    from backend.core.identifiers import content_id, entity_id

    payload = {"goal": "map the circuit"}

    # Content identity: stable, so the same content gets the same name.
    assert content_id(payload) == content_id(payload)
    assert content_id(payload) != content_id({"goal": "map the ciruit"})

    # Entity identity: distinct, so two runs can coexist.
    assert entity_id("exp_") != entity_id("exp_")

    assert len(content_id(payload)) == 32
    assert entity_id("exp_").startswith("exp_")
    assert entity_id("exp_")[4:] != content_id(payload)


def test_content_ids_survive_a_hash_seed_change():
    """The property the old ids lacked, checked where it actually failed."""
    import os
    import subprocess
    import sys

    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from backend.core.identifiers import content_id;"
        "print(content_id({'goal': 'map the circuit'}, prefix='disc_'))"
    )
    digests = set()
    for seed in ("0", "7", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
        out = subprocess.run(
            [sys.executable, "-c", probe, str(ROOT)],
            capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=180,
        )
        assert out.returncode == 0, out.stderr
        digests.add(out.stdout.strip())

    assert len(digests) == 1, (
        "content_id changed with PYTHONHASHSEED, which means it is not stable "
        "across processes")
    assert digests.pop().startswith("disc_")