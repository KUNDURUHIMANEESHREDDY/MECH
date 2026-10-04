"""The results artifact's provenance must be derived, not asserted.

The rule under test
------------------
`scripts/capture_results.py` writes `docs/results/capture.json`, and the README's
Results section quotes it. That artifact carried a literal

    "meta": { ..., "provenance": "live", ... }

built at the top of `main()`, before any capture had run and never revised
afterwards -- `provenance` appeared exactly once in the entire file, in that
literal. So the one artifact the repository cites as measured evidence declared
itself live unconditionally, and would have kept doing so if a capture ever
failed quietly.

Two things are asserted here, one behavioural and one structural:

* `derive_meta_provenance` reports what the captures produced and withholds
  `live` when any expected section is absent.
* The `meta` literal no longer hardcodes the label, and `EXPECTED_SECTIONS`
  matches the keys `main()` actually assigns -- checked against the AST, so the
  list cannot drift away from the code it describes.

Negative controls run for this file: injecting `"provenance": "live"` back into
the `meta` literal, deleting a section from `EXPECTED_SECTIONS`, and making the
derivation truthiness-based instead of presence-based were each caught.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "capture_results.py"
LIVE = "live"
UNAVAILABLE = "unavailable"


def _derive(data: Dict[str, Any]) -> Dict[str, Any]:
    from scripts.capture_results import derive_meta_provenance

    return derive_meta_provenance(data)


def _full() -> Dict[str, Any]:
    from scripts.capture_results import EXPECTED_SECTIONS

    return {section: {"result": 1} for section in EXPECTED_SECTIONS}


# ── Behaviour ──────────────────────────────────────────────────────────

def test_a_complete_run_reports_live():
    out = _derive(_full())

    assert out["provenance"] == LIVE
    assert out["measured_sections"] == "8/8"
    assert "reason" not in out, "a complete run must not carry a failure reason"


@pytest.mark.parametrize("absent", [
    "architecture", "sanity_checks", "ioi", "logit_lens",
    "head_sweep", "layer_ablation", "steering", "inspection",
])
def test_any_missing_section_withholds_live(absent):
    """Every section individually, not a sample of them.

    A derivation that withheld on one missing key but not another would still be
    the defect this file exists to close -- an artifact claiming a complete
    measurement run from an incomplete one.
    """
    from scripts.capture_results import EXPECTED_SECTIONS

    data = _full()
    del data[absent]

    out = _derive(data)

    assert out["provenance"] == UNAVAILABLE, (
        f"a run missing {absent} was still labelled {LIVE!r}")
    assert out["measured_sections"] == f"{len(EXPECTED_SECTIONS) - 1}/8"
    assert absent in out["reason"], "the reason must name the missing section"
    assert "must not be cited as one" in out["reason"]


def test_an_empty_section_still_counts_as_measured():
    """Presence, not truthiness.

    An empty head sweep is a real measurement -- it swept every head and found
    none worth reporting. Treating an empty-but-present section as missing would
    withhold `live` from a complete run, which is a different kind of lie: it
    makes honest output look broken.
    """
    data = _full()
    data["head_sweep"] = {}
    data["steering"] = []
    data["layer_ablation"] = 0

    out = _derive(data)

    assert out["provenance"] == LIVE, (
        "an empty but present section is a measurement, not a missing one")
    assert out["measured_sections"] == "8/8"


def test_an_empty_artifact_withholds_live():
    out = _derive({})

    assert out["provenance"] == UNAVAILABLE
    assert out["measured_sections"] == "0/8"


def test_an_incomplete_run_replaces_the_completeness_note():
    """The blanket note claims every value was measured.

    If a section is missing that sentence is false, so it has to be replaced
    rather than left sitting above a withheld label.
    """
    data = _full()
    del data["ioi"]
    data["meta"] = {"note": "Every value is measured from live GPT-2 weights."}

    out = _derive(data)

    assert out["provenance"] == UNAVAILABLE
    assert "INCOMPLETE RUN" in out["note"]
    assert "Every value is measured" not in out["note"]


# ── Structure ──────────────────────────────────────────────────────────

def test_the_meta_literal_no_longer_asserts_live():
    """The original defect, pinned at its original location.

    Parsed rather than grepped: the string `live` appears legitimately all over
    this file for model ids and section names, so only the assignment inside the
    `meta` dict literal is of interest.
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and key.value == "provenance"
                    and isinstance(value, ast.Constant) and value.value == LIVE):
                offenders.append(f"line {node.lineno}")

    assert not offenders, (
        "the capture artifact asserts provenance='live' at "
        + ", ".join(offenders)
        + " instead of deriving it from the sections that produced data")


def test_expected_sections_matches_what_main_actually_assigns():
    """The list must describe the code, so it is checked against the code.

    `main()` populates `data` via `data[<section>] = timed(<label>, <fn>)`.
    Those subscript keys are the sections that can exist; EXPECTED_SECTIONS is
    the claim that all of them are required. If a capture is added and the tuple
    is not updated, this fails rather than the derivation quietly ignoring it.
    """
    from scripts.capture_results import EXPECTED_SECTIONS

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    main_node = next(n for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name == "main")

    assigned = set()
    for node in ast.walk(main_node):
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "timed"):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)):
                    assigned.add(target.slice.value)

    assert assigned, "could not find the data[...] = timed(...) assignments"
    assert set(EXPECTED_SECTIONS) == assigned, (
        "EXPECTED_SECTIONS has drifted from the captures main() performs. "
        f"declared-only: {sorted(set(EXPECTED_SECTIONS) - assigned)}; "
        f"assigned-only: {sorted(assigned - set(EXPECTED_SECTIONS))}")


def test_the_derivation_is_reachable_from_main():
    """Otherwise the function is decoration and main() keeps asserting."""
    source = inspect.getsource(
        __import__("scripts.capture_results", fromlist=["main"]).main)

    assert "derive_meta_provenance" in source, (
        "main() no longer calls derive_meta_provenance, so the artifact's "
        "provenance is whatever the meta literal says")
    assert '"provenance": "live"' not in source, (
        "main() still hardcodes a live provenance")