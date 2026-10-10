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
    from scripts.capture_results import EXPECTED_SECTIONS

    out = _derive(_full())

    total = len(EXPECTED_SECTIONS)
    assert out["provenance"] == LIVE
    assert out["measured_sections"] == f"{total}/{total}"
    assert "reason" not in out, "a complete run must not carry a failure reason"


def _each_section() -> list:
    """Every required section, so a new one is covered without editing a list.

    A hardcoded parameter list is how a guard rots: adding a section left the
    test asserting about the old eight and never checking the ninth. This reads
    the tuple, and `test_every_expected_section_is_exercised` below makes sure
    that stays a real check rather than a vacuous one.
    """
    from scripts.capture_results import EXPECTED_SECTIONS

    return sorted(EXPECTED_SECTIONS)


def test_every_expected_section_is_exercised():
    """The guard above is only worth its name if it really iterates.

    It delegates to `_each_section`, so this asserts the count is what the
    registry says rather than trusting the helper silently to have shrunk.
    """
    from scripts.capture_results import CAPTURES, EXPECTED_SECTIONS

    exercised = set(_each_section())
    assert exercised == set(EXPECTED_SECTIONS)
    assert exercised == {section for section, _label, _name in CAPTURES}


@pytest.mark.parametrize("absent", _each_section())
def test_any_missing_section_withholds_live(absent):
    """Every section individually, not a sample of them.

    A derivation that withheld on one missing key but not another would still be
    the defect this file exists to close -- an artifact claiming a complete
    measurement run from an incomplete one.
    """
    from scripts.capture_results import EXPECTED_SECTIONS

    total = len(EXPECTED_SECTIONS)
    data = _full()
    del data[absent]

    out = _derive(data)

    assert out["provenance"] == UNAVAILABLE, (
        f"a run missing {absent} was still labelled {LIVE!r}")
    assert out["measured_sections"] == f"{total - 1}/{total}"
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

    from scripts.capture_results import EXPECTED_SECTIONS

    total = len(EXPECTED_SECTIONS)
    assert out["provenance"] == LIVE, (
        "an empty but present section is a measurement, not a missing one")
    assert out["measured_sections"] == f"{total}/{total}"


def test_an_empty_artifact_withholds_live():
    from scripts.capture_results import EXPECTED_SECTIONS

    out = _derive({})

    total = len(EXPECTED_SECTIONS)
    assert out["provenance"] == UNAVAILABLE
    assert out["measured_sections"] == f"0/{total}"


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


def test_expected_sections_matches_the_captures_actually_registered():
    """The list must describe the code, so it is checked against the code.

    `CAPTURES` is the registry `run_captures()` iterates; `EXPECTED_SECTIONS` is
    the claim that all of its entries are required for a `live` label. If a
    capture is registered and the tuple is not updated, this fails rather than
    the derivation quietly ignoring it.

    Parsed from the AST rather than imported so the comparison is between two
    independent declarations. Deriving EXPECTED_SECTIONS from CAPTURES would make
    this compare a tuple with itself and pass no matter what either said.
    """
    from scripts.capture_results import CAPTURES, EXPECTED_SECTIONS

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    captures_node = next(
        n for n in tree.body
        if isinstance(n, ast.AnnAssign)
        and getattr(n.target, "id", None) == "CAPTURES"
    )
    registered = [
        element.elts[0].value
        for element in captures_node.value.elts
        if isinstance(element, ast.Tuple) and element.elts
    ]

    assert registered, "could not read the CAPTURES registry from the AST"
    assert set(EXPECTED_SECTIONS) == set(registered), (
        "EXPECTED_SECTIONS has drifted from the captures that actually run. "
        f"declared-only: {sorted(set(EXPECTED_SECTIONS) - set(registered))}; "
        f"registered-only: {sorted(set(registered) - set(EXPECTED_SECTIONS))}")

def test_every_registered_capture_names_a_distinct_section():
    """A duplicate key would silently overwrite an earlier measurement.

    `data[section] = result.payload` means two captures sharing a key leave one
    result unrepresented in `data` while the run report claims both ran.
    """
    from scripts.capture_results import CAPTURES

    keys = [section for section, _label, _fn_name in CAPTURES]
    assert len(keys) == len(set(keys)), f"duplicate section key in CAPTURES: {keys}"


def test_every_registered_capture_name_resolves_to_a_callable():
    """A typo in the registry is a NameError at run time, after the sweep.

    `run_captures` does handle the miss -- it records the section as failed
    rather than raising -- which is exactly why a typo would not announce
    itself until someone read the artifact. Checked at collection time instead.
    """
    import scripts.capture_results as cap

    missing = [
        f"{section} -> {name}"
        for section, _label, name in cap.CAPTURES
        if not callable(getattr(cap, name, None))
    ]
    assert not missing, f"CAPTURES names something that is not a capture: {missing}"


def test_every_required_section_declares_the_keys_it_validates():
    """`REQUIRED_KEYS` drives the structural validation, so a section missing
    from it is one whose payload shape is never checked."""
    from scripts.capture_results import CAPTURES, REQUIRED_KEYS

    registered = {section for section, _label, _fn_name in CAPTURES}
    assert set(REQUIRED_KEYS) == registered, (
        "REQUIRED_KEYS and CAPTURES disagree. "
        f"no validation declared for: {sorted(registered - set(REQUIRED_KEYS))}; "
        f"validation for unregistered sections: "
        f"{sorted(set(REQUIRED_KEYS) - registered)}")


def test_the_derivation_is_reachable_from_main():
    """Otherwise the function is decoration and main() keeps asserting."""
    source = inspect.getsource(
        __import__("scripts.capture_results", fromlist=["main"]).main)

    assert "derive_meta_provenance" in source, (
        "main() no longer calls derive_meta_provenance, so the artifact's "
        "provenance is whatever the meta literal says")
    assert '"provenance": "live"' not in source, (
        "main() still hardcodes a live provenance")