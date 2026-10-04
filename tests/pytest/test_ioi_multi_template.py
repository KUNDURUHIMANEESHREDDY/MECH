"""Multi-template IOI: a result from one frame is not a result about IOI.

Why these tests exist
--------------------
The pipeline already cycled 8 prompt frames and tallied faithfulness per frame,
but nothing tested any of it and nothing read the result. `cross_template_consistent`
was a single boolean that was False in two different situations:

  * the frames *disagreed* -- the circuit works on some surface forms and not
    others, so the aggregate is not a statement about indirect object
    identification;
  * agreement could not be *tested* -- only one frame produced a usable
    measurement, so there was nothing to compare.

A consumer reading `cross_template_consistent: false` would reasonably conclude
the circuit had been shown not to generalise. In the second case it had been
shown nothing at all. Absence of evidence and evidence of absence are not the
same claim, and one boolean cannot carry both.

So the verdict stays fail-closed, and `cross_template_consistency_measured`
separately records whether it was reachable.

The prompt-construction tests matter for a different reason: the ABB corruption
is what makes the IOI task IOI. If the corrupted prompt kept the subject as giver,
the task degrades into something else and every downstream faithfulness figure
describes the wrong experiment. That is checkable without loading weights.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from backend.science.reproducibility.ioi_pipeline import (
    CROSS_TEMPLATE_AGREEMENT_TOLERANCE,
    _IOI_FRAMES,
    _build_prompt,
    _make_high_fidelity_ioi_prompts,
)

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "backend" / "science" / "reproducibility" / "ioi_pipeline.py"


# ── Frame coverage ──────────────────────────────────────────────────────

def test_all_eight_frames_are_used():
    assert len(_IOI_FRAMES) == 8, (
        f"the frame count changed to {len(_IOI_FRAMES)}; the README's "
        f"multi-template table and this file both describe eight")


@pytest.mark.parametrize("n", [8, 9, 20, 40, 100])
def test_every_frame_is_exercised_at_any_prompt_count(n):
    """Rotation, not sampling.

    With random frame choice a small n covers two or three frames and the result
    is frame-specific while looking like a general one. Deterministic rotation
    means even n=8 covers all eight.
    """
    prompts = _make_high_fidelity_ioi_prompts(n, seed=42)

    assert {p["frame_id"] for p in prompts} == set(range(len(_IOI_FRAMES)))


def test_frames_are_assigned_evenly():
    """Not just covered -- covered evenly, so no frame dominates a mean."""
    n = 40
    prompts = _make_high_fidelity_ioi_prompts(n, seed=42)
    counts: Dict[int, int] = {}
    for p in prompts:
        counts[p["frame_id"]] = counts.get(p["frame_id"], 0) + 1

    assert len(set(counts.values())) == 1, (
        f"frames are unevenly covered: {counts}. An uneven split lets one frame "
        f"carry the aggregate.")


def test_frame_assignment_does_not_depend_on_the_seed():
    """The seed varies the names; it must not vary which frame a prompt uses.

    Otherwise `n_templates` and the per-frame tallies would shift with the seed
    and two runs would not be comparable.
    """
    a = _make_high_fidelity_ioi_prompts(24, seed=42)
    b = _make_high_fidelity_ioi_prompts(24, seed=43)

    assert [p["frame_id"] for p in a] == [p["frame_id"] for p in b]
    assert [p["text"] for p in a] != [p["text"] for p in b], (
        "the seed must still change the names, or it does nothing")


def test_prompt_construction_is_deterministic():
    assert (_make_high_fidelity_ioi_prompts(20, seed=42)
            == _make_high_fidelity_ioi_prompts(20, seed=42))


# ── The ABB corruption, which is what makes the task IOI ────────────────

def test_the_clean_prompt_has_the_subject_as_giver():
    frame = _IOI_FRAMES[0]
    text = _build_prompt(frame, "Frank", "Alice", corrupted=False)

    assert text.endswith(f", Frank {frame['verb']} a {frame['item']} to"), text


def test_the_corrupted_prompt_has_the_indirect_object_as_giver():
    """Swap the giver to the IO and the target becomes distributionally wrong.

    If this silently stopped happening the task would stop being indirect object
    identification, and every faithfulness figure would describe a different
    experiment while still being reported as IOI.
    """
    frame = _IOI_FRAMES[0]
    text = _build_prompt(frame, "Frank", "Alice", corrupted=True)

    assert text.endswith(f", Alice {frame['verb']} a {frame['item']} to"), text
    assert "Frank" not in text.split("to the ", 1)[1], (
        "the subject is still the giver in the corrupted prompt, so the target "
        "is not distributionally wrong")


@pytest.mark.parametrize("index", range(8))
def test_every_generated_prompt_pair_keeps_the_corruption(index):
    prompts = _make_high_fidelity_ioi_prompts(8, seed=42)
    p = prompts[index]

    giver_in_corrupted = p["corrupted_text"].split("to the ", 1)[1].split(" ")[1]

    assert giver_in_corrupted == p["indirect_object"], (
        f"frame {p['frame_id']}: corrupted giver is {giver_in_corrupted}, "
        f"expected the indirect object {p['indirect_object']}")
    assert p["subject"] != p["indirect_object"], (
        "subject and indirect object are the same name, so the corruption is a "
        "no-op for this prompt")


def test_the_target_is_the_indirect_object():
    prompts = _make_high_fidelity_ioi_prompts(8, seed=42)

    for p in prompts:
        assert p["target"] == f" {p['indirect_object']}"
        assert p["text"].endswith("to"), "the prompt must end at the target"


def test_names_are_distinct_within_a_prompt():
    """Three distinct names: `rng.sample` guarantees it, and the corruption
    depends on it."""
    for p in _make_high_fidelity_ioi_prompts(20, seed=42):
        assert p["subject"] != p["indirect_object"]


# ── The verdict, and whether it was reachable ──────────────────────────

def _consistency_block(usable: List[float]) -> Dict[str, Any]:
    """Re-derive the verdict the pipeline computes, from usable values.

    Mirrors the block in `_run_live` so the rule can be tested without loading
    GPT-2. `test_the_pipeline_computes_consistency_the_same_way` checks the
    mirror against the real source, so this cannot drift into testing itself.
    """
    measured = len(usable) >= 2
    spread = round(max(usable) - min(usable), 4) if measured else None
    return {
        "cross_template_consistent": bool(
            measured and spread < CROSS_TEMPLATE_AGREEMENT_TOLERANCE),
        "cross_template_consistency_measured": measured,
        "cross_template_spread": spread,
    }


def test_agreement_needs_at_least_two_frames():
    """The core fix: one frame is untested, not refuted."""
    block = _consistency_block([0.72])

    assert block["cross_template_consistency_measured"] is False
    assert block["cross_template_consistent"] is False, (
        "fail-closed: the verdict stays False when there is nothing to test")
    assert block["cross_template_spread"] is None


def test_frames_that_agree_are_consistent():
    block = _consistency_block([0.72, 0.74, 0.71, 0.73])

    assert block["cross_template_consistency_measured"] is True
    assert block["cross_template_consistent"] is True
    assert block["cross_template_spread"] == 0.03


def test_the_measured_frames_agree_within_tolerance():
    """The real per-frame numbers, 40 prompts over 8 frames, live weights.

    Frame 4 (market/bought/apple) produced one usable prompt out of five and
    scored 1.0. Included, it made the spread 0.3831 and the pipeline report
    that the frames disagreed -- which was an artifact of comparing a
    single-prompt mean against five-prompt means. Excluded, the seven comparable
    frames span 0.1851 and agree.

    Recorded because the first reading was wrong and the correction is the
    finding: the disagreement was manufactured by the sampling, not present in
    the frames.
    """
    measured = [0.802, 0.7152, 0.6169, 0.7445, 0.7017, 0.7894, 0.7697]
    block = _consistency_block(measured)

    assert block["cross_template_consistency_measured"] is True
    assert block["cross_template_spread"] == pytest.approx(0.1851, abs=1e-4)
    assert block["cross_template_consistent"] is True

    # And the contamination, stated: the same set plus the one-prompt frame.
    contaminated = _consistency_block(measured + [1.0])
    assert contaminated["cross_template_spread"] == pytest.approx(0.3831, abs=1e-4)
    assert contaminated["cross_template_consistent"] is False


def test_a_frame_below_the_usable_minimum_is_excluded_from_the_comparison():
    """One usable prompt is an observation, not a frame-level estimate.

    Folding it in makes the verdict a function of which prompts happened to
    work: it can manufacture disagreement from one lucky prompt or agreement
    from one unlucky one.

    This calls the pipeline's own `_comparable_templates`. An earlier version of
    this test re-implemented the filter locally and passed even when the
    pipeline's filter was deleted -- a guard agreeing with itself. The negative
    control that caught it is why the rule lives in a function.
    """
    from backend.science.reproducibility.ioi_pipeline import (
        MIN_USABLE_PROMPTS_PER_TEMPLATE,
        _comparable_templates,
    )

    assert MIN_USABLE_PROMPTS_PER_TEMPLATE == 3

    per_template = [
        {"frame_id": 0, "usable": 5, "n": 5, "circuit_faithfulness": 0.802,
         "place": "store", "verb": "gave", "item": "drink"},
        {"frame_id": 1, "usable": 5, "n": 5, "circuit_faithfulness": 0.6169,
         "place": "library", "verb": "offered", "item": "book"},
        {"frame_id": 4, "usable": 1, "n": 5, "circuit_faithfulness": 1.0,
         "place": "market", "verb": "bought", "item": "apple"},
    ]

    compared, excluded = _comparable_templates(per_template)

    assert [t["frame_id"] for t in compared] == [0, 1]
    assert [e["frame_id"] for e in excluded] == [4]
    assert "below the 3 required" in excluded[0]["why"]


def test_an_excluded_frame_is_reported_not_silently_dropped():
    """A frame that produced almost nothing usable is itself a finding.

    Dropping it would make the aggregate look better-covered than it is: 7
    comparable frames out of 8 reads very differently from 8 out of 8.
    """
    from backend.science.reproducibility.ioi_pipeline import _comparable_templates

    per_template = [{"frame_id": i, "usable": 5, "n": 5,
                     "circuit_faithfulness": 0.7} for i in range(7)]
    per_template.append({"frame_id": 7, "usable": 1, "n": 5,
                         "circuit_faithfulness": 1.0, "place": "market",
                         "verb": "bought", "item": "apple"})

    compared, excluded = _comparable_templates(per_template)

    assert len(compared) == 7
    assert len(excluded) == 1
    assert excluded[0]["place"] == "market"
    assert excluded[0]["usable"] == 1


def test_a_frame_with_no_value_at_all_is_excluded_everywhere():
    """`circuit_faithfulness: None` is not a frame-level estimate either."""
    from backend.science.reproducibility.ioi_pipeline import _comparable_templates

    per_template = [
        {"frame_id": 0, "usable": 5, "n": 5, "circuit_faithfulness": 0.7},
        {"frame_id": 1, "usable": 0, "n": 5, "circuit_faithfulness": None},
    ]

    compared, excluded = _comparable_templates(per_template)

    assert [t["frame_id"] for t in compared] == [0]
    assert excluded == [], (
        "a frame with no value is dropped by the value filter, not reported as "
        "insufficient -- listing it twice would miscount")


def test_frames_that_really_disagree_are_inconsistent():
    """A spread well past the tolerance, from fully-covered frames.

    Distinct from the measured case above: these frames all have five usable
    prompts, so the verdict is a real negative rather than a sampling artifact.
    """
    block = _consistency_block([0.30, 0.92, 0.55, 0.61])

    assert block["cross_template_consistency_measured"] is True
    assert block["cross_template_spread"] == pytest.approx(0.62, abs=1e-4)
    assert block["cross_template_consistent"] is False


def test_the_tolerance_boundary_is_inclusive_of_disagreement():
    """Exactly at the tolerance counts as disagreement."""
    block = _consistency_block([0.60, 0.60 + CROSS_TEMPLATE_AGREEMENT_TOLERANCE])

    assert block["cross_template_consistent"] is False


def test_the_tolerance_is_reported_alongside_the_verdict():
    """A judgement call about what counts as generalisation must be visible.

    Otherwise a reader who disagrees with 0.25 has no way to find it.
    """
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    keys: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)

    for required in ("cross_template_consistent",
                     "cross_template_consistency_measured",
                     "cross_template_spread",
                     "cross_template_reason",
                     "cross_template_tolerance",
                     "per_template"):
        assert required in keys, f"{required} is not in the reported metrics"


def test_the_pipeline_computes_consistency_the_same_way():
    """Guard the mirror used by the tests above against drift.

    If the pipeline's rule changed and this helper did not, the tests above
    would keep passing while describing behaviour that no longer exists -- a
    guard testing its own assumptions.
    """
    source = PIPELINE.read_text(encoding="utf-8")

    assert "cross_template_measured = len(usable_faithfulness) >= 2" in source, (
        "the pipeline's testability rule changed; update _consistency_block")
    assert ("cross_template_spread < CROSS_TEMPLATE_AGREEMENT_TOLERANCE"
            in source), (
        "the pipeline's tolerance comparison changed; update _consistency_block")
    assert source.count("CROSS_TEMPLATE_AGREEMENT_TOLERANCE = ") == 1, (
        "the tolerance must be defined once, not shadowed per branch")


# ── The measured per-frame table must match what the README publishes ───

def test_the_readme_publishes_the_measured_per_frame_table():
    """Every per-frame figure the README states must be in the artifact.

    Re-running the pipeline and forgetting to update the README is a test
    failure, not a quiet discrepancy.
    """
    import re

    artifact = ROOT / "docs" / "results" / "ioi_template_audit.json"
    assert artifact.exists(), (
        f"{artifact.name} is missing, so the README's per-frame table has "
        f"nothing to be checked against")

    run = json.loads(artifact.read_text(encoding="utf-8"))["run"]
    assert run.get("provenance") == "live", (
        "the artifact backing a published figure is not live")

    metrics = run["observed_metrics"]
    per_frame = metrics["per_template"]
    assert len(per_frame) == len(_IOI_FRAMES)

    flat = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))

    for entry in per_frame:
        label = f"{entry['place']}/{entry['verb']}/{entry['item']}"
        assert label in flat, f"README does not list the measured frame {label}"
        value = entry["circuit_faithfulness"]
        if value is not None:
            assert f"{value}" in flat, (
                f"README does not quote {label} faithfulness {value}")

    assert f"{metrics['cross_template_spread']}" in flat, (
        "README does not quote the measured cross-template spread")
    assert str(metrics["cross_template_consistent"]) in flat, (
        "README does not state the measured consistency verdict")


def test_the_readme_stated_verdict_matches_the_measurement():
    """Bidirectional: the README may not contradict the artifact in either direction.

    Written when the measurement said the frames disagreed, so it asserted a
    negative. The measurement then changed -- excluding a one-prompt frame moved
    the spread from 0.3831 to 0.1851 and the verdict to `true` -- which is
    exactly when a stale claim is most likely to survive, because the earlier
    finding was alarming and felt worth keeping.

    So the guard no longer encodes an expected outcome. It reads the verdict the
    README publishes and requires it to equal the measured one.
    """
    import re

    artifact = ROOT / "docs" / "results" / "ioi_template_audit.json"
    if not artifact.exists():
        pytest.skip("no artifact yet")

    metrics = json.loads(artifact.read_text(encoding="utf-8"))["run"][
        "observed_metrics"]
    assert metrics["cross_template_consistency_measured"] is True, (
        "the published figure rests on fewer than two comparable frames; the "
        "README's cross-template claim must be revisited")

    flat = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    stated = re.search(
        r"cross_template_consistent:\s*`?(true|false)`?", flat)
    assert stated, (
        "the README does not state a cross_template_consistent verdict at all")

    assert stated.group(1) == str(metrics["cross_template_consistent"]).lower(), (
        f"the README states cross_template_consistent: {stated.group(1)} but the "
        f"measured value is {metrics['cross_template_consistent']}. Whichever is "
        f"right, they cannot both stand.")


def test_the_tolerance_is_stated_where_the_verdict_is_published():
    """0.25 is a judgement call. A reader who disagrees must be able to find it
    next to the number it governs, not buried in the pipeline."""
    import re

    flat = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))

    assert f"{CROSS_TEMPLATE_AGREEMENT_TOLERANCE}" in flat, (
        "the README publishes a consistency verdict without naming the "
        "tolerance that produced it")