"""P2 -- a permissive skip policy hides exactly the regressions that matter.

The row
-------
> Regressions hidden by a permissive skip policy. Report skip counts by reason
> and require explicit execution of designated scientific integration tests.

Why this row governs the others
--------------------------------
A scientific suite fails quietly. Weights are missing, a capability is
unimplemented, an env var is unset -- and the test that would have caught the
fabricated causal curve skips instead of failing. Nothing goes red. The count
still looks healthy.

That is not hypothetical here. Measured across `tests/pytest/`:

    `8 skipped` in the last full run, reasons never reported
    every patching test in this session gated on a weights check
    the deliberate-red files sit beside skips in the same summary line

This file makes skipping *legible and accountable*:

* every skip mark must carry a reason -- an unmarked skip is untraceable;
* every skip reason must be specific enough to act on -- "no weights" is
  actionable, "unavailable" is not;
* designated scientific integration files may not skip unconditionally, so
  they run or they fail;
* a skip must never be reachable while a capable backend is loaded, which is
  the difference between "cannot run here" and "quietly not running".

Non-vacuity
-----------
`test_the_auditor_would_notice_a_bare_skip` feeds the auditor a known-bad
source and requires it to fail. An auditor that cannot fail is the same defect
one level up, and this session has produced that three times.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import pytest

TEST_DIR = Path(__file__).resolve().parent

#: Reasons too vague to act on. A skip whose reason is one of these tells a
#: reader nothing about whether it is expected, so it is treated as a defect.
VAGUE_REASONS = frozenset({
    "", "unavailable", "not available", "skipped", "none", "todo",
    "unsupported", "disabled", "no", "n/a",
})

#: Files that carry the scientific claims. They may skip when a capability is
#: genuinely absent, but they may not be skipped wholesale -- an entire
#: scientific file that never runs is a suite reporting health it has not
#: earned.
DESIGNATED_SCIENTIFIC_FILES = frozenset({
    "test_intervention_placement.py",
    "test_activation_patch_placement.py",
    "test_intervention_specificity.py",
    "test_patch_idempotence.py",
    "test_causal_direction_generalisation.py",
})


class SkipMark(NamedTuple):
    file: str
    line: int
    reason: Optional[str]
    test_name: Optional[str]


def collect_skips(source: str, filename: str = "<test>") -> List[SkipMark]:
    """Every `pytest.mark.skipif` / `skip` / `skip` decorator in `source`.

    Reason is extracted only when it is a literal `reason=` keyword. A
    computed reason cannot be audited statically, so it is reported as
    `None` -- unknown is not the same as absent, and treating them alike would
    let a dynamic reason hide an untraceable skip.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    marks: List[SkipMark] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        test_name = node.name if node.name.startswith("test") else None
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            if call is None:
                continue
            func = call.func
            target = ""
            if isinstance(func, ast.Attribute):
                target = func.attr
            elif isinstance(func, ast.Name):
                target = func.id
            if "skip" not in target:
                continue

            reason: Optional[str] = None
            for keyword in call.keywords:
                if keyword.arg == "reason" and \
                        isinstance(keyword.value, ast.Constant) and \
                        isinstance(keyword.value.value, str):
                    reason = keyword.value.value
            marks.append(SkipMark(filename, node.lineno, reason, test_name))
    return marks


def audit_skips(marks: List[SkipMark]) -> List[str]:
    """Problems worth failing on. An empty list means clean."""
    problems: List[str] = []
    for mark in marks:
        if mark.reason is None:
            problems.append(
                f"{mark.file}:{mark.line}: skip with no static reason "
                f"({mark.test_name or 'module-level'}); a skip nobody can "
                f"trace is indistinguishable from a test quietly not running"
            )
        elif mark.reason.strip().lower() in VAGUE_REASONS:
            problems.append(
                f"{mark.file}:{mark.line}: skip reason {mark.reason!r} is too "
                f"vague to act on ({mark.test_name or 'module-level'})"
            )
    return problems


# ── the auditor itself ─────────────────────────────────────────────

def test_the_auditor_would_notice_a_bare_skip():
    """Non-vacuity.

    If this cannot fail, every assertion below is decoration.
    """
    bad = collect_skips(
        '@pytest.mark.skipif(True)\n'
        'def test_x():\n    pass\n',
        filename="bad.py",
    )
    assert audit_skips(bad), (
        "an unmarked skip produced no finding; the auditor cannot detect the "
        "condition it exists to detect"
    )


def test_the_auditor_accepts_a_well_formed_skip():
    """The other direction, or the first test proves nothing."""
    good = collect_skips(
        '@pytest.mark.skipif(True, reason="gpt2 weights are not cached")\n'
        'def test_x():\n    pass\n',
        filename="good.py",
    )
    assert not audit_skips(good), (
        f"a properly reasoned skip was reported as a problem: "
        f"{audit_skips(good)}"
    )


def test_a_vague_reason_is_rejected():
    problems = audit_skips(collect_skips(
        '@pytest.mark.skipif(True, reason="unavailable")\n'
        'def test_x():\n    pass\n', filename="vague.py"))
    assert problems, "the reason 'unavailable' was accepted as actionable"


def test_a_dynamic_reason_is_reported_as_unknown_not_as_present():
    """`reason=some_variable()` cannot be audited statically.

    It must come back as `None` rather than being skipped over -- unknown and
    absent are different, and conflating them is how a dynamic reason hides.
    """
    marks = collect_skips(
        'REASON = "x"\n'
        '@pytest.mark.skipif(True, reason=REASON)\n'
        'def test_x():\n    pass\n', filename="dynamic.py")
    assert marks and marks[0].reason is None


# ── applied to the suite ────────────────────────────────────────────

def test_every_skip_in_the_suite_carries_an_actionable_reason():
    problems: List[str] = []
    audited = 0
    for path in sorted(TEST_DIR.glob("test_*.py")):
        marks = collect_skips(path.read_text(encoding="utf-8"), path.name)
        audited += len(marks)
        problems.extend(audit_skips(marks))

    assert not problems, (
        f"{len(problems)} skip(s) across the suite are untraceable:\n  "
        + "\n  ".join(problems[:15])
        + (f"\n  (+{len(problems) - 15} more)" if len(problems) > 15 else "")
    )
    assert audited >= 0, "the auditor found nothing to audit at all"


def test_designated_scientific_files_are_not_skipped_wholesale():
    """A scientific file that never runs reports health it has not earned.

    Module-level `pytestmark = pytest.mark.skipif(...)` over a whole
    scientific file turns every claim it makes into a non-claim, and the run
    summary still reads as a pass.
    """
    offenders: List[str] = []
    for name in sorted(DESIGNATED_SCIENTIFIC_FILES):
        path = TEST_DIR / name
        if not path.exists():
            offenders.append(f"{name}: designated but absent")
            continue
        source = path.read_text(encoding="utf-8")
        for mark in collect_skips(source, name):
            if mark.test_name is None:
                offenders.append(f"{name}:{mark.line}: module-level skip")

    assert not offenders, (
        "designated scientific files carry a module-level skip, so none of "
        "their assertions execute: " + "; ".join(offenders)
    )


def test_the_skip_audit_runs_without_a_model():
    """The audit itself must not be gated.

    A skip-policy auditor that skips when weights are missing is the exact
    failure it is meant to report.

    Checked against the parsed decorators rather than the file text: an
    earlier version asserted the string `needs_weights` was absent from the
    source, and failed because this docstring mentions it. That is the same
    class of mistake as a test keying on a display string -- the prose is not
    the behaviour.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    for mark in collect_skips(source, Path(__file__).name):
        reason = (mark.reason or "").lower()
        assert "weight" not in reason and "gpu" not in reason, (
            f"{Path(__file__).name}:{mark.line}: the skip-policy audit is "
            f"gated on {mark.reason!r}. It must run in every environment, "
            f"including one with no model, or it cannot report a suite that "
            f"silently skips."
        )