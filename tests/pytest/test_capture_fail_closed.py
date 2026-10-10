"""A capture that fails must still leave an artifact that says so.

The rule under test
------------------
`scripts/capture_results.py` writes `docs/results/capture.json`, and the README's
Results section is generated from it. `derive_meta_provenance` was written to
withhold `provenance: "live"` when a section did not produce data -- fail-closed,
as the surrounding comment claimed.

It could not deliver that. `main()` ran each capture as a bare statement:

    data["ioi"] = timed("ioi (3 templates)", capture_ioi)

with no exception boundary anywhere. So an exception inside `capture_ioi()`:

1. propagated out of `main()`,
2. meant `capture.json` was never written,
3. meant the derivation never ran, so its `unavailable` verdict never existed.

The fail-closed artifact did not exist. There was no partial result, no record
of which sections had completed, and — because nothing was written — no way to
tell an aborted run from one that had never been attempted. A caller checking
only "does `capture.json` exist" would read the *previous* run's artifact.

Six behaviours are asserted here, and each fails on the pre-fix code:

* `run_section` absorbs an exception instead of propagating it.
* A failed section is represented: status, timings, error type, message.
* `run_captures` omits the failed section's data while other sections still run.
* `main()` writes the artifact anyway, with `provenance: "unavailable"`.
* `main()` exits non-zero, so a broken capture cannot look successful.
* The artifact is written atomically, so a crash cannot truncate it.

Negative controls: making `run_section` re-raise, making `main()` return 0 on
an incomplete run, and restoring the truncate-in-place write were each caught.
On the last one the control reproduces both halves of the defect at once -- the
previous good artifact destroyed, and a bare `NaN` token written into the file
that replaces it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "capture_results.py"

from scripts import capture_results as cap  # noqa: E402

LIVE = "live"
UNAVAILABLE = "unavailable"


# ── Fixtures ────────────────────────────────────────────────────────────

def _payload(section: str) -> Dict[str, Any]:
    """A minimal payload satisfying `REQUIRED_KEYS` for `section`."""
    return {key: "ok" for key in cap.REQUIRED_KEYS[section]}


def _stub_captures(monkeypatch, failing: Tuple[str, ...] = ()) -> None:
    """Replace every registered capture with a stub; named ones raise.

    Stubs the module attributes rather than replacing `CAPTURES`, so these tests
    drive the real registry through the real name resolution instead of a
    stand-in for it.
    """
    for section, _label, name in cap.CAPTURES:
        if section in failing:
            def boom(_section: str = section) -> None:
                raise RuntimeError(f"{_section} exploded")
            monkeypatch.setattr(cap, name, boom)
        else:
            def stub(_section: str = section) -> Dict[str, Any]:
                return _payload(_section)
            monkeypatch.setattr(cap, name, stub)


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch):
    """Point the script's output at a tmp dir and keep it off the real engine."""
    monkeypatch.setattr(cap, "OUT_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(cap, "IMG_DIR", str(tmp_path / "images"))
    monkeypatch.setattr(cap, "render_figures", lambda data: [])
    return tmp_path


# ── run_section: the failure must be absorbed ───────────────────────────

def test_a_failing_capture_does_not_propagate():
    """The defect, at its source.

    On the pre-fix code this was `timed()` calling `fn()` bare, and the
    exception escaped `main()`. Everything downstream of here depends on this
    call returning.
    """
    def boom():
        raise RuntimeError("engine refused to load")

    result = cap.run_section("ioi", "ioi", boom)

    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
    assert "engine refused to load" in result.error_message


def test_a_failure_is_recorded_not_just_counted():
    """Status alone is not enough to resume a run.

    The artifact has to say which section failed, when, how long it had been
    running, and why -- that is what makes an interrupted run actionable rather
    than merely detectable.
    """
    def boom():
        raise ValueError("no cached activations")

    result = cap.run_section("head_sweep", "sweep", boom)
    record = result.as_dict()

    assert record["status"] == "failed"
    assert record["started_at"] and record["finished_at"]
    assert isinstance(record["duration_s"], float)
    assert record["error"]["type"] == "ValueError"
    assert record["error"]["message"] == "no cached activations"
    assert "ValueError" in record["error"]["traceback"]


def test_a_success_carries_timings_and_the_payload():
    result = cap.run_section("ioi", "ioi", lambda: _payload("ioi"))

    assert result.status == "ok"
    assert result.error_type is None
    assert isinstance(result.duration_s, float)
    assert result.payload == _payload("ioi")
    assert "error" not in result.as_dict(), "a clean run must not carry an error block"


def test_the_payload_is_not_duplicated_into_the_run_report():
    """`as_dict` is the report; the measurement is `data[section]`.

    Inlining it would double the artifact's size for no reader.
    """
    result = cap.run_section("ioi", "ioi", lambda: _payload("ioi"))
    assert "payload" not in result.as_dict()


def test_a_capture_that_raises_base_exception_is_still_absorbed():
    """KeyboardInterrupt mid-sweep.

    This is the case that makes the artifact worth having at all: a Ctrl-C
    during the 144-head sweep loses the timings and the sections that already
    completed, because nothing was written on the way out. Absorbing
    `BaseException` means an interrupted run reports how far it got.
    """
    def interrupted():
        raise KeyboardInterrupt()

    result = cap.run_section("head_sweep", "sweep", interrupted)

    assert result.status == "failed"
    assert result.error_type == "KeyboardInterrupt"
    assert result.duration_s is not None


# ── run_captures: isolation ─────────────────────────────────────────────

def test_one_failure_does_not_stop_the_others(monkeypatch):
    _stub_captures(monkeypatch, failing=("ioi",))

    data, results = cap.run_captures()

    assert "ioi" not in data, "a failed section must not put data in `data`"
    assert "head_sweep" in data, "an unrelated section must still be captured"
    assert len(results) == len(cap.EXPECTED_SECTIONS)


def test_patching_a_capture_name_actually_injects_the_failure(monkeypatch):
    """The registry must resolve captures by name, not hold live references.

    This was a real defect found by trying to use the obvious fault-injection
    handle and watching it do nothing: `monkeypatch.setattr` rebinds the module
    attribute, a registry holding function objects keeps calling the original,
    and the run reports `8/8` while the injected failure never happened. In a
    change about failures not passing unnoticed, a broken injection path is
    worth its own test.
    """
    def boom():
        raise RuntimeError("injected")

    monkeypatch.setattr(cap, "capture_ioi", boom)

    data, results = cap.run_captures()

    by_section = {r.section: r for r in results}
    assert by_section["ioi"].status == "failed"
    assert by_section["ioi"].error_message == "injected"
    assert "ioi" not in data


def test_a_capture_name_that_does_not_exist_is_reported_not_raised(monkeypatch):
    """A registry typo must not abort the run.

    `run_captures` turns the NameError into a recorded failure, which is the
    difference between "one section is unavailable" and "no artifact at all".
    """
    monkeypatch.setattr(cap, "CAPTURES", (("ioi", "ioi", "capture_nonexistent"),))

    data, results = cap.run_captures()

    assert data == {}
    assert results[0].status == "failed"
    assert results[0].error_type == "NameError"
    assert "capture_nonexistent" in results[0].error_message


def test_a_failed_section_still_has_a_run_report_entry(monkeypatch):
    _stub_captures(monkeypatch, failing=("ioi",))

    _data, results = cap.run_captures()

    report = {r.section: r.as_dict() for r in results}
    assert report["ioi"]["status"] == "failed"
    assert report["head_sweep"]["status"] == "ok"


def test_the_report_never_carries_a_payload(monkeypatch):
    """Otherwise every measurement is written twice."""
    _stub_captures(monkeypatch)

    _data, results = cap.run_captures()

    assert all("payload" not in r.as_dict() for r in results)


# ── Validation replaces default=str ─────────────────────────────────────

def test_a_payload_failing_validation_counts_as_a_failed_section():
    """A section that returns something unusable is not a measurement.

    To be exact about `default=str`: it had not corrupted the committed
    artifact -- the engine returns Python floats, so the fallback never fired.
    The risk it left was silent and latent: a capture returning a tensor would
    have produced `status: "ok"`, `provenance: "live"`, and a
    `"tensor(0.5)"` string where a number belonged. Validation is what makes
    that shape fail instead.
    """

    def wrong_shape():
        return {"n_templates": 3}  # missing every other required key

    result = cap.run_section("ioi", "ioi", wrong_shape)

    assert result.status == "failed"
    assert result.error_type == "ValueError"
    assert "missing required key" in result.error_message


def test_a_non_dict_payload_is_rejected():
    result = cap.run_section("ioi", "ioi", lambda: [1, 2, 3])

    assert result.status == "failed"
    assert result.error_type == "TypeError"


def test_an_empty_but_valid_section_still_counts_as_measured(monkeypatch):
    """Presence, not truthiness -- an empty head sweep is a real result.

    `REQUIRED_KEYS` is presence-based for the same reason
    `derive_meta_provenance` is: a section that swept every head and found none
    worth reporting still ran, and withholding `live` from it would make honest
    output look broken.
    """
    payload = {k: [] for k in cap.REQUIRED_KEYS["head_sweep"]}
    result = cap.run_section("head_sweep", "sweep", lambda: payload)

    assert result.status == "ok"

    data = {section: _payload(section) for section in cap.EXPECTED_SECTIONS}
    data["head_sweep"] = payload
    derived = cap.derive_meta_provenance(data)

    assert derived["provenance"] == LIVE
    assert derived["measured_sections"] == f"{len(cap.EXPECTED_SECTIONS)}/{len(cap.EXPECTED_SECTIONS)}"


def test_numpy_and_torch_values_are_converted_to_numbers():
    """The conversion `default=str` used to fake, done properly."""
    import numpy as np
    import torch

    converted = cap.to_json_native({
        "a": np.float32(0.5),
        "b": np.int64(7),
        "c": [np.float64(1.5), torch.tensor(2.5)],
        "d": np.arange(3),
    })

    assert converted == {"a": 0.5, "b": 7, "c": [1.5, 2.5], "d": [0, 1, 2]}
    assert all(
        isinstance(v, (int, float))
        for v in (converted["a"], converted["b"], *converted["c"])
    )
    # And it must survive a real dump with no fallback encoder.
    json.dumps(converted)


def test_an_unserialisable_type_raises_instead_of_being_stringified():
    class Opaque:
        def __repr__(self) -> str:
            return "Opaque(...)"

    with pytest.raises(TypeError) as excinfo:
        cap.to_json_native({"rows": [{"tensor": Opaque()}]})

    assert "unserialisable Opaque" in str(excinfo.value)
    # The path is what makes this diagnosable rather than merely loud.
    assert "$.rows[0].tensor" in str(excinfo.value)


def test_non_finite_floats_are_rejected():
    """`json.dump` writes bare `NaN`, which is not valid JSON.

    A downstream reader in another language rejects the whole file, so a single
    NaN measurement would take down a whole artifact.
    """
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError) as excinfo:
            cap.to_json_native({"clean_ld": bad})
        assert "non-finite" in str(excinfo.value)


def test_the_artifact_is_not_written_with_a_fallback_encoder():
    """Parsed, not grepped.

    `default=str` is the specific line this keeps out. It had not corrupted the
    committed artifact -- the engine returns floats -- but it converts
    unserialisable values by `str()` and reports nothing, so a capture that
    started returning a tensor would produce a clean `status: "ok"` and a
    `"tensor(...)"` string in a measurement field. The dumps are located by AST
    so the assertion cannot be satisfied by the word appearing in a comment.
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "dump":
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                offenders.append(f"line {node.lineno}")

    assert not offenders, (
        "json.dump is called with default= at " + ", ".join(offenders)
        + "; unserialisable values must raise in to_json_native instead of "
          "being stringified into something that parses as a measurement")


# ── main(): the artifact exists and the exit code is honest ─────────────

def test_main_writes_the_artifact_even_when_a_capture_fails(
        isolated, monkeypatch):
    _stub_captures(monkeypatch, failing=("ioi",))

    exit_code = cap.main()

    path = isolated / "results" / "capture.json"
    assert path.exists(), (
        "no artifact was written; the whole point is that an aborted run "
        "leaves a record rather than nothing")

    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["meta"]["provenance"] == UNAVAILABLE
    measured = len(cap.EXPECTED_SECTIONS) - 1
    assert artifact["meta"]["measured_sections"] == f"{measured}/{len(cap.EXPECTED_SECTIONS)}"
    assert exit_code != 0


def test_main_records_the_failure_reason_in_the_artifact(isolated, monkeypatch):
    _stub_captures(monkeypatch, failing=("ioi",))

    cap.main()

    artifact = json.loads(
        (isolated / "results" / "capture.json").read_text(encoding="utf-8"))
    assert "ioi" in artifact["meta"]["reason"]
    assert "RuntimeError" in artifact["meta"]["reason"]
    assert "exploded" in artifact["meta"]["reason"]


def test_main_carries_a_per_section_run_report(isolated, monkeypatch):
    _stub_captures(monkeypatch, failing=("ioi",))

    cap.main()

    artifact = json.loads(
        (isolated / "results" / "capture.json").read_text(encoding="utf-8"))
    sections = artifact["sections"]

    assert set(sections) == set(cap.EXPECTED_SECTIONS)
    assert sections["ioi"]["status"] == "failed"
    assert sections["ioi"]["error"]["type"] == "RuntimeError"
    assert sections["head_sweep"]["status"] == "ok"
    for record in sections.values():
        assert record["duration_s"] is not None


def test_main_exits_zero_on_a_complete_run(isolated, monkeypatch):
    _stub_captures(monkeypatch)

    exit_code = cap.main()

    artifact = json.loads(
        (isolated / "results" / "capture.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert artifact["meta"]["provenance"] == LIVE


def test_an_incomplete_run_replaces_the_completeness_claim(
        isolated, monkeypatch):
    """The blanket note claims every value was measured.

    With a section missing that sentence is false, so it has to be replaced
    rather than left sitting above a withheld label.
    """
    _stub_captures(monkeypatch, failing=("steering",))

    cap.main()

    artifact = json.loads(
        (isolated / "results" / "capture.json").read_text(encoding="utf-8"))
    assert "INCOMPLETE RUN" in artifact["meta"]["note"]
    assert "Every value is measured" not in artifact["meta"]["note"]


def test_an_earlier_artifact_is_never_left_half_written(isolated, monkeypatch):
    """The stale-file trap.

    With a non-atomic write, a crash or a full disk left a truncated
    `capture.json`: valid JSON, missing sections, indistinguishable from a run
    that legitimately measured less. `os.replace` means the destination is
    either the previous artifact or the complete new one.
    """
    path = isolated / "results" / "capture.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"meta": {"provenance": "live"}, "sentinel": true}\n',
                    encoding="utf-8")

    _stub_captures(monkeypatch, failing=("ioi",))
    cap.main()

    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert "sentinel" not in artifact, "the new artifact did not replace the old one"
    assert artifact["meta"]["provenance"] == UNAVAILABLE
    assert not list(path.parent.glob("*.tmp")), "a temp file was left behind"


def test_the_write_is_atomic_by_construction(tmp_path):
    """The temp-file-then-rename shape, asserted on the helper itself."""
    target = tmp_path / "nested" / "capture.json"

    cap.write_json_atomic(str(target), {"a": 1})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert not (tmp_path / "nested" / "capture.json.tmp").exists()


def test_a_failed_serialisation_leaves_the_previous_artifact_intact(tmp_path):
    """`allow_nan=False` and a raising encoder mean the dump can fail.

    If that failure happened after a truncate-and-write, the good artifact would
    already be gone. The temp file absorbs it.
    """
    target = tmp_path / "capture.json"
    target.write_text('{"previous": true}\n', encoding="utf-8")

    with pytest.raises((ValueError, TypeError)):
        cap.write_json_atomic(str(target), {"bad": float("nan")})

    assert json.loads(target.read_text(encoding="utf-8")) == {"previous": True}
    assert not list(tmp_path.glob("*.tmp")), (
        "a failed dump left a temp file next to the good artifact, where it "
        "reads as a newer result")


# ── Figures ─────────────────────────────────────────────────────────────

def test_render_figures_survives_a_missing_section(monkeypatch, tmp_path):
    """A `KeyError` here would abort the run *after* the captures finished.

    That is the same lost-artifact failure one layer down: hours of measurement
    discarded because a figure routine indexed a key that was not there.
    """
    monkeypatch.setattr(cap, "IMG_DIR", str(tmp_path))
    data = {"head_sweep": {"all_heads": [
        {"layer": 0, "head": 0, "delta": -1.0, "effect": 1.0}]}}

    written = cap.render_figures(data)

    assert isinstance(written, list)
    assert any("ioi-head-sweep" in path for path in written)


def test_render_figures_tolerates_a_completely_empty_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(cap, "IMG_DIR", str(tmp_path))

    assert cap.render_figures({}) == []


def test_render_figures_skips_a_grid_with_no_numeric_effect(monkeypatch, tmp_path):
    """Nothing plottable is no figure.

    `np.nanmax` over an all-NaN slice warns and returns nan, which then
    propagates into the colour limits. An empty figure reads as a measurement of
    nothing, which is the failure this section's whole design avoids.
    """
    monkeypatch.setattr(cap, "IMG_DIR", str(tmp_path))
    heads = [{"layer": 0, "head": 0, "delta": None, "effect": None}]

    assert cap.render_figures({"head_sweep": {"all_heads": heads}}) == []
    assert not (tmp_path / "ioi-head-sweep.png").exists()


def test_render_figures_does_not_hardcode_the_gpt2_head_grid(monkeypatch, tmp_path):
    """A 12x12 grid is a GPT-2-small assumption.

    With a different checkpoint the figure would either fail or plot into the
    wrong number of cells. Sized from the data instead. (Full removal of the
    hardcoded model assumptions across the script is a separate change; this
    asserts only that the figure cannot silently mislabel a sweep.)
    """
    monkeypatch.setattr(cap, "IMG_DIR", str(tmp_path))
    heads = [{"layer": l, "head": h, "delta": -0.1, "effect": 0.1}
             for l in range(4) for h in range(6)]

    cap.render_figures({"head_sweep": {"all_heads": heads}})

    assert (tmp_path / "ioi-head-sweep.png").exists()


# ── Structural ──────────────────────────────────────────────────────────

def test_main_calls_the_runner_rather_than_running_captures_directly():
    """Otherwise the exception boundary is decoration again.

    `main()` used to hold eight `data[...] = timed(...)` statements. The fix is
    only real while every capture goes through `run_captures`.
    """
    source = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    main_node = next(n for n in source.body
                     if isinstance(n, ast.FunctionDef) and n.name == "main")

    called = {
        node.func.id
        for node in ast.walk(main_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "run_captures" in called, (
        "main() no longer calls run_captures, so a capture can again raise "
        "past the artifact write")
    assert not (called & {"timed"}), (
        "main() still calls timed(), the unabsorbed runner")


def test_run_section_cannot_raise_on_the_ordinary_failure_paths():
    """Structural counterpart to the behaviour tests.

    The except clause must be `BaseException` and must assign a status, so a
    later edit that narrows it to `Exception` or drops the assignment fails
    here rather than in a run nobody is watching.
    """
    source = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    node = next(n for n in source.body
                if isinstance(n, ast.FunctionDef) and n.name == "run_section")

    handlers = [n for n in ast.walk(node) if isinstance(n, ast.ExceptHandler)]
    assert handlers, "run_section has no except handler"

    broad = [h for h in handlers
             if h.type is None or getattr(h.type, "id", None) == "BaseException"]
    assert broad, (
        "run_section only catches Exception; a KeyboardInterrupt mid-sweep "
        "would again discard the run report")
