"""A benchmark report must render, and must not invent a verdict.

The rule under test
-------------------
**An artifact that cannot be produced is a defect, and an artifact that asserts
more than was measured is worse.** Both applied to
`BenchmarkRunner.generate_artifact_package`, the function that writes the
published report.

Two defects, both in the same method, both found while making
`reproduce_everything.py` runnable.

1. The report crashed on every CPU run (P1 #7)
----------------------------------------------
The performance dashboard aggregated GPU profiler fields with no None handling:

    avg_tps  = sum(r.tokens_per_sec for r in suite.task_results) / len(...)
    avg_lat  = sum(r.mean_latency_ms for r in suite.task_results) / len(...)
    max_vram = max(r.peak_vram_mb   for r in suite.task_results)
    avg_gpu  = sum(r.gpu_util_pct   for r in suite.task_results) / len(...)
    total_flops = sum(r.flops       for r in suite.task_results)

Every one of those is `Optional[...] = None`, and the profiler leaves them None
whenever CUDA is unavailable -- which is every CPU run, including the setup this
repository documents and uses. Measured, before the fix:

    sum(tokens_per_sec)/len(...)   -> TypeError: unsupported operand type(s)
                                     for +: 'int' and 'NoneType'
    max(peak_vram_mb), 2 results   -> TypeError: '>' not supported between
                                     instances of 'NoneType' and 'NoneType'
    sum(...)/len([])               -> ZeroDivisionError: division by zero

`max()` needs two or more results on a model before it compares, so a
single-task suite masked it; `sum()` failed on the very first None.

The location is what made this serious. The crash was in the *artifact* path, not
the measurement path: the science completed, and then the report could not be
written. `generate_artifact_package` runs after `run_full_suite` has already done
the work.

Rendering an absent value as 0.0 was not an option either. `0.0` reads as
"measured, and the model needs no memory", which is a claim. So aggregation skips
absent values, tracks how many it saw, and renders `not measured` -- and a value
that genuinely is 0.0 still renders as 0.0.

2. The report invented a verdict (P1 #8)
----------------------------------------
One line, per row:

    status = "PASS" if r.fidelity_pct > 90 else "WARN"

It ignored provenance entirely. `is_fixture` and `mode` were both on the result,
and `--mode mock` therefore produced a report whose every row read PASS with a
tick. Verified on real output from this repository's own mock run -- nine tasks,
fidelities 97.58 to 99.81, every one of which the old renderer would have
labelled a passing measurement:

    ioi              | 99.78% | ... | PASS
    induction_heads  | 99.73% | ... | PASS
    greater_than     | 99.81% | ... | PASS
    logit_lens       | 98.72% | ... | PASS
    sae              | 99.79% | ... | PASS
    copy_task        | 97.58% | ... | PASS
    arithmetic       | 99.41% | ... | PASS
    factual_recall   | 99.49% | ... | PASS
    universality     | 99.60% | ... | PASS

After the fix every one of those rows reads `FIXTURE - not a measurement`.

`PASS`/`WARN` was also the wrong vocabulary. The validation suite distinguishes
MEASURED from PASS and NOT_RUN from ERROR; this renderer invented a third pair
that collapsed them. Verdicts now follow what happened, and the 90% threshold is a
named constant that never overrides provenance -- agreement with a published
figure is a reproduction check, not a verdict on a mechanism.

3. The entry point itself
-------------------------
`frontend/scripts/reproduce_everything.py` imported `backend.benchmarks.*`, which
does not exist; the package is `backend/benchmarking`. That import was at module
scope, so it failed loudly -- unlike the deleted orchestrator, nothing depended on
it working. Every method it calls exists in the real package.

Its `--mode` also defaulted to `mock`, while `run_full_suite` defaults to
`ExecutionMode.MOCK` too, so both layers agreed on fixtures before anything ran.
The default is now `reference`, and the completion banner is no longer
unconditional.

Its step 1 claimed to verify golden datasets by calling
`DatasetManager.list_datasets()` -- a method that does not exist. The
AttributeError was swallowed by a bare `except` that printed "Dataset Warning",
so the check never ran while the output read as though it had. `DatasetManager`
has no enumeration method at all; every one of its methods takes a specific
dataset name. Verification is a separate script and is now named as such.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pytest

from backend.benchmarking.benchmark_runner import (
    BenchmarkReport,
    BenchmarkRunner,
    ModelBenchmarkSuite,
    NOT_MEASURED,
    _cell,
    _interval,
    _max_of,
    _mean_of,
    _sum_of,
    _verdict,
)
from backend.benchmarking.benchmark_tasks import (
    BenchmarkResult,
    BenchmarkTask,
    ExecutionMode,
)


def _result(*, mode=ExecutionMode.REFERENCE, fidelity=95.0, is_fixture=False,
            **overrides) -> BenchmarkResult:
    """A result with every optional field absent, as a CPU run produces."""
    base: dict[str, Any] = dict(
        task_id=BenchmarkTask.IOI,
        model_id="gpt2",
        backend="hf",
        mode=mode,
        primary_score=fidelity,
        reference_score=88.0,
        fidelity_pct=fidelity,
        runtime_s=1.0,
        peak_memory_mb=None,
        confidence_interval_low=None,
        confidence_interval_high=None,
        is_fixture=is_fixture,
        backend_effective="transformer_lens",
        run_id="r1",
    )
    base.update(overrides)
    return BenchmarkResult(**base)


def _suite(results: List[BenchmarkResult]) -> ModelBenchmarkSuite:
    return ModelBenchmarkSuite(
        model_id="gpt2",
        backend="hf",
        mode=results[0].mode if results else ExecutionMode.REFERENCE,
        n_layers=12,
        n_params_b=0.124,
        task_results=results,
        coverage_pct=100.0,
        mean_fidelity_pct=95.0,
        run_id="r1",
    )


def _package(tmp_path: Path, results: List[BenchmarkResult]) -> str:
    report = BenchmarkReport(
        suites=[_suite(results)],
        overall_coverage_pct=100.0,
        overall_fidelity_pct=95.0,
        models_tested=1,
        tasks_tested=len(results),
    )
    runner = BenchmarkRunner()
    return runner.generate_artifact_package(
        report, output_dir=str(tmp_path / "report"))


# ── P1 #7: the report must render on a CPU run ─────────────────────────────

def test_report_renders_with_every_gpu_field_absent(tmp_path):
    """The regression test for the crash.

    One result, so `max()` never compares -- `sum()` alone is enough to fail.
    """
    out = _package(tmp_path, [_result()])
    markdown = (Path(out) / "report.md").read_text(encoding="utf-8")
    assert "Performance Dashboard" in markdown
    assert NOT_MEASURED in markdown


def test_report_renders_with_two_results_on_one_model(tmp_path):
    """Two or more results is where `max()` over Nones raised TypeError."""
    out = _package(tmp_path, [_result(), _result(fidelity=88.0)])
    markdown = (Path(out) / "report.md").read_text(encoding="utf-8")
    assert markdown.count(NOT_MEASURED) >= 5


def test_report_renders_with_an_empty_suite(tmp_path):
    """`len(...) == 0` raised ZeroDivisionError."""
    out = _package(tmp_path, [])
    assert (Path(out) / "report.md").exists()


def test_report_renders_when_some_fields_are_present(tmp_path):
    """A mixed run must not drop the values it does have."""
    out = _package(tmp_path, [
        _result(tokens_per_sec=100.0, mean_latency_ms=10.0,
                peak_vram_mb=2048.0, gpu_util_pct=55.0, flops=1.5e9),
        _result(fidelity=88.0),  # nothing measured
    ])
    row = next(
        line for line in (Path(out) / "report.md").read_text(
            encoding="utf-8").splitlines()
        if line.startswith("| gpt2 |")
    )
    assert "100.0" in row, row
    assert "2048.0" in row, row
    assert "55.0%" in row, row
    assert NOT_MEASURED not in row, (
        "a measured field must not be reported as unmeasured")


def test_a_measured_zero_is_not_reported_as_unmeasured(tmp_path):
    """0.0 and absent are different facts, and the renderer must keep them apart."""
    out = _package(tmp_path, [
        _result(tokens_per_sec=0.0, mean_latency_ms=0.0,
                peak_vram_mb=0.0, gpu_util_pct=0.0, flops=0.0),
    ])
    row = next(
        line for line in (Path(out) / "report.md").read_text(
            encoding="utf-8").splitlines()
        if line.startswith("| gpt2 |")
    )
    assert NOT_MEASURED not in row, (
        f"a measured 0.0 rendered as {NOT_MEASURED!r}; the two are not the same: "
        f"{row}")
    assert row.count("0.0") >= 5, row


def test_absent_and_measured_zero_render_differently(tmp_path):
    """Same field, two runs: one absent, one genuinely zero."""
    absent = next(
        line for line in (Path(_package(tmp_path / "a", [_result()])) / "report.md")
        .read_text(encoding="utf-8").splitlines()
        if line.startswith("| gpt2 |")
    )
    zero = next(
        line for line in (
            Path(_package(tmp_path / "z", [
                _result(tokens_per_sec=0.0, mean_latency_ms=0.0,
                        peak_vram_mb=0.0, gpu_util_pct=0.0, flops=0.0),
            ])) / "report.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("| gpt2 |")
    )
    assert absent != zero
    assert NOT_MEASURED in absent
    assert NOT_MEASURED not in zero


def test_reducers_skip_absent_and_non_numeric_values():
    class Row:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    rows = [Row(a=1.0, b=None), Row(a=3.0, b="not-a-number"), Row(a=None, b=2.0)]

    mean_a, n_a = _mean_of(rows, "a")
    assert mean_a == 2.0 and n_a == 2
    total_b, n_b = _sum_of(rows, "b")
    assert total_b == 2.0 and n_b == 1, "a string must be skipped, not raise"
    max_a, n_max = _max_of(rows, "a")
    assert max_a == 3.0 and n_max == 2


def test_a_reducer_over_nothing_is_none_not_zero():
    assert _mean_of([], "a") == (None, 0)
    assert _max_of([], "a") == (None, 0)
    assert _sum_of([], "a") == (None, 0)


def test_cell_never_formats_a_percentage_onto_the_unmeasured_marker():
    """An early version appended `%` outside the formatter: "not measured%"."""
    assert _cell(None, ".1f", 0) == NOT_MEASURED
    assert "%" not in _cell(None, ".1f", 0)
    assert _cell(0.0, ".1f", 1) == "0.0"


# ── P1 #8: the verdict must follow evidence ────────────────────────────────

@pytest.mark.parametrize("fidelity", [99.78, 97.58, 91.0, 90.0, 55.0])
def test_a_fixture_never_gets_a_measured_verdict(fidelity):
    """Every one of these rendered PASS before the fix."""
    result = _result(mode=ExecutionMode.MOCK, fidelity=fidelity, is_fixture=True)
    verdict = _verdict(result)
    assert verdict.startswith("FIXTURE"), verdict
    assert "PASS" not in verdict.upper().replace("NOT A MEASUREMENT", "")


def test_mock_mode_alone_is_enough_to_withhold_the_verdict():
    """The historical records carry `mode` and predate `is_fixture`."""
    result = _result(mode=ExecutionMode.MOCK, fidelity=99.78, is_fixture=False)
    assert result.is_fixture is False, "precondition: the newer field is absent"
    assert _verdict(result).startswith("FIXTURE")


def test_a_real_measurement_above_the_threshold_says_so():
    verdict = _verdict(_result(fidelity=95.0))
    assert verdict.startswith("MEASURED")
    assert "reproduces published" in verdict
    assert "90" in verdict, "the deciding threshold should be visible"


def test_a_real_measurement_below_the_threshold_is_not_called_a_pass():
    verdict = _verdict(_result(fidelity=55.0))
    assert verdict.startswith("MEASURED")
    assert "below reference" in verdict


def test_no_verdict_uses_the_word_pass():
    """PASS is not in this project's scientific vocabulary."""
    for result in (_result(fidelity=99.0), _result(fidelity=10.0),
                   _result(mode=ExecutionMode.MOCK, fidelity=99.0,
                           is_fixture=True)):
        verdict = _verdict(result)
        assert not verdict.startswith("PASS"), verdict
        assert " WARN" not in verdict, verdict


def test_provenance_beats_the_threshold_in_both_directions():
    """A fixture above threshold and a fixture below it are both fixtures."""
    high = _verdict(_result(mode=ExecutionMode.MOCK, fidelity=99.9, is_fixture=True))
    low = _verdict(_result(mode=ExecutionMode.MOCK, fidelity=1.0, is_fixture=True))
    assert high.startswith("FIXTURE") and low.startswith("FIXTURE")


# ── confidence intervals ───────────────────────────────────────────────────

def test_absent_interval_renders_as_not_measured_not_none():
    """It used to interpolate `[None, None]` straight into the report."""
    assert _interval(_result()) == NOT_MEASURED


def test_an_interval_that_was_not_derived_says_so():
    text = _interval(_result(confidence_interval_low=80.0,
                             confidence_interval_high=90.0,
                             confidence_interval_derived=False))
    assert "not derived from samples" in text


def test_a_derived_interval_names_its_method():
    text = _interval(_result(confidence_interval_low=80.0,
                             confidence_interval_high=90.0,
                             confidence_interval_derived=True,
                             confidence_interval_method="bootstrap"))
    assert "[80.0, 90.0]" in text and "bootstrap" in text


# ── the artifact as a whole ────────────────────────────────────────────────

def test_a_full_mock_report_marks_every_row_as_a_fixture(tmp_path):
    """The real regression, on the shape this repository actually produces."""
    rows = [
        _result(mode=ExecutionMode.MOCK, fidelity=f, is_fixture=True,
                task_id=t)
        for t, f in ((BenchmarkTask.IOI, 99.78),
                     (BenchmarkTask.SAE, 99.79),
                     (BenchmarkTask.ARITHMETIC, 99.41))
    ]
    markdown = (Path(_package(tmp_path, rows)) / "report.md").read_text(
        encoding="utf-8")

    body = [ln for ln in markdown.splitlines() if ln.startswith("| ioi")
            or ln.startswith("| sae") or ln.startswith("| arithmetic")]
    assert body, "expected per-task rows in the report"
    for line in body:
        assert "FIXTURE" in line, line
        assert "PASS" not in line, line


def test_the_json_artifact_records_provenance(tmp_path):
    out = _package(tmp_path, [_result(mode=ExecutionMode.MOCK, fidelity=99.0,
                                      is_fixture=True)])
    payload = json.loads((Path(out) / "report.json").read_text(encoding="utf-8"))
    blob = json.dumps(payload)
    assert "is_fixture" in blob or "FIXTURE" in blob.upper()


# ── the entry point ────────────────────────────────────────────────────────

def test_reproduce_everything_imports_and_defaults_to_real_weights():
    import frontend.scripts.reproduce_everything as script

    source = Path(script.__file__).read_text(encoding="utf-8")
    assert 'default=ExecutionMode.REFERENCE.value' in source, (
        "the default mode must be real weights, not mock")
    assert 'default="mock"' not in source


def test_reproduce_everything_no_longer_claims_to_verify_datasets():
    import frontend.scripts.reproduce_everything as script

    source = Path(script.__file__).read_text(encoding="utf-8")
    assert "list_datasets" not in source.replace(
        "while calling DatasetManager.list_datasets(), a method", ""), (
        "list_datasets does not exist; the call always raised and the bare "
        "except made the output read as though the check had run")
    assert "verify_golden_datasets.py" in source, (
        "dataset verification is a separate script and should be named")


def test_reproduce_everything_partitions_measured_from_fixtures():
    import frontend.scripts.reproduce_everything as script

    report = BenchmarkReport(suites=[_suite([
        _result(fidelity=95.0),
        _result(mode=ExecutionMode.MOCK, fidelity=99.0, is_fixture=True),
        _result(mode=ExecutionMode.MOCK, fidelity=98.0, is_fixture=False),
    ])])
    measured, fixtures = script._partition(report)
    assert len(measured) == 1
    assert len(fixtures) == 2, "mock mode must count as a fixture"


def test_reproduce_everything_exits_nonzero_on_a_fixture_only_run(monkeypatch,
                                                                  capsys, tmp_path):
    import frontend.scripts.reproduce_everything as script

    class FixtureRunner:
        def run_full_suite(self, mode=None, tier=1):
            return BenchmarkReport(suites=[_suite([
                _result(mode=ExecutionMode.MOCK, fidelity=99.0, is_fixture=True),
            ])], overall_coverage_pct=100.0, overall_fidelity_pct=99.0,
                models_tested=1, tasks_tested=1)

        def generate_artifact_package(self, report, output_dir="x"):
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            for name in ("report.md", "report.json", "report.csv",
                         "raw_experiment_data.json"):
                (target / name).write_text("{}", encoding="utf-8")
            return str(target)

    monkeypatch.setattr(script, "BenchmarkRunner", FixtureRunner)
    code = script.main(["--mode", "mock", "--output", str(tmp_path / "out")])
    out = capsys.readouterr().out

    assert code == 1
    assert "NOTHING WAS MEASURED" in out
    assert "Do not cite" in out


def test_reproduce_everything_exits_zero_when_something_measured(monkeypatch,
                                                                  capsys, tmp_path):
    import frontend.scripts.reproduce_everything as script

    class RealRunner:
        def run_full_suite(self, mode=None, tier=1):
            return BenchmarkReport(suites=[_suite([
                _result(fidelity=95.0),
                _result(mode=ExecutionMode.MOCK, fidelity=99.0, is_fixture=True),
            ])], overall_coverage_pct=100.0, overall_fidelity_pct=97.0,
                models_tested=1, tasks_tested=2)

        def generate_artifact_package(self, report, output_dir="x"):
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            for name in ("report.md", "report.json", "report.csv",
                         "raw_experiment_data.json"):
                (target / name).write_text("{}", encoding="utf-8")
            return str(target)

    monkeypatch.setattr(script, "BenchmarkRunner", RealRunner)
    code = script.main(["--mode", "reference", "--output", str(tmp_path / "out")])
    out = capsys.readouterr().out

    assert code == 0
    assert "1 result(s) were measured" in out
    assert "Fixtures          : 1" in out


# ── the repo-wide guard, now empty ─────────────────────────────────────────

def test_no_unresolvable_imports_remain():
    """`backend.benchmarks` was the last one. The guard is now a hard failure."""
    from tests.pytest.test_entry_points import (
        EXPECTED_GUARDED_OPTIONAL,
        EXPECTED_UNRESOLVED,
        _unresolved_imports,
    )

    assert EXPECTED_UNRESOLVED == {}, (
        f"tolerated unresolvable imports remain: {EXPECTED_UNRESOLVED}")

    unexpected = [
        f"{f}:{ln} -> {mod}"
        for f, ln, mod in _unresolved_imports()
        if f not in EXPECTED_GUARDED_OPTIONAL
    ]
    assert not unexpected, unexpected


def test_no_module_imports_the_benchmarks_package():
    """Checked on the AST, via the shared resolver.

    The first version of this test searched file text for "backend.benchmarks"
    and matched its own docstring, the docstring of `reproduce_everything.py`,
    the comment in `kg_integrator.py`, and the two test files that mention the
    name. That is the fifth time in this effort that a guard has matched the
    *explanation* of a fix rather than the defect. An unresolved import is a
    property of the import graph, so it is tested there.
    """
    from tests.pytest.test_entry_points import _unresolved_imports

    offenders = [
        f"{f}:{ln} -> {mod}"
        for f, ln, mod in _unresolved_imports()
        if mod.startswith("backend.benchmarks")
    ]
    assert not offenders, offenders
