"""A consumer must honour the producer's verdict about comparability.

The rule under test
-------------------
**A benchmark that was measured but could not be compared is neither a pass nor a
regression, and no downstream component may issue a verdict against a baseline
the pipeline has already declared incomparable.**

This is the audit's P1 #8: "you have two competing concepts — scientific status
versus report rendering." The reality was worse than two vocabularies. There were
nine, and in one place two components of a *single pipeline* disagreed about
whether a comparison was meaningful, with the disagreeing one driving the
platform's health score and firing its alerts.

The defect, measured on live weights before the fix
---------------------------------------------------
`ValidationBenchmarkScheduler` correctly declined to score two of its five
benchmarks. IOI came back `status=MEASURED`, `baseline_is_comparable=False`, with
the reason "the published baseline is not the same measurement; the published
circuit scores 0.6695 through this harness." Induction-heads likewise.

`RegressionDetector.analyze_results` then read `current_fidelity` and ignored
`status` and `baseline_is_comparable` entirely:

    reg_fid_01  bm_ioi  Fidelity  change=-17.73%   severity=HIGH
    reg_fid_02  bm_ind  Fidelity  change=-28.78%   severity=HIGH
    reg_rt_01   bm_ioi  Runtime   change=+6391.35%  severity=MEDIUM
    reg_rt_02   bm_ind  Runtime   change=+716.40%   severity=MEDIUM

Those four events drove `platform_health_score` to 80 and fired HIGH-severity
alerts. So the platform reported a measurable regression on the two numbers this
project is most careful about, from comparisons it had explicitly ruled invalid.

The runtime figures deserve their own note: 1200 ms is a published baseline from
the authors' hardware, and 77896 ms is this machine. Comparing them yields
"+6391%", which is a fact about two different computers, not about this platform.

The two false answers
---------------------
Fixing the detector was necessary but not sufficient, because it exposed the
converse lie. `platform_health_score` was `100 - (regressions * 5)`, evaluated
unconditionally, so zero events became a perfect **100.0** — for a suite in which
nothing had been checked. Before the fix the same formula returned 80.0 from
invalid comparisons; after it, 100.0 from no comparisons at all. The score is now
`None` with a reason, matching how `overall_pass_rate` and
`reproducibility_score` in the same dataclass already handle the unmeasurable.

And `AlertEngine.generate_alerts` closed on an empty event list with:

    title="Platform Health Optimal"
    description="All golden benchmarks passing baseline fidelity and latency specs."

which is the lie stated outright. Every regression alert also carried a fixed
invented diagnosis -- "Transformers package upgrade from 4.38.2 to 4.39.0 altered
activation caching", with the remedy "pin transformers==4.38.2" -- for every
benchmark and every metric, naming a package version that is not installed (4.57.6)
and naming a cause no code in the pipeline observes.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from backend.validation.alert_engine import AlertEngine
from backend.validation.benchmark_scheduler import (
    GoldenBenchmarkResult,
    ValidationBenchmarkScheduler,
)
from backend.validation.benchmark_status import (
    BENCHMARK_STATUSES,
    ERROR,
    FIXTURE,
    MEASURED,
    NOT_RUN,
    PASS,
    REGRESSION,
    describe,
    is_comparable,
    is_measured,
    is_unmeasured,
    is_verdict,
)
from backend.validation.health_dashboard import (
    ContinuousHealthReport,
    HealthDashboardEngine,
)
from backend.validation.regression_detector import RegressionDetector

ROOT = Path(__file__).resolve().parents[2]


def _result(**over: Any) -> GoldenBenchmarkResult:
    base: Dict[str, Any] = dict(
        benchmark_id="bm_x",
        name="Synthetic",
        target_model="GPT2-S",
        published_baseline_fidelity=0.88,
        current_fidelity=0.70,
        published_baseline_runtime_ms=1200.0,
        current_runtime_ms=900.0,
        published_baseline_vram_gb=4.2,
        current_vram_gb=None,
        status=MEASURED,
        measured=True,
        reason="synthetic",
        baseline_is_comparable=True,
        runtime_is_comparable=True,
    )
    base.update(over)
    return GoldenBenchmarkResult(**base)


# ── the vocabulary ─────────────────────────────────────────────────────────

def test_the_vocabulary_separates_measured_from_scored():
    """The distinction the old comment could not express."""
    assert MEASURED in BENCHMARK_STATUSES
    assert is_measured(MEASURED) and is_measured(PASS) and is_measured(REGRESSION)
    assert not is_measured(NOT_RUN) and not is_measured(ERROR)
    assert not is_measured(FIXTURE), "a fixture is not a measurement"

    assert is_verdict(PASS) and is_verdict(REGRESSION)
    assert not is_verdict(MEASURED), (
        "MEASURED is the state that exists precisely because it is NOT a verdict")
    assert not is_verdict(NOT_RUN)

    assert is_unmeasured(NOT_RUN) and is_unmeasured(ERROR)


def test_comparability_is_fail_closed():
    """`None` means *not established*, and an unestablished comparison is not one."""
    assert is_comparable(True)
    assert not is_comparable(False)
    assert not is_comparable(None), (
        "an unestablished comparison must not be treated as a comparison")


def test_describe_is_stable_and_counts_nothing_as_unknown_silently():
    assert describe([]) == "none"
    assert describe([NOT_RUN, NOT_RUN, PASS]) == "NOT_RUN=2, PASS=1"
    assert describe([None]) == "UNKNOWN=1"


# ── the detector must honour comparability ─────────────────────────────────

def test_the_detector_scores_a_comparable_baseline():
    """The capability is retained -- this is not "never detect a regression"."""
    events = RegressionDetector().analyze_results([_result()])
    assert len(events) == 1
    assert events[0].metric_type == "Fidelity"
    assert events[0].benchmark_id == "bm_x"


def test_the_detector_declines_an_incomparable_fidelity_baseline():
    """The core fix. -20% against an incomparable baseline is not a regression."""
    result = _result(baseline_is_comparable=False, status=MEASURED)
    assert RegressionDetector().analyze_results([result]) == []


def test_the_detector_declines_when_comparability_was_never_established():
    """`None` -- the default on the dataclass -- must not produce a verdict."""
    result = _result(baseline_is_comparable=None)
    assert RegressionDetector().analyze_results([result]) == []


def test_the_detector_declines_an_incomparable_runtime_baseline():
    """1200 ms from a paper vs 77896 ms here is a fact about two computers."""
    result = _result(current_runtime_ms=77896.15,
                     runtime_is_comparable=False)
    events = RegressionDetector().analyze_results([result])
    assert [e for e in events if e.metric_type == "Runtime Latency"] == []


def test_a_not_run_benchmark_produces_no_events():
    result = _result(current_fidelity=None, current_runtime_ms=None,
                     status=NOT_RUN, measured=False,
                     baseline_is_comparable=None, runtime_is_comparable=None)
    assert RegressionDetector().analyze_results([result]) == []


def test_declined_comparisons_are_reported_with_reasons():
    """Silence has to be legible, or it reads as health."""
    detector = RegressionDetector()
    results = [
        _result(benchmark_id="bm_ioi", baseline_is_comparable=False,
                runtime_is_comparable=False),
        # Must actually regress to produce an event. 0.95 against a 0.88
        # baseline is +7.9%, which is an improvement, so the first draft of this
        # test asserted an event that correctly did not exist.
        _result(benchmark_id="bm_ok", current_fidelity=0.70),
    ]
    skipped = detector.incomparable_results(results)

    assert [s["benchmark_id"] for s in skipped] == ["bm_ioi"]
    reasons = " ".join(skipped[0]["declined_comparisons"])
    assert "not the same measurement" in reasons
    assert "other hardware" in reasons

    events = detector.analyze_results(results)
    assert [e.benchmark_id for e in events] == ["bm_ok"], (
        "the comparable benchmark must still be scored -- this fix removes the "
        "invalid comparison, not the capability")


def test_thresholds_are_configurable_and_named():
    detector = RegressionDetector(fidelity_drop_pct=-50.0,
                                  runtime_slowdown_pct=1000.0)
    assert detector.analyze_results([_result()]) == [], (
        "a -20% drop must not fire when the threshold is -50%")


# ── the score must not be a lie in either direction ────────────────────────

@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    """A dashboard whose graph store is isolated to tmp_path."""
    import backend.validation.health_dashboard as hd
    from backend.knowledge_graph.graph_store import GraphStore

    original = hd.GraphStore
    monkeypatch.setattr(
        hd, "GraphStore",
        lambda *a, **k: original(storage_path=str(tmp_path / "kg.json")))
    return HealthDashboardEngine()


def test_health_score_is_none_when_nothing_was_comparable(dashboard, monkeypatch):
    """The false 100. Zero events must not read as a perfect score."""
    monkeypatch.setattr(dashboard.scheduler, "execute_validation_suite",
                        lambda: [_result(baseline_is_comparable=False,
                                         runtime_is_comparable=False,
                                         status=MEASURED)])

    report = dashboard.run_continuous_validation()

    assert report.platform_health_score is None
    assert report.health_score_measured is False
    assert report.health_score_reason
    assert "comparable baseline" in report.health_score_reason
    assert report.to_dict()["platform_health_score"] is None


def test_health_score_is_derived_when_something_was_comparable(
        dashboard, monkeypatch):
    # status=PASS, matching what the scheduler assigns to a comparable benchmark
    # that did not regress. The first draft left status=MEASURED alongside
    # baseline_is_comparable=True -- a state the scheduler cannot produce -- and
    # then asserted a score.
    monkeypatch.setattr(dashboard.scheduler, "execute_validation_suite",
                        lambda: [_result(current_fidelity=0.95, status=PASS)])

    report = dashboard.run_continuous_validation()

    assert report.platform_health_score == 100.0
    assert report.health_score_measured is True
    assert report.overall_pass_rate == 100.0
    assert report.passed_benchmarks == 1


def test_health_score_does_not_depend_on_status_and_comparability_agreeing(
        dashboard, monkeypatch):
    """The precondition for the formula is comparability, not a scored verdict.

    Gating on `scorable_bm` (PASS + REGRESSION) made the answer depend on two
    fields agreeing. This asserts the gate asks the right question.
    """
    monkeypatch.setattr(dashboard.scheduler, "execute_validation_suite",
                        lambda: [_result(current_fidelity=0.95, status=MEASURED)])

    report = dashboard.run_continuous_validation()

    assert report.platform_health_score == 100.0, (
        "the comparison was possible, so a score is derivable even though this "
        "contradictory status did not produce a PASS")
    assert report.health_score_measured is True


def test_health_score_is_penalised_by_a_real_regression(dashboard, monkeypatch):
    monkeypatch.setattr(dashboard.scheduler, "execute_validation_suite",
                        lambda: [_result(current_fidelity=0.70,
                                         status=REGRESSION)])

    report = dashboard.run_continuous_validation()

    assert report.platform_health_score == 95.0, (
        "one regression costs 5 points, and that must still happen")
    assert len(report.regression_events) == 1
    assert report.failed_benchmarks == 1
    assert report.overall_pass_rate == 0.0


def test_pass_rate_is_none_not_zero_when_nothing_is_scorable(dashboard,
                                                             monkeypatch):
    monkeypatch.setattr(dashboard.scheduler, "execute_validation_suite",
                        lambda: [_result(baseline_is_comparable=False,
                                         runtime_is_comparable=False,
                                         status=MEASURED)])

    report = dashboard.run_continuous_validation()

    assert report.overall_pass_rate is None
    assert report.pass_rate_measured is False
    assert report.passed_benchmarks == 0 and report.failed_benchmarks == 0


def test_to_dict_round_trips_with_a_none_health_score(dashboard, monkeypatch):
    # Both flags off. The first draft set only `baseline_is_comparable=False` and
    # left `runtime_is_comparable` at the helper's default of True -- so a runtime
    # comparison genuinely was possible, `comparable_any` was correctly True, and
    # the score was correctly derived. The helper's default, not the code, was
    # what was wrong.
    monkeypatch.setattr(dashboard.scheduler, "execute_validation_suite",
                        lambda: [_result(baseline_is_comparable=False,
                                         runtime_is_comparable=False,
                                         status=MEASURED)])

    payload = dashboard.run_continuous_validation().to_dict()

    assert payload["platform_health_score"] is None
    assert payload["health_score_measured"] is False
    assert json.dumps(payload), "the report must still serialise"


def test_health_score_is_derived_when_only_runtime_is_comparable(
        dashboard, monkeypatch):
    """Comparability is per-metric, so one comparable axis is enough."""
    monkeypatch.setattr(dashboard.scheduler, "execute_validation_suite",
                        lambda: [_result(baseline_is_comparable=False,
                                         runtime_is_comparable=True,
                                         status=MEASURED)])

    report = dashboard.run_continuous_validation()

    assert report.platform_health_score == 100.0
    assert report.health_score_measured is True
    assert report.declined_comparisons, (
        "the fidelity comparison is still declined and must be visible")


def test_two_validation_runs_get_distinct_graph_nodes(dashboard):
    """`val_run_{int(time.time())}` collided inside one clock tick."""
    import backend.validation.health_dashboard as hd

    dashboard.run_continuous_validation()
    dashboard.run_continuous_validation()

    store = dashboard.graph_store
    run_nodes = [n for n in store.nodes.values()
                 if str(getattr(n, "node_id", "")).startswith("val_run_")]
    assert len(run_nodes) == 2, (
        f"two runs produced {len(run_nodes)} nodes; the second overwrote the "
        f"first")


# ── alerts must not invent a verdict or a cause ────────────────────────────

def test_no_events_with_nothing_scorable_is_not_reported_as_optimal():
    """The lie, stated outright."""
    alerts = AlertEngine().generate_alerts([], scorable_count=0, declined_count=2)

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.severity == "WARNING"
    assert "Optimal" not in alert.title
    assert "passing baseline" not in alert.description
    assert "not" in alert.description and "clean result" in alert.description


def test_no_events_with_something_scorable_is_an_info_not_a_warning():
    alerts = AlertEngine().generate_alerts([], scorable_count=3, declined_count=0)

    assert alerts[0].severity == "INFO"
    assert alerts[0].title == "No regressions detected"
    assert "3 benchmark comparison(s)" in alerts[0].description


def test_no_alert_ever_claims_optimal_health():
    """The literal string must not come back, whatever the inputs."""
    for scorable, declined in ((0, 0), (0, 5), (3, 0), (5, 2)):
        blob = json.dumps([
            a.__dict__ for a in
            AlertEngine().generate_alerts([], scorable_count=scorable,
                                          declined_count=declined)
        ])
        assert "Platform Health Optimal" not in blob
        assert "Stable environment state" not in blob


def test_a_regression_alert_does_not_invent_a_cause():
    """It used to blame a transformers upgrade for every metric, every time."""
    from backend.validation.regression_detector import RegressionEvent

    event = RegressionEvent(
        event_id="reg_fid_01", benchmark_id="bm_ioi", benchmark_name="IOI",
        metric_type="Fidelity", baseline_value=0.88, current_value=0.70,
        percentage_change=-20.5, severity="HIGH",
    )
    alert = AlertEngine().generate_alerts([event], scorable_count=1)[0]

    assert alert.severity == "CRITICAL"
    assert "Not determined" in alert.lik_cause_placeholder if False else True
    assert "Not determined" in alert.likely_cause
    assert "4.38.2" not in json.dumps(alert.__dict__)
    assert "4.39.0" not in json.dumps(alert.__dict__)
    assert "activation caching" not in json.dumps(alert.__dict__)


def test_remediation_is_derived_from_the_metric_not_fixed_text():
    from backend.validation.regression_detector import RegressionEvent

    def alert_for(metric_type):
        event = RegressionEvent(
            event_id="reg_1", benchmark_id="bm", benchmark_name="B",
            metric_type=metric_type, baseline_value=1.0, current_value=2.0,
            percentage_change=100.0, severity="MEDIUM",
        )
        return AlertEngine().generate_alerts([event], scorable_count=1)[0]

    fidelity = alert_for("Fidelity")
    runtime = alert_for("Runtime Latency")

    assert fidelity.recommended_action != runtime.recommended_action
    assert "baseline_is_comparable" in fidelity.recommended_action
    assert "golden record" in runtime.recommended_action


def test_the_hardcoded_diagnosis_is_gone_from_the_source():
    """The strings must not survive in executable code.

    Checked against `executable_source`, which strips docstrings. The first draft
    read the raw file and matched the docstring in `generate_alerts` that quotes
    the removed diagnosis in order to explain why it was removed -- the sixth time
    in this effort a guard has matched the explanation of a fix instead of the
    defect.
    """
    from tests.pytest.source_assert import executable_source

    body = executable_source(
        (ROOT / "backend" / "validation" / "alert_engine.py")
        .read_text(encoding="utf-8"))
    for phrase in ("4.38.2", "4.39.0", "activation caching",
                   "Platform Health Optimal", "Stable environment state",
                   "pin transformers"):
        assert phrase not in body, phrase


# ── the contract must be declared, not restated per field ──────────────────

def test_the_scheduler_status_contract_names_only_real_statuses():
    """It read `# PASS, REGRESSION, WARNING, NOT_RUN`.

    WARNING was never produced by any code path and MEASURED -- which the
    scheduler assigns -- was not listed. A reader trusting it would not have known
    MEASURED existed.

    Read from executable source, because the replacement comment quotes the old
    one in order to explain what was wrong with it.
    """
    from tests.pytest.source_assert import executable_source

    body = executable_source(
        (ROOT / "backend" / "validation" / "benchmark_scheduler.py")
        .read_text(encoding="utf-8"))
    assert "WARNING" not in body, (
        "WARNING is not a status this scheduler can produce, and it must not "
        "appear in its contract")

    # Every status the scheduler can assign must be in the shared vocabulary.
    tree = ast.parse(
        (ROOT / "backend" / "validation" / "benchmark_scheduler.py")
        .read_text(encoding="utf-8"))
    assigned = {
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in {"PASS", "REGRESSION", "MEASURED", "NOT_RUN",
                           "ERROR", "FIXTURE", "WARNING"}
    }
    assert assigned <= BENCHMARK_STATUSES, assigned
    assert "WARNING" not in assigned


def test_the_vocabulary_lives_in_one_module():
    """The scheduler, detector and dashboard must not each restate it."""
    status = ROOT / "backend" / "validation" / "benchmark_status.py"
    assert status.exists()

    consumers = [
        "benchmark_scheduler.py", "regression_detector.py",
        "health_dashboard.py",
    ]
    for name in consumers:
        source = (ROOT / "backend" / "validation" / name).read_text(
            encoding="utf-8")
        assert "from .benchmark_status import" in source, (
            f"{name} should import the shared vocabulary rather than re-spell it")


def test_the_scheduler_imports_rather_than_re_spelling_the_statuses():
    """Its assignments go through the imported names."""
    from tests.pytest.source_assert import executable_source

    body = executable_source(
        (ROOT / "backend" / "validation" / "benchmark_scheduler.py")
        .read_text(encoding="utf-8"))
    for literal in ('status="PASS"', 'status="REGRESSION"',
                    'status="MEASURED"', 'status="NOT_RUN"'):
        assert literal not in body, (
            f"{literal} should be `status=<imported constant>` so the vocabulary "
            f"has exactly one definition")


def test_health_dashboard_does_not_compare_statuses_as_bare_literals():
    """It used `r.status == "PASS"` etc. A fifth status would land in no bucket."""
    source = (ROOT / "backend" / "validation" / "health_dashboard.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for operand in node.comparators:
            if (isinstance(operand, ast.Constant)
                    and isinstance(operand.value, str)
                    and operand.value in BENCHMARK_STATUSES):
                offenders.append(operand.value)
    assert not offenders, (
        f"statuses compared as bare literals: {offenders}; import them from "
        f"benchmark_status instead")
