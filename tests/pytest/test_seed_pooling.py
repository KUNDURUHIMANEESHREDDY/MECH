"""Pooling across seeds: the interval must rest on seeds, not on prompt count.

Why these tests exist
--------------------
A Wilson interval over per-prompt outcomes bounds precision *within* one draw.
That is the right shape for the wrong question when the measurement was repeated
once: prompts generated under one seed share that seed's name set and RNG
stream, so they are not independent, and pooling 100 of them reports a precision
the design never had.

The fix is to make the seed the unit of variation -- a Student-t interval on the
seed-level values -- and to state `n_seeds` wherever a score is reported, so a
reader can tell one draw from five.

These tests pin the fail-closed behaviour specifically. The temptation in a
statistics helper is to make every field return a number, because a number is
easier to serialise and reads as a result. Every one of these cases must instead
return None plus a reason:

  * one seed           -- no variance, so no seed-level interval exists
  * zero usable seeds  -- nothing to pool
  * all seeds identical -- a zero-width interval implies precision that was
                           never observed, so it is withheld
  * a seed that did not measure -- absent, never counted as zero
  * scipy unavailable   -- the spread is real, so report it and withhold the
                           interval rather than approximate a quantile

The last two matter most. Treating "not attempted" as "scored zero" is the exact
error that previously let an unmeasured metric earn a fidelity figure.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from backend.science.statistics.seed_pooling import (
    MIN_SEEDS_FOR_INTERVAL,
    pool_across_seeds,
    pool_metric,
)


def _seeds(values: List[float], base: int = 42):
    return [(base + i, v) for i, v in enumerate(values)]


# ── The interval is a t interval on seeds, not a Wilson on prompts ───────

def test_the_interval_matches_scipy_exactly():
    """If this drifts from scipy the number is not a measurement, it is a
    reimplementation nobody checked."""
    from scipy import stats
    import numpy as np

    values = [0.6654, 0.6506, 0.6636, 0.6695, 0.6712]
    pool = pool_across_seeds(_seeds(values), target="circuit_faithfulness")

    mean = np.mean(values)
    sd = np.std(values, ddof=1)
    low, high = stats.t.interval(0.95, len(values) - 1, loc=mean,
                                 scale=sd / math.sqrt(len(values)))

    assert pool.derived is True
    assert pool.mean == pytest.approx(mean, abs=1e-6)
    assert pool.sd == pytest.approx(sd, abs=1e-6)
    assert pool.ci_low == pytest.approx(low, abs=1e-6)
    assert pool.ci_high == pytest.approx(high, abs=1e-6)


def test_t_is_used_not_z():
    """The reason t and not z, asserted rather than asserted-in-prose.

    With 5 seeds, z understates the half-width by ~29%. A result that is only
    significant under z is not significant, and the whole point of moving to
    seed-level pooling is to stop small-n results reading as conclusive.
    """
    from scipy import stats

    values = [0.70, 0.68, 0.72, 0.69, 0.71]
    pool = pool_across_seeds(_seeds(values))

    n = len(values)
    sd = pool.sd
    t_half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    z_half = stats.norm.ppf(0.975) * sd / math.sqrt(n)

    assert pool.half_width == pytest.approx(t_half, abs=1e-6)
    assert pool.half_width > z_half * 1.35, (
        "the interval is narrower than a z interval by more than the "
        "correction factor; t is not being applied")
    assert f"df={n - 1}" in pool.method


def test_the_seed_count_is_reported_not_implied():
    pool = pool_across_seeds(_seeds([0.70, 0.68, 0.72], base=7))

    assert pool.n_seeds_requested == 3
    assert pool.n_seeds_used == 3
    assert pool.seeds == (7, 8, 9)


def test_the_per_seed_values_are_kept():
    """An interval says how precise; the values say whether to believe it.

    A bimodal spread and a tight cluster can share a mean and a half-width. Only
    the values distinguish them, so they must survive into the output.
    """
    values = [0.10, 0.90, 0.11, 0.89, 0.10]
    pool = pool_across_seeds(_seeds(values))

    assert pool.values == tuple(values)
    assert pool.observed_range == (0.10, 0.90)


# ── Fail-closed: every case that cannot support an interval ──────────────

def test_one_seed_yields_a_point_estimate_and_no_interval():
    pool = pool_across_seeds(_seeds([0.72]))

    assert pool.n_seeds_used == 1
    assert pool.derived is False
    assert pool.mean == pytest.approx(0.72)
    assert (pool.ci_low, pool.ci_high, pool.half_width) == (None, None, None)
    assert pool.sd is None, "a single value has no spread; reporting 0.0 would claim it does"
    assert "one draw" in pool.reason.lower()


def test_zero_usable_seeds_withholds_everything():
    pool = pool_across_seeds([])

    assert pool.derived is False
    assert pool.mean is None
    assert (pool.ci_low, pool.ci_high, pool.sd) == (None, None, None)
    assert pool.reason


def test_identical_seeds_withhold_rather_than_report_zero_width():
    """The specific false precision a zero-width interval produces.

    Identical values across every seed usually means the sample never varied --
    not that the measurement is infinitely precise. A zero-width interval would
    be indistinguishable from a perfect measurement to any consumer.
    """
    pool = pool_across_seeds(_seeds([0.72] * 5))

    assert pool.derived is False
    assert pool.mean == pytest.approx(0.72)
    assert pool.sd == 0.0, "the observed absence of variation is a real fact"
    assert (pool.ci_low, pool.ci_high, pool.half_width) == (None, None, None)
    assert "zero-width" in pool.reason


def test_a_seed_that_did_not_measure_is_discarded_not_zeroed():
    pool = pool_across_seeds([(42, 0.70), (43, None), (44, 0.72), (45, "n/a")])

    assert pool.n_seeds_requested == 4
    assert pool.n_seeds_used == 2
    assert pool.values == (0.70, 0.72)
    assert [d["seed"] for d in pool.discarded] == [43, 45]
    assert "discarded" in pool.reason


def test_booleans_are_not_treated_as_numbers():
    """`isinstance(True, int)` is True in Python.

    A pipeline reporting success flags where counts belong would otherwise have
    them silently averaged into a proportion.
    """
    pool = pool_across_seeds(_seeds([True, False, True]))

    assert pool.n_seeds_used == 0, "bools must not enter the mean"
    assert pool.mean is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_discarded(bad):
    pool = pool_across_seeds([(42, 0.70), (43, bad), (44, 0.72)])

    assert pool.n_seeds_used == 2
    assert pool.derived is True


# ── Bounds are clamped, and the clamping is disclosed ───────────────────

def test_an_interval_beyond_the_metric_range_is_clamped_and_says_so():
    """A CI on a proportion can cross the boundary; one outside [0,1] cannot be
    read as a proportion. Clamped -- but the clamp is recorded, because a
    silently narrowed interval is the same defect as a fabricated one."""
    # mean 0.944, sd 0.0472, t(df=4)=2.776 -> half-width 0.0573, so the
    # unclamped interval is [0.885365, 1.002635] and genuinely crosses the
    # boundary. Verified against the reported reason rather than hardcoded, so
    # this cannot drift into asserting a number the module never produced.
    values = [0.90, 1.00, 0.92, 0.99, 0.91]
    pool = pool_across_seeds(_seeds(values), target="accuracy", bounds=(0.0, 1.0))

    assert pool.derived is True
    assert pool.ci_high == 1.0
    assert pool.ci_low < pool.ci_high
    assert "clamped" in pool.reason
    assert f"{pool.mean + 0.057317:.6f}" in pool.reason or "1.002635" in pool.reason, (
        "the unclamped bound must be stated, not just the clamped one")


def test_an_interval_inside_the_range_is_not_clamped():
    pool = pool_across_seeds(_seeds([0.50, 0.52, 0.51, 0.49, 0.50]),
                             bounds=(0.0, 1.0))

    assert pool.derived is True
    assert pool.reason is None or "clamped" not in pool.reason


def test_min_seeds_threshold_is_what_the_module_claims():
    assert MIN_SEEDS_FOR_INTERVAL == 2


# ── pool_metric: reading the metric out of per-seed result dicts ────────

def test_pool_metric_reads_observed_metrics():
    results = [
        {"seed": 42, "observed_metrics": {"circuit_faithfulness": 0.70}},
        {"seed": 43, "observed_metrics": {"circuit_faithfulness": 0.72}},
        {"seed": 44, "observed_metrics": {"circuit_faithfulness": 0.71}},
    ]
    pool = pool_metric(results, metric="circuit_faithfulness",
                       target="circuit_faithfulness", bounds=(0.0, 1.0))

    assert pool.n_seeds_used == 3
    assert pool.seeds == (42, 43, 44)
    assert pool.mean == pytest.approx(0.71)
    assert pool.derived is True


def test_pool_metric_does_not_treat_a_missing_metric_as_zero():
    """The error that previously let an unmeasured metric earn a fidelity score.

    `observed_metrics.get(metric, 0.0)` is the tempting one-liner here. It turns
    "not attempted" into "attempted and scored zero", which lowers the mean and
    looks like a measurement.
    """
    results = [
        {"seed": 42, "observed_metrics": {"circuit_faithfulness": 0.70}},
        {"seed": 43, "observed_metrics": {}},          # metric absent
        {"seed": 44, "observed_metrics": {"circuit_faithfulness": 0.72}},
    ]
    pool = pool_metric(results, metric="circuit_faithfulness")

    assert pool.values == (0.70, 0.72), "the absent metric must not become 0.0"
    assert pool.mean == pytest.approx(0.71)
    assert pool.n_seeds_used == 2


def test_pool_metric_survives_a_result_with_no_observed_metrics_at_all():
    results = [{"seed": 42}, {"seed": 43}, {"seed": 44}]
    pool = pool_metric(results, metric="circuit_faithfulness")

    assert pool.n_seeds_used == 0
    assert pool.mean is None
    assert pool.derived is False


def test_pool_metric_tolerates_a_non_integer_seed():
    results = [
        {"seed": "42", "observed_metrics": {"m": 0.7}},
        {"seed": None, "observed_metrics": {"m": 0.8}},
    ]
    pool = pool_metric(results, metric="m")

    assert pool.n_seeds_used == 2


# ── to_dict: None must survive serialisation ────────────────────────────

def test_to_dict_keeps_none_as_none():
    """A withheld interval that serialises to 0.0 is not withheld.

    `round(None)` raises and `None or 0` yields 0, so this has to be checked
    rather than assumed.
    """
    import json

    pool = pool_across_seeds(_seeds([0.72]))
    payload = pool.to_dict()

    assert payload["ci_low"] is None
    assert payload["half_width"] is None
    assert payload["derived"] is False

    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["ci_low"] is None
    assert round_tripped["ci_high"] is None
    assert round_tripped["sd"] is None


def test_to_dict_is_json_serialisable_when_derived():
    import json

    pool = pool_across_seeds(_seeds([0.70, 0.72, 0.71]), bounds=(0.0, 1.0))
    payload = json.loads(json.dumps(pool.to_dict()))

    assert payload["derived"] is True
    assert payload["n_seeds_used"] == 3
    assert isinstance(payload["values"], list)


# ── The IOI pipeline's audit actually pools ────────────────────────────

class _StubbedPipeline:
    """Exercises the real `run_stability_audit` with a stubbed `run`.

    The live pipeline cannot run here -- mock mode refuses by design ("no random
    or synthetic logit fallback was generated"), and a live run is ~78 s per
    seed on this machine. The aggregation is what is under test, so `run` is
    replaced and the pooling code path is executed unchanged.
    """

    @staticmethod
    def build(values: Dict[int, Any]):
        from backend.science.reproducibility.ioi_pipeline import IOIReproductionPipeline

        calls: List[Dict[str, Any]] = []

        def fake_run(self, n_prompts=100, seed=42, model_variant="small"):
            calls.append({"n_prompts": n_prompts, "seed": seed,
                          "model_variant": model_variant})
            value = values.get(seed)
            if value is None:
                return {"status": "unavailable", "provenance": "unavailable",
                        "reason": "no live measurement for this seed"}
            return {"status": "completed", "provenance": "live",
                    "observed_metrics": {"circuit_faithfulness": value}}

        original = IOIReproductionPipeline.run
        IOIReproductionPipeline.run = fake_run
        return IOIReproductionPipeline(mock_mode=True), calls, original

    @staticmethod
    def restore(original):
        from backend.science.reproducibility.ioi_pipeline import IOIReproductionPipeline

        IOIReproductionPipeline.run = original


@pytest.fixture
def stubbed():
    applied = []

    def _apply(values: Dict[int, Any]):
        pipe, calls, original = _StubbedPipeline.build(values)
        applied.append(original)
        return pipe, calls

    yield _apply
    for original in applied:
        _StubbedPipeline.restore(original)


def test_the_audit_pools_across_seeds(stubbed):
    pipe, calls = stubbed({42: 0.70, 43: 0.72, 44: 0.71, 45: 0.69, 46: 0.73})

    audit = pipe.run_stability_audit(n_seeds=5, n_prompts=6)

    assert audit["status"] == "completed"
    assert audit["provenance"] == "live"
    assert audit["n_seeds_run"] == 5
    assert audit["n_seeds_live"] == 5
    assert audit["pooled"]["derived"] is True
    assert audit["pooled"]["n_seeds_used"] == 5
    assert audit["pooled"]["seeds"] == [42, 43, 44, 45, 46]
    assert audit["pooled"]["mean"] == pytest.approx(0.71)
    assert audit["pooled"]["ci_low"] < audit["pooled"]["mean"] < audit["pooled"]["ci_high"]
    assert audit["reason"] is None
    assert [c["seed"] for c in calls] == [42, 43, 44, 45, 46]


def test_the_audit_returns_a_mapping_not_a_bare_list(stubbed):
    """It used to return List[Dict] -- runs, not a finding.

    A list of per-seed results has no interval and no seed count, so nothing can
    be concluded from it. The return type is the fix.
    """
    pipe, _ = stubbed({42: 0.70, 43: 0.72, 44: 0.71})

    audit = pipe.run_stability_audit(n_seeds=3)

    assert isinstance(audit, dict)
    assert not isinstance(audit, list)
    assert audit["per_seed"], "per-seed values must be reported"
    assert len(audit["per_seed"]) == 3


def test_the_audit_reports_n_prompts_per_seed(stubbed):
    pipe, calls = stubbed({42: 0.70, 43: 0.71, 44: 0.72})

    audit = pipe.run_stability_audit(n_seeds=3, n_prompts=7)

    assert audit["n_prompts_per_seed"] == 7
    assert all(c["n_prompts"] == 7 for c in calls), (
        "n_prompts was hardcoded to 40; a caller could not spend more prompts "
        "per seed without editing the pipeline")


def test_a_single_seed_audit_is_a_point_estimate_not_an_interval(stubbed):
    pipe, _ = stubbed({42: 0.72})

    audit = pipe.run_stability_audit(n_seeds=1)

    assert audit["n_seeds_run"] == 1
    assert audit["pooled"]["derived"] is False
    assert audit["pooled"]["ci_low"] is None
    assert audit["status"] == "completed", "the run happened; only the pooling did not"
    assert audit["reason"] and "Pooling is incomplete" in audit["reason"]


def test_seeds_that_did_not_run_are_absent_not_zero(stubbed):
    pipe, _ = stubbed({42: 0.70, 43: None, 44: 0.72, 45: None, 46: 0.71})

    audit = pipe.run_stability_audit(n_seeds=5)

    assert audit["n_seeds_run"] == 5
    assert audit["n_seeds_live"] == 3
    assert audit["pooled"]["n_seeds_used"] == 3
    assert 0.0 not in audit["pooled"]["values"], (
        "a seed that did not run must be absent, not scored zero")
    assert audit["status"] == "completed"
    assert "did not run live" in audit["reason"]


def test_an_audit_where_nothing_ran_is_unavailable(stubbed):
    pipe, _ = stubbed({})

    audit = pipe.run_stability_audit(n_seeds=3)

    assert audit["status"] == "unavailable"
    assert audit["provenance"] == "unavailable"
    assert audit["pooled"]["mean"] is None
    assert audit["pooled"]["derived"] is False


def test_zero_seeds_is_refused_rather_than_reported_as_an_empty_success(stubbed):
    pipe, calls = stubbed({42: 0.70})

    audit = pipe.run_stability_audit(n_seeds=0)

    assert audit["status"] == "unavailable"
    assert calls == [], "n_seeds=0 must not run the pipeline at all"


def test_the_audit_is_not_publication_eligible_when_nothing_ran_live(stubbed):
    pipe, _ = stubbed({})

    audit = pipe.run_stability_audit(n_seeds=2)

    assert audit["validation_eligible"] is False
    assert audit["publication_eligible"] is False


# ── The scheduler must not be able to omit the seed count ───────────────

def test_every_scheduler_measurement_branch_states_its_seed_count():
    """A new branch that forgets `n_seeds` silently reinstates the ambiguity.

    This is the whole deliverable of the seed-pooling work: a reported score
    says how many independent draws it rests on. A return dict that omits the
    field leaves `n_samples=10` to speak for itself, and `n_samples` counts
    prompts -- ten prompts from one draw is not ten observations.

    Checked over the AST rather than by calling each pipeline, because four of
    the five need live GPT-2 weights and a GPU to reach.
    """
    import ast
    from pathlib import Path

    path = (Path(__file__).resolve().parents[2]
            / "backend" / "validation" / "benchmark_scheduler.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_measure")

    returns = [n for n in ast.walk(fn)
               if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)]
    assert returns, "no dict returns found in _measure; the shape changed"

    missing = []
    for node in returns:
        keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
        if "n_seeds" not in keys:
            missing.append(node.lineno)

    assert not missing, (
        "these _measure return dicts do not state n_seeds, so a score from them "
        "cannot be told apart from one draw: "
        + ", ".join(f"line {line}" for line in missing))


def test_the_scheduler_defaults_to_one_seed_and_says_so():
    """Pooling must be opt-in, because five IOI seeds is five times the cost.

    The default stays 1 -- but a default of 1 is only honest if it is recorded,
    which is what the guard above enforces.
    """
    import ast
    import inspect
    from pathlib import Path

    from backend.validation.benchmark_scheduler import ValidationBenchmarkScheduler

    signature = inspect.signature(ValidationBenchmarkScheduler._measure)
    assert signature.parameters["n_seeds"].default == 1, (
        "the default seed count changed; if pooling is now the default, the "
        "runtime cost of every benchmark run just multiplied")

    path = (Path(__file__).resolve().parents[2]
            / "backend" / "validation" / "benchmark_scheduler.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_run_one")
    assert "n_seeds" in [a.arg for a in fn.args.args], (
        "_run_one does not accept n_seeds, so the scheduler cannot request "
        "pooling at all")


def test_pooling_is_refused_for_pipelines_that_do_not_support_it():
    """Only IOI has a pooling path. Asking another pipeline must not pretend.

    `_measure("induction_heads", n_seeds=5)` has to stay a single draw and say
    so, rather than reporting n_seeds=5 because that is what was asked for.
    """
    import ast
    from pathlib import Path

    path = (Path(__file__).resolve().parents[2]
            / "backend" / "validation" / "benchmark_scheduler.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_measure")

    # Every branch other than ioi must report n_seeds <= 1.
    branch_n_seeds = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                if (isinstance(key, ast.Constant) and key.value == "n_seeds"
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, int)):
                    branch_n_seeds.append(value.value)

    assert branch_n_seeds, "no branch states a literal n_seeds"
    assert all(n <= 1 for n in branch_n_seeds), (
        "a single-draw branch claims more than one seed: "
        f"{branch_n_seeds}")


# ── The README's quoted figures must be the measured ones ───────────────

def test_the_readme_quotes_the_measured_seed_audit():
    """The audit table in README section 5 must match the artifact.

    A measured number copied into prose by hand rots the moment the run is
    repeated. Every figure the README states about the seed audit is checked
    against `docs/results/ioi_seed_audit.json` instead, so re-running the audit
    and forgetting to update the README is a test failure rather than a quiet
    discrepancy.
    """
    import json
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    artifact = root / "docs" / "results" / "ioi_seed_audit.json"
    assert artifact.exists(), (
        f"{artifact.name} is missing, so the README's audit table has nothing "
        f"to be checked against")

    data = json.loads(artifact.read_text(encoding="utf-8"))
    audit = data["audit"]
    pooled = audit["pooled"]

    assert audit["provenance"] == "live", (
        "the artifact backing a published figure is not live")
    assert audit["n_seeds_live"] == audit["n_seeds_run"] > 1

    readme = (root / "README.md").read_text(encoding="utf-8")
    # Normalise whitespace: the table wraps across lines, and a guard that
    # breaks on reformatting gets deleted rather than obeyed.
    flat = re.sub(r"\s+", " ", readme)

    per_seed = {s["seed"]: s["value"] for s in audit["per_seed"]}
    for seed, value in per_seed.items():
        assert f"{value}" in flat, (
            f"README does not quote the measured value {value} for seed {seed}")

    for figure in (pooled["mean"], pooled["sd"]):
        assert f"{figure}" in flat, (
            f"README does not quote the measured {figure}")

    for bound in (pooled["ci_low"], pooled["ci_high"]):
        assert f"{bound:.4f}" in flat, (
            f"README does not quote the measured CI bound {bound:.4f}")

    assert f"{data['elapsed_s']} s" in flat, (
        "README does not quote the measured elapsed time")


def test_the_published_baseline_is_not_presented_as_reached():
    """0.88 must not appear as a result the harness attains.

    The pooled interval tops out well below it, and the same-harness figure for
    the published circuit is lower still. Presenting 0.88 as something this
    harness measured is the claim this whole section exists to prevent.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    data = json.loads((root / "docs" / "results" / "ioi_seed_audit.json")
                      .read_text(encoding="utf-8"))
    pooled = data["audit"]["pooled"]

    assert pooled["ci_high"] < 0.88, (
        "the pooled interval now reaches the published baseline; the README "
        "text claiming it does not must be revisited")