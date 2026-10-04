"""Pool a measurement across seeds, and say how many seeds it came from.

The problem this exists to fix
------------------------------
A confidence interval over per-prompt outcomes answers "if we drew 100 prompts,
how precise is the proportion?". It is a Wilson interval, computed inside
`benchmark_tasks.py`, from the pipeline's real `correct`/`trials`.

That is the right shape for the wrong question when `n_seeds == 1`. Prompts
generated under one seed are not independent draws -- they share that seed's
name set and RNG stream -- so pooling 100 of them and reporting a narrow
interval claims precision the design does not have. As the README puts it: a
single 100-prompt draw is still one draw.

So the interval has to come from the **seed-level** values, with the seed as the
unit of variation:

  * `n_seeds = 1`  -> no interval is derivable from seeds at all. Say so.
  * `n_seeds >= 2` -> Student-t on the seed means, `df = n_seeds - 1`.

Why t and not z: with 3-5 seeds the sampling distribution of the mean is
markedly non-normal, and z would understate the interval by exactly the amount
that makes a small-n result look conclusive. The t quantile at df=4 is 2.776
against z's 1.960 -- a 42% wider interval, which is the honest number.

This is deliberately *not* a Wilson interval over the pooled prompts. Both can be
reported, and `benchmark_tasks.py` keeps the Wilson one, because it bounds a
different quantity. Conflating them would present a narrow prompt-level interval
as though it covered seed-to-seed variation, which is the error this module
exists to prevent.

Fail-closed throughout: every field is `None` rather than a default when the
inputs do not support it, and `derived` is False unless an interval was actually
computed from real values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["SeedPool", "pool_across_seeds", "MIN_SEEDS_FOR_INTERVAL"]

#: Fewer than this many usable seed values cannot produce a seed-level interval.
#: One value has no variance; zero has nothing at all.
MIN_SEEDS_FOR_INTERVAL = 2


@dataclass(frozen=True)
class SeedPool:
    """The pooled outcome of a measurement repeated across seeds.

    `values` is kept rather than discarded. An interval tells a reader how
    precise the estimate is; the per-seed values tell them whether that
    precision came from seeds that agree or seeds that happen to cancel out.
    Reporting the mean alone lets a bimodal spread read as a tight result.
    """

    n_seeds_requested: int
    n_seeds_used: int
    values: Tuple[float, ...]
    seeds: Tuple[int, ...] = ()
    mean: Optional[float] = None
    sd: Optional[float] = None
    observed_range: Optional[Tuple[float, float]] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    half_width: Optional[float] = None
    method: Optional[str] = None
    target: Optional[str] = None
    derived: bool = False
    reason: Optional[str] = None
    discarded: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_seeds_requested": self.n_seeds_requested,
            "n_seeds_used": self.n_seeds_used,
            "seeds": list(self.seeds),
            "values": [round(v, 6) for v in self.values],
            "mean": (round(self.mean, 6) if self.mean is not None else None),
            "sd": (round(self.sd, 6) if self.sd is not None else None),
            "observed_range": (
                [round(self.observed_range[0], 6),
                 round(self.observed_range[1], 6)]
                if self.observed_range is not None else None),
            "ci_low": (round(self.ci_low, 6) if self.ci_low is not None else None),
            "ci_high": (round(self.ci_high, 6) if self.ci_high is not None else None),
            "half_width": (round(self.half_width, 6)
                           if self.half_width is not None else None),
            "method": self.method,
            "target": self.target,
            "derived": self.derived,
            "reason": self.reason,
            "discarded": self.discarded,
        }


def _finite(value: Any) -> bool:
    """A value usable in an interval: a real, finite number.

    `bool` is excluded deliberately. `isinstance(True, int)` is True in Python,
    so a pipeline that reports success flags as counts would otherwise have its
    booleans silently averaged into a proportion.
    """
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)))


def pool_across_seeds(
    per_seed: Sequence[Tuple[int, Any]],
    *,
    confidence: float = 0.95,
    target: Optional[str] = None,
    bounds: Optional[Tuple[float, float]] = None,
) -> SeedPool:
    """Pool one measured value per seed into a mean and a seed-level interval.

    `per_seed` is `(seed, value)` pairs in the order they were run.

    `bounds` optionally declares the natural range of the quantity (e.g. `(0, 1)`
    for a proportion). The interval is clamped into it, because a Student-t
    interval on a proportion can cross the boundary and a confidence interval
    outside the metric's own domain is not interpretable. Clamping is recorded
    in `reason` rather than applied silently.
    """
    n_requested = len(per_seed)
    values: List[float] = []
    seeds: List[int] = []
    discarded: List[Dict[str, Any]] = []

    for seed, value in per_seed:
        if _finite(value):
            values.append(float(value))
            seeds.append(int(seed))
        else:
            discarded.append({
                "seed": seed,
                "value": value if not isinstance(value, (list, dict)) else repr(value),
                "why": "no finite numeric measurement for this seed",
            })

    n_used = len(values)

    if n_used == 0:
        return SeedPool(
            n_seeds_requested=n_requested, n_seeds_used=0, values=(), seeds=(),
            target=target,
            reason=("No seed produced a usable measurement, so there is "
                    "nothing to pool and no interval to report."),
            discarded=discarded)

    mean = sum(values) / n_used
    observed_range = (min(values), max(values))

    if n_used < MIN_SEEDS_FOR_INTERVAL:
        return SeedPool(
            n_seeds_requested=n_requested, n_seeds_used=n_used,
            values=tuple(values), seeds=tuple(seeds),
            mean=round(mean, 6), sd=None, observed_range=observed_range,
            target=target, derived=False,
            reason=(f"{n_used} seed produced a value. One draw has no variance, "
                    "so no seed-level interval can be computed from it. This is "
                    "a point estimate, not a measured quantity with known "
                    "precision."),
            discarded=discarded)

    variance = sum((v - mean) ** 2 for v in values) / (n_used - 1)
    sd = math.sqrt(variance)

    # Identical values across every seed: sd is exactly 0 and the t interval
    # collapses to a point. That is not high precision -- it is the absence of
    # any observed variation, which for a sampled measurement means the sample
    # never varied. Report the collapse rather than a zero-width interval,
    # which would imply false precision of the kind the Wilson bounds were
    # introduced to avoid.
    if sd == 0.0:
        return SeedPool(
            n_seeds_requested=n_requested, n_seeds_used=n_used,
            values=tuple(values), seeds=tuple(seeds),
            mean=round(mean, 6), sd=0.0, observed_range=observed_range,
            target=target, derived=False,
            reason=(f"All {n_used} seeds returned exactly {mean!r}. The sample "
                    "never varied, so the interval is withheld rather than "
                    "reported as zero-width: a zero-width interval reads as "
                    "perfect precision, and what was actually observed is that "
                    "nothing moved."),
            discarded=discarded)

    try:
        from scipy import stats as _stats
        quantile = float(_stats.t.ppf(0.5 + confidence / 2.0, df=n_used - 1))
    except Exception as exc:  # pragma: no cover - scipy is a declared dependency
        return SeedPool(
            n_seeds_requested=n_requested, n_seeds_used=n_used,
            values=tuple(values), seeds=tuple(seeds),
            mean=round(mean, 6), sd=round(sd, 6), observed_range=observed_range,
            target=target, derived=False,
            reason=(f"The seed spread was measured (sd={sd:.6f}) but the t "
                    f"quantile is unavailable ({type(exc).__name__}), so no "
                    f"interval is reported rather than an approximated one."),
            discarded=discarded)

    half_width = quantile * sd / math.sqrt(n_used)
    low, high = mean - half_width, mean + half_width

    clamped = False
    if bounds is not None:
        floor, ceiling = bounds
        clamped = low < floor or high > ceiling
        low, high = max(floor, low), min(ceiling, high)

    reason = None
    if clamped:
        reason = (f"The Student-t interval [{mean - half_width:.6f}, "
                  f"{mean + half_width:.6f}] extends beyond the metric's "
                  f"natural range {bounds}, so it was clamped to "
                  f"[{low:.6f}, {high:.6f}]. The unclamped bounds are the "
                  f"bracketed pair above.")
    if discarded:
        extra = (f" {len(discarded)} seed(s) were discarded because they "
                 f"produced no usable value; they are listed under `discarded` "
                 f"rather than counted as failures or quietly dropped.")
        reason = (reason + extra) if reason else extra.lstrip()

    return SeedPool(
        n_seeds_requested=n_requested,
        n_seeds_used=n_used,
        values=tuple(values),
        seeds=tuple(seeds),
        mean=round(mean, 6),
        sd=round(sd, 6),
        observed_range=observed_range,
        ci_low=round(low, 6),
        ci_high=round(high, 6),
        half_width=round((high - low) / 2, 6),
        method=(f"Student t interval at {confidence:.0%} on {n_used} "
                f"seed-level means, df={n_used - 1}, t={quantile:.4f}"),
        target=target,
        derived=True,
        reason=reason,
        discarded=discarded,
    )


def pool_metric(
    results: Iterable[Dict[str, Any]],
    *,
    metric: str,
    seed_of: str = "seed",
    confidence: float = 0.95,
    target: Optional[str] = None,
    bounds: Optional[Tuple[float, float]] = None,
) -> SeedPool:
    """Pool one metric out of per-seed pipeline result dicts.

    Tolerates a missing metric per seed: a seed that did not measure is recorded
    in `discarded`, not counted as a zero. Treating "not attempted" as "scored
    zero" is the specific error that previously let an unmeasured metric earn a
    fidelity figure.
    """
    pairs: List[Tuple[int, Any]] = []
    for index, result in enumerate(results):
        seed = result.get(seed_of, index) if isinstance(result, dict) else index
        try:
            seed_int = int(seed)
        except (TypeError, ValueError):
            seed_int = index
        observed = (result.get("observed_metrics", {}) if isinstance(result, dict)
                    else {})
        value = observed.get(metric) if isinstance(observed, dict) else None
        pairs.append((seed_int, value))
    return pool_across_seeds(pairs, confidence=confidence, target=target,
                             bounds=bounds)