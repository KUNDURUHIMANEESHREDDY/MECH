"""Campaign Analytics Engine.

Extracts empirical performance statistics across all historical research campaigns:
  - Algorithm Success Rates
  - Average Runtime (ms) & Cost ($)
  - Average Bayesian Belief Gain (Delta P) & Information Density
  - Failure Mode Analysis (e.g., Running Universality before ACDC)
  - Compute Efficiency Metrics (Confidence Gain per TFLOP / per $1 USD)
  - Optimal Experiment Sequence Rankings
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..interpretability.discovery.research_campaign_manager import ResearchCampaignManager, ResearchCampaign


@dataclass
class AlgorithmPerformanceStat:
    """Empirical performance metrics for a single discovery algorithm."""
    algorithm_name: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    success_rate: float
    avg_runtime_ms: float
    avg_compute_flops: float
    avg_belief_gain: float
    confidence_gain_per_tflop: float
    confidence_gain_per_dollar: float


@dataclass
class SequencePerformanceStat:
    """Empirical performance metrics for an ordered sequence of algorithms.

    `convergence_speed_score` is Optional because a real convergence speed needs
    a comparison point. Without one, the honest answer is None, not 0.95.
    """
    sequence: List[str]
    occurrence_count: int
    avg_final_posterior: Optional[float]
    avg_total_runtime_ms: Optional[float]
    convergence_speed_score: Optional[float]
    recommendation_rank: int


@dataclass
class FailureModePattern:
    """Identified anti-pattern or failure mode in campaign execution.

    `root_cause` and `remedy` are Optional. Diagnosing *why* an algorithm
    failed is a separate act from observing that it did, and this engine only
    observes. Shipping a confident causal story next to a real failure count
    makes the count look diagnosed when it is not.
    """
    pattern_id: str
    description: str
    sequence: List[str]
    failure_count: int
    root_cause: Optional[str] = None
    remedy: Optional[str] = None


@dataclass
class CampaignAnalyticsReport:
    """Comprehensive analytics artifact extracted across all campaign history."""
    total_campaigns_analyzed: int
    overall_success_rate: float
    total_compute_flops: float
    total_usd_spent: float
    algorithm_leaderboard: List[AlgorithmPerformanceStat]
    best_sequences: List[SequencePerformanceStat]
    failure_patterns: List[FailureModePattern]
    generated_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_campaigns_analyzed": self.total_campaigns_analyzed,
            "overall_success_rate": round(self.overall_success_rate, 4),
            "total_compute_flops": round(self.total_compute_flops, 2),
            "total_usd_spent": round(self.total_usd_spent, 2),
            "algorithm_leaderboard": [a.__dict__ for a in self.algorithm_leaderboard],
            "best_sequences": [s.__dict__ for s in self.best_sequences],
            "failure_patterns": [f.__dict__ for f in self.failure_patterns],
            # Sequences and failure modes are derived from the archive now, so
            # they are empty when nothing has been observed. Empty means
            # "no evidence", which is the whole point.
            "sequences_measured": bool(self.best_sequences),
            "failure_patterns_measured": bool(self.failure_patterns),
            "convergence_speed_measured": bool(self.best_sequences) and all(
                s.convergence_speed_score is not None for s in self.best_sequences),
            "failure_root_causes_diagnosed": bool(self.failure_patterns) and all(
                f.root_cause is not None for f in self.failure_patterns),
            "generated_at": self.generated_at,
        }


class CampaignAnalyticsEngine:
    """Analyzes historical campaign logs to compute empirical performance benchmarks."""

    def __init__(self, campaign_manager: Optional[ResearchCampaignManager] = None) -> None:
        self.campaign_manager = campaign_manager or ResearchCampaignManager()

    def analyze(self) -> CampaignAnalyticsReport:
        """Executes full empirical analytics extraction across all campaign archives."""
        campaigns = self.campaign_manager.list_all()
        total_campaigns = len(campaigns)

        if total_campaigns == 0:
            return self._build_empty_report()

        successful_campaigns = sum(1 for c in campaigns if c.status == "Completed")
        overall_success_rate = successful_campaigns / max(1, total_campaigns)

        total_flops = sum(c.compute_consumed_flops for c in campaigns)
        total_usd = sum(10.0 - c.budget_remaining_usd for c in campaigns)

        # 1. Algorithm Leaderboard Analytics
        alg_counts: Dict[str, Dict[str, float]] = {}
        for c in campaigns:
            for exp in c.completed_experiments:
                alg = exp.algorithm_name
                if alg not in alg_counts:
                    alg_counts[alg] = {"total": 0, "success": 0, "runtime": 0.0, "flops": 0.0, "delta_p": 0.0}
                alg_counts[alg]["total"] += 1
                alg_counts[alg]["success"] += 1
                alg_counts[alg]["runtime"] += exp.runtime_ms
                alg_counts[alg]["flops"] += exp.compute_consumed_flops
                alg_counts[alg]["delta_p"] += (exp.uncertainty_before - exp.uncertainty_after)

            for exp in c.failed_experiments:
                alg = exp.algorithm_name
                if alg not in alg_counts:
                    alg_counts[alg] = {"total": 0, "success": 0, "runtime": 0.0, "flops": 0.0, "delta_p": 0.0}
                alg_counts[alg]["total"] += 1
                alg_counts[alg]["runtime"] += exp.runtime_ms
                alg_counts[alg]["flops"] += exp.compute_consumed_flops

        leaderboard: List[AlgorithmPerformanceStat] = []
        for alg, stats in alg_counts.items():
            tot = int(stats["total"])
            succ = int(stats["success"])
            s_rate = round(succ / max(1, tot), 3)
            avg_rt = round(stats["runtime"] / max(1, tot), 1)
            avg_fl = round(stats["flops"] / max(1, tot), 1)
            avg_dp = round(stats["delta_p"] / max(1, succ), 3)

            # Compute Efficiency: Delta P per 1e12 FLOPs & per $1 USD
            tflops = max(0.1, avg_fl / 1e12)
            gain_per_tflop = round(avg_dp / tflops, 3)
            gain_per_dollar = round(avg_dp / (tflops * 0.05), 3)

            leaderboard.append(AlgorithmPerformanceStat(
                algorithm_name=alg,
                total_runs=tot,
                successful_runs=succ,
                failed_runs=tot - succ,
                success_rate=s_rate,
                avg_runtime_ms=avg_rt,
                avg_compute_flops=avg_fl,
                avg_belief_gain=avg_dp,
                confidence_gain_per_tflop=gain_per_tflop,
                confidence_gain_per_dollar=gain_per_dollar
            ))

        leaderboard.sort(key=lambda x: (x.success_rate, x.avg_belief_gain), reverse=True)

        # 2. Experiment sequences, grouped by the order they actually ran in.
        #
        # These used to be two hand-written literals: occurrence_count=42,
        # avg_final_posterior=0.962, convergence_speed_score=0.95, ranked 1 and
        # 2. They were returned for every archive including an empty one, which
        # made a fixed recommendation list look like the output of mining.
        seq_runs: Dict[Tuple[str, ...], List[Dict[str, float]]] = {}
        for c in campaigns:
            ordered = tuple(exp.algorithm_name for exp in c.completed_experiments)
            if not ordered:
                continue
            final_posterior = 1.0 - c.completed_experiments[-1].uncertainty_after
            seq_runs.setdefault(ordered, []).append({
                "runtime": sum(exp.runtime_ms for exp in c.completed_experiments),
                "posterior": final_posterior,
            })

        seq_rows = []
        for ordered, runs in seq_runs.items():
            seq_rows.append({
                "sequence": list(ordered),
                "occurrence_count": len(runs),
                "avg_final_posterior": round(
                    sum(r["posterior"] for r in runs) / len(runs), 4),
                "avg_total_runtime_ms": round(
                    sum(r["runtime"] for r in runs) / len(runs), 2),
            })

        # Rank by measured occurrence, then by measured final posterior.
        seq_rows.sort(key=lambda r: (r["occurrence_count"],
                                     r["avg_final_posterior"]), reverse=True)

        # Convergence speed is relative: it needs at least two observed
        # sequences to compare against. With one, there is no comparison point,
        # so the score is None rather than a flattering constant.
        runtimes = [r["avg_total_runtime_ms"] for r in seq_rows]
        slowest = max(runtimes) if runtimes else 0.0
        comparable = slowest > 0 and len(seq_rows) >= 2

        best_seqs = [
            SequencePerformanceStat(
                sequence=row["sequence"],
                occurrence_count=row["occurrence_count"],
                avg_final_posterior=row["avg_final_posterior"],
                avg_total_runtime_ms=row["avg_total_runtime_ms"],
                convergence_speed_score=(
                    round((slowest - row["avg_total_runtime_ms"]) / slowest, 4)
                    if comparable else None
                ),
                recommendation_rank=index + 1,
            )
            for index, row in enumerate(seq_rows)
        ]

        # 3. Failure modes, counted from failed_experiments.
        #
        # Previously two invented patterns with failure_count=22 and 9, plus a
        # root cause and a remedy for each -- a causal diagnosis presented
        # beside a real-looking tally. Only the tally is derivable here.
        failure_counts: Dict[str, int] = {}
        failure_sequences: Dict[str, List[str]] = {}
        for c in campaigns:
            for exp in c.failed_experiments:
                failure_counts[exp.algorithm_name] = \
                    failure_counts.get(exp.algorithm_name, 0) + 1
                failure_sequences.setdefault(exp.algorithm_name, []).append(
                    exp.algorithm_name)

        fail_patterns = [
            FailureModePattern(
                pattern_id=f"fail_{alg}",
                description=f"{alg} failed {count} time(s) across archived campaigns",
                sequence=[alg],
                failure_count=count,
                root_cause=None,
                remedy=None,
            )
            for alg, count in sorted(failure_counts.items(),
                                     key=lambda kv: kv[1], reverse=True)
        ]

        return CampaignAnalyticsReport(
            total_campaigns_analyzed=total_campaigns,
            overall_success_rate=round(overall_success_rate, 4),
            total_compute_flops=total_flops,
            total_usd_spent=total_usd,
            algorithm_leaderboard=leaderboard,
            best_sequences=best_seqs,
            failure_patterns=fail_patterns
        )

    def _build_empty_report(self) -> CampaignAnalyticsReport:
        """Fallback empty report if no campaigns are archived."""
        return CampaignAnalyticsReport(
            total_campaigns_analyzed=0,
            overall_success_rate=0.0,
            total_compute_flops=0.0,
            total_usd_spent=0.0,
            algorithm_leaderboard=[],
            best_sequences=[],
            failure_patterns=[]
        )
