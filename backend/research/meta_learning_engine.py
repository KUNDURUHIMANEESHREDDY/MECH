"""Multi-Campaign Meta-Learning Engine.

Derives research execution strategies from historical campaign analytics.

What this used to do, and why it was wrong:

* Algorithm priority weights were seeded with a fixed ranking
  ({"attribution_patching": 1.0, "acdc": 0.95, "transcoders": 0.85, ...}) and
  only *overwritten* for algorithms that happened to appear in the leaderboard.
  Any algorithm absent from the archive kept its invented weight, so the policy
  looked learned while containing a hardcoded opinion about algorithms that had
  never run.
* `recommended_sequence`, `pruned_sequences`, `expected_convergence_speed`
  (0.95) and `expected_final_posterior` (0.962) were literals selected by an
  `if task_category == "ioi"` branch. Nothing in the analytics report was read
  to produce them, and they were emitted even for an empty archive.
* `policy_confidence` was `0.80 + campaigns * 0.05`. Confidence rose with the
  campaign count and nothing else -- not with any measurement, and not with
  whether the archive contained a single experiment.

The policy is now derived from the report, and every field that cannot be
derived is None with a reason. An engine that has learned nothing reports that
it has learned nothing.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .campaign_analytics import CampaignAnalyticsEngine, CampaignAnalyticsReport

# Below this many completed experiments, an observed success rate carries too
# little evidence to rank algorithms by. Arbitrary, but explicit: the point is
# that the threshold is a decision, not an accident.
MIN_RUNS_TO_RANK = 5


@dataclass
class OptimizedPlannerPolicy:
    """Execution policy derived from campaign history.

    The `expected_*` fields are Optional. An expectation requires a basis, and
    an empty archive has none.
    """
    task_category: str
    recommended_sequence: Optional[List[str]]
    pruned_sequences: List[List[str]]
    algorithm_priority_weights: Dict[str, float]
    expected_convergence_speed: Optional[float]
    expected_final_posterior: Optional[float]
    policy_confidence: Optional[float]
    learned_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    # Why the unmeasured fields are unmeasured. Present so a consumer reading
    # `policy_confidence: None` learns something instead of guessing.
    measured: bool = False
    unmeasured_reason: Optional[str] = None
    ranked_algorithms: int = 0
    ranked_on_runs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_category": self.task_category,
            "recommended_sequence": self.recommended_sequence,
            "pruned_sequences": self.pruned_sequences,
            "algorithm_priority_weights": self.algorithm_priority_weights,
            "expected_convergence_speed": self.expected_convergence_speed,
            "expected_final_posterior": self.expected_final_posterior,
            "policy_confidence": self.policy_confidence,
            "measured": self.measured,
            "unmeasured_reason": self.unmeasured_reason,
            "ranked_algorithms": self.ranked_algorithms,
            "ranked_on_runs": self.ranked_on_runs,
            # A derived ranking is real data; it is still not a validated policy.
            "provenance": "live" if self.measured else "unavailable",
            "validation_eligible": False,
            "publication_eligible": False,
            "learned_at": self.learned_at,
        }


class MetaLearningEngine:
    """Derives research planning policies from archived campaign evidence."""

    def __init__(
        self,
        analytics_engine: Optional[CampaignAnalyticsEngine] = None,
    ) -> None:
        self.analytics_engine = analytics_engine or CampaignAnalyticsEngine()

    def learn_policy(self, task_category: str = "ioi") -> OptimizedPlannerPolicy:
        """Derive a policy for a task category from what actually ran.

        With an empty or too-small archive this returns a policy with no
        sequence, no expectations, and no confidence. It does not fall back to
        a default opinion dressed as a learned one.
        """
        report: CampaignAnalyticsReport = self.analytics_engine.analyze()

        # Rank only algorithms with enough completed runs behind them, and
        # only from observed success. No seeding: an algorithm that never ran
        # gets no weight, because there is no evidence about it.
        weights: Dict[str, float] = {}
        for stat in report.algorithm_leaderboard:
            if stat.total_runs < MIN_RUNS_TO_RANK:
                continue
            weights[stat.algorithm_name] = round(stat.success_rate, 4)

        ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        total_runs = sum(s.total_runs for s in report.algorithm_leaderboard)

        if not ranked:
            return OptimizedPlannerPolicy(
                task_category=task_category,
                recommended_sequence=None,
                pruned_sequences=[],
                algorithm_priority_weights={},
                expected_convergence_speed=None,
                expected_final_posterior=None,
                policy_confidence=None,
                measured=False,
                unmeasured_reason=(
                    "No algorithm in the archive has at least "
                    f"{MIN_RUNS_TO_RANK} completed runs, so nothing can be "
                    "ranked. Previously this returned a fixed sequence with "
                    "expected_convergence_speed=0.95 regardless of the archive."
                ),
                ranked_algorithms=0,
                ranked_on_runs=total_runs,
            )

        recommended = [alg for alg, _ in ranked]
        pruned = [s.sequence for s in report.failure_patterns]

        # Sequence-level statistics come from the observed sequences, and only
        # when the archive actually contains sequences to average.
        seq_rows = [s for s in report.best_sequences if s.convergence_speed_score is not None]
        exp_speed = (
            round(sum(s.convergence_speed_score for s in seq_rows) / len(seq_rows), 4)
            if seq_rows else None
        )
        posteriors = [s.avg_final_posterior for s in report.best_sequences
                      if s.avg_final_posterior is not None]
        exp_posterior = (
            round(sum(posteriors) / len(posteriors), 4) if posteriors else None
        )

        # Confidence is the spread of the evidence, not a function of its
        # volume alone: it reflects both how many runs back the ranking and how
        # much the top algorithms actually differ from each other. If every
        # observed algorithm succeeded equally, there is nothing to be
        # confident about.
        spread = (ranked[0][1] - ranked[-1][1]) if len(ranked) > 1 else 0.0
        volume = min(1.0, total_runs / 50.0)
        policy_conf = round(min(0.99, 0.50 + (spread * 0.40) + (volume * 0.09)), 4)

        return OptimizedPlannerPolicy(
            task_category=task_category,
            recommended_sequence=recommended,
            pruned_sequences=pruned,
            algorithm_priority_weights=weights,
            expected_convergence_speed=exp_speed,
            expected_final_posterior=exp_posterior,
            policy_confidence=policy_conf,
            measured=True,
            unmeasured_reason=(
                None if (exp_speed is not None and exp_posterior is not None)
                else (
                    "Algorithm ranking is derived from observed runs, but "
                    "sequence-level expectations are absent because the "
                    "archive does not contain enough distinct sequences to "
                    "compare."
                )
            ),
            ranked_algorithms=len(ranked),
            ranked_on_runs=total_runs,
        )
