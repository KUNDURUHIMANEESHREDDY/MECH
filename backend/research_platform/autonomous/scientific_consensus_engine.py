"""Scientific Consensus Engine — Aggregates reviewer feedback and resolves conflicts.

Integrates with the Peer Review Panel to generate diagnostic scorecards and
detailed reasoning traces for mechanism claims.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Dict, List

from .peer_review_panel import PeerReviewResult, ReviewVerdict


class ScientificConsensusEngine:
    """Combines evidence across multiple experiments and reviewers to resolve findings."""

    def synthesize_consensus(self, experimental_outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarise agreement across supplied outcomes.

        This previously returned the fixed sentence "L8_N402 acts as primary
        Indirect Object Identifier across IOI prompts" at confidence 0.94 for
        *any* input -- including an empty list, where `conflicts_resolved` was
        simply `1 if len(...) > 1 else 0`. A consensus engine that emits a
        specific mechanistic conclusion irrespective of the evidence is the
        single most dangerous shape this codebase can have, because every
        downstream consumer reads it as a reviewed finding.

        It now reports what the supplied outcomes actually agree on, and
        states no mechanism it was not given.
        """
        outcomes = [o for o in (experimental_outcomes or []) if isinstance(o, dict)]
        if not outcomes:
            return {
                "outcomes_analyzed": 0,
                "conflicts_resolved": 0,
                "unified_consensus_statement": None,
                "consensus_confidence": 0.0,
                "agreement": None,
                "provenance": "unavailable",
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": ("No experimental outcomes were supplied, so there "
                           "is nothing to reach consensus over. This method "
                           "previously returned a fixed mechanistic claim at "
                           "confidence 0.94 regardless of its input."),
            }

        # Report the distribution of whatever verdict field the caller used,
        # rather than asserting a conclusion of our own.
        verdicts: Dict[str, int] = {}
        for outcome in outcomes:
            key = str(outcome.get("verdict")
                      or outcome.get("outcome_state")
                      or outcome.get("status")
                      or "unspecified").lower()
            verdicts[key] = verdicts.get(key, 0) + 1

        total = len(outcomes)
        top_verdict, top_count = max(verdicts.items(), key=lambda kv: kv[1])
        unanimous = top_count == total

        return {
            "outcomes_analyzed": total,
            "verdict_distribution": verdicts,
            "majority_verdict": top_verdict,
            "unanimous": unanimous,
            "agreement_fraction": round(top_count / total, 4),
            "conflicts_resolved": total - top_count,
            # No mechanism is synthesised here; the callers' own claims are the
            # only statements available, and this method cannot verify them.
            "unified_consensus_statement": None,
            "consensus_confidence": 0.0,
            "provenance": "reference",
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": ("Consensus here means agreement between the supplied "
                       "verdicts, not a verified mechanism. This engine does "
                       "not synthesise mechanistic claims."),
        }


    def generate_peer_review_report(self, review_result: PeerReviewResult, output_dir: str = "benchmark_report") -> str:
        """Generates a detailed peer_review_report.md with reasoning traces and scorecards."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        report_path = os.path.join(output_dir, "peer_review_report.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Scientific Peer Review Report\n\n")
            f.write(f"**Claim ID**: `{review_result.claim_id}`\n")
            f.write(f"**Final Verdict**: {self._get_verdict_badge(review_result.final_verdict)}\n")
            f.write(f"**Overall Diagnostic Score**: {review_result.overall_score_pct}%\n")
            f.write(f"**Consensus Type**: {'Unanimous PASS' if review_result.unanimous_pass else 'Mixed Review'}\n\n")

            f.write("## Reviewer Scorecard\n\n")
            f.write("| Reviewer | Score | Verdict | Reasoning |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for out in review_result.reviewer_outputs:
                f.write(f"| {out.reviewer_name} | {out.score}/{out.max_score} | {out.verdict.value} | {', '.join(out.reasoning) if out.reasoning else 'Criteria met.'} |\n")

            f.write("\n## Reasoning Traces\n\n")
            for out in review_result.reviewer_outputs:
                f.write(f"### {out.reviewer_name} Trace\n")
                if out.reasoning:
                    f.write("- **Issues Flagged**:\n")
                    for r in out.reasoning:
                        f.write(f"  - {r}\n")
                else:
                    f.write("- **Logic**: All methodology and data rigor standards were satisfied according to established benchmarks.\n")
                f.write("\n")

            f.write("---\n")
            f.write(f"*Report generated at {_dt.datetime.utcnow().isoformat()}Z*\n")

        return report_path

    def _get_verdict_badge(self, verdict: ReviewVerdict) -> str:
        if verdict == ReviewVerdict.PASS:
            return "✅ **PASS**"
        if verdict == ReviewVerdict.REVISION_REQUIRED:
            return "⚠️ **REVISION REQUIRED**"
        return "❌ **FAIL**"
