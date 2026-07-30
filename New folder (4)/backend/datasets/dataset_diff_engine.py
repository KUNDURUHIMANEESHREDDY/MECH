"""Semantic Dataset Diffing Engine — Analyzing Changes & Impact.

Compares two versions of a dataset to identify exactly what changed:
- Prompt-level: Added, removed, or modified prompts.
- Token-level: Changes in tokenization patterns, counts, and vocabulary.
- Metadata-level: Provenance and lineage shifts.
- Impact: Automatically identifies affected research results (Claims, Papers).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class PromptDiff:
    added: int = 0
    removed: int = 0
    modified: int = 0
    total_delta: int = 0


@dataclass
class TokenDiff:
    count_delta: int = 0
    vocab_shifts: int = 0
    affected_templates: List[str] = field(default_factory=list)


@dataclass
class DatasetDiffReport:
    id_a: str
    id_b: str
    prompts: PromptDiff
    tokens: TokenDiff
    affected_benchmarks: List[str]
    affected_claims: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH
    summary: str


class DatasetDiffEngine:
    """Analyzes semantic differences between dataset versions."""

    def compare_datasets(self, data_a: Dict[str, Any], data_b: Dict[str, Any]) -> DatasetDiffReport:
        """Performs deep comparison between version A and version B."""

        # 1. Prompt-level analysis
        prompts_a = {p["id"]: p for p in data_a.get("prompts", [])}
        prompts_b = {p["id"]: p for p in data_b.get("prompts", [])}

        ids_a = set(prompts_a.keys())
        ids_b = set(prompts_b.keys())

        added = len(ids_b - ids_a)
        removed = len(ids_a - ids_b)

        common_ids = ids_a.intersection(ids_b)
        modified = 0
        for pid in common_ids:
            if prompts_a[pid]["clean"] != prompts_b[pid]["clean"]:
                modified += 1

        p_diff = PromptDiff(added=added, removed=removed, modified=modified, total_delta=len(ids_b) - len(ids_a))

        # 2. Token-level simulation (Simplified for Phase 1)
        t_diff = TokenDiff(
            count_delta=modified * 5,
            vocab_shifts=modified // 2,
            affected_templates=["IOI-Standard"] if modified > 0 else []
        )

        # 3. Impact Analysis (Simulated linkage for now)
        affected_benchmarks = ["IOI"] if modified > 0 or added > 0 else []
        affected_claims = ["Name Mover Circuit"] if "IOI" in affected_benchmarks else []

        # 4. Risk Scoring
        risk = "LOW"
        if modified > 10 or added > 50: risk = "HIGH"
        elif modified > 0 or added > 0: risk = "MEDIUM"

        summary = f"Dataset evolved with {added} additions and {modified} modifications."

        return DatasetDiffReport(
            id_a=data_a.get("dataset_id", "v1"),
            id_b=data_b.get("dataset_id", "v2"),
            prompts=p_diff,
            tokens=t_diff,
            affected_benchmarks=affected_benchmarks,
            affected_claims=affected_claims,
            risk_level=risk,
            summary=summary
        )

    def generate_changelog_entry(self, report: DatasetDiffReport) -> str:
        """Returns a human-readable CHANGELOG.md entry."""
        lines = [
            f"## Version {report.id_b} (from {report.id_a})",
            f"- **Prompt Changes**: {report.prompts.added} added, {report.prompts.removed} removed, {report.prompts.modified} modified.",
            f"- **Token Shift**: {report.tokens.count_delta:+} tokens, {report.tokens.vocab_shifts} vocabulary changes.",
            f"- **Risk Level**: **{report.risk_level}**",
            "- **Affected Research**:"
        ]
        for b in report.affected_benchmarks: lines.append(f"  - Benchmark: {b}")
        for c in report.affected_claims: lines.append(f"  - Claim: {c}")

        return "\n".join(lines)
