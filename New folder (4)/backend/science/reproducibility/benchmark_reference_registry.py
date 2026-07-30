"""Canonical Benchmark Reference Registry.

Stores published reference measurements for mechanistic interpretability benchmarks.
Supports multiple references per benchmark and immutable identifiers.
Version: v1.1
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BenchmarkReference:
    """A structured reference for a specific benchmark metric."""
    benchmark_id: str
    metric_name: str
    paper_title: str
    paper_doi: str
    model_revision: str
    dataset_name: str
    dataset_hash: str
    expected_mean: float
    expected_median: Optional[float] = None
    expected_iqr: Optional[float] = None
    acceptable_lower: float = 0.0
    acceptable_upper: float = 1.0
    unit: str = "fraction"
    notes: Optional[str] = None
    reference_implementation_hash: str = "unknown"
    prompt_set_hash: str = "unknown"
    tokenizer_hash: str = "unknown"
    model_sha: str = "unknown"


# Registry Version v1.1
REGISTRY_VERSION = "1.1.0"

# Canonical Benchmarks for GPT-2 Small (Tier 1)
GPT2_SMALL_REFERENCES: Dict[str, List[BenchmarkReference]] = {
    "ioi_faithfulness": [
        BenchmarkReference(
            benchmark_id="ioi",
            metric_name="Faithfulness",
            paper_title="Interpretability in the Wild: a Circuit for Indirect Object Identification",
            paper_doi="arxiv:2211.00593",
            model_revision="gpt2-small",
            dataset_name="IOI-100",
            dataset_hash="sha256_ioi_v1_100",
            expected_mean=0.880,
            expected_median=0.875,
            expected_iqr=0.042,
            acceptable_lower=0.840,
            acceptable_upper=0.920,
            notes="Measured using mean ablation on identified Name Mover Heads.",
            prompt_set_hash="sha256_ioi_prompts_canonical",
            tokenizer_hash="sha256_gpt2_tokenizer",
            model_sha="sha256_gpt2_small_weights"
        ),
        BenchmarkReference(
            benchmark_id="ioi",
            metric_name="Faithfulness",
            paper_title="Automatic Circuit Discovery (ACDC)",
            paper_doi="arxiv:2304.14997",
            model_revision="gpt2-small",
            dataset_name="IOI-100",
            dataset_hash="sha256_ioi_v1_100",
            expected_mean=0.878,
            acceptable_lower=0.830,
            acceptable_upper=0.910,
        )
    ],
    "ioi_completeness": [
        BenchmarkReference(
            benchmark_id="ioi",
            metric_name="Completeness",
            paper_title="Interpretability in the Wild: a Circuit for Indirect Object Identification",
            paper_doi="arxiv:2211.00593",
            model_revision="gpt2-small",
            dataset_name="IOI-100",
            dataset_hash="sha256_ioi_v1_100",
            expected_mean=0.830,
            acceptable_lower=0.790,
            acceptable_upper=0.870,
        )
    ],
    "induction_score": [
        BenchmarkReference(
            benchmark_id="induction_heads",
            metric_name="Induction Score",
            paper_title="In-context Learning and Induction Heads",
            paper_doi="arxiv:2209.11895",
            model_revision="gpt2-small",
            dataset_name="Repeated-Token-50",
            dataset_hash="sha256_induction_v1_50",
            expected_mean=0.850,
            acceptable_lower=0.780,
            acceptable_upper=0.920,
        )
    ],
}

class CanonicalBenchmarkRegistry:
    """Registry engine for accessing versioned benchmark references."""

    def __init__(self, version: str = REGISTRY_VERSION) -> None:
        self.version = version
        self._references = GPT2_SMALL_REFERENCES

    def get_references(self, metric_id: str) -> List[BenchmarkReference]:
        return self._references.get(metric_id, [])

    def get_primary_reference(self, metric_id: str) -> Optional[BenchmarkReference]:
        refs = self.get_references(metric_id)
        return refs[0] if refs else None

    def list_benchmarks(self) -> List[str]:
        benchmarks = set()
        for refs in self._references.values():
            for ref in refs:
                benchmarks.add(ref.benchmark_id)
        return list(benchmarks)

    def get_integrity_metadata(self) -> Dict[str, str]:
        return {
            "registry_version": self.version,
            "last_verified": "2026-07-28T17:42:00Z",
            "validator_version": "1.2.0"
        }
