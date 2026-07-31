"""Science reproducibility package."""

from .reproducibility_report import MetricResult, ReproducibilityReport, ReproducibilityReportEngine
from .benchmark_certificate import BenchmarkCertificate, CertificateEngine
from .model_fingerprint import ModelFingerprint, ModelFingerprintEngine
from .scientific_validator import ValidationStats, ScientificValidator
from .session_recorder import ScientificSessionRecorder
from .validation_history import ValidationHistoryLogger
from .research_manifest import ResearchManifest, ResearchManifestEngine
from .research_snapshot import ResearchSnapshot, ResearchSnapshotEngine
from .paper_registry import ExpectedMetric, Paper, BenchmarkRegistry
from .audit_logger import AuditEntry, ScientificAuditLogger
from .dataset_versioning import DatasetVersionManifest, DatasetVersioningEngine
from .canonical_circuit_registry import CircuitReference, CanonicalCircuitRegistry
from .benchmark_reference_registry import BenchmarkReference, CanonicalBenchmarkRegistry
from .benchmark_runner import BenchmarkRunner
from .induction_heads_pipeline import InductionHeadsPipeline
from .ioi_pipeline import IOIReproductionPipeline
from .logit_lens_pipeline import LogitLensPipeline
from .greater_than_pipeline import GreaterThanCircuitPipeline
from .factual_recall_pipeline import FactualRecallPipeline
from .arithmetic_pipeline import ArithmeticPipeline
from .sae_pipeline import SAEReproductionPipeline
from .copy_task_pipeline import CopyTaskPipeline

# Backward-compatible aliases
IOIPipeline = IOIReproductionPipeline
GreaterThanPipeline = GreaterThanCircuitPipeline
SAEPipeline = SAEReproductionPipeline

__all__ = [
    "MetricResult", "ReproducibilityReport", "ReproducibilityReportEngine",
    "BenchmarkCertificate", "CertificateEngine",
    "ModelFingerprint", "ModelFingerprintEngine",
    "ValidationStats", "ScientificValidator",
    "ScientificSessionRecorder",
    "ValidationHistoryLogger",
    "ResearchManifest", "ResearchManifestEngine",
    "ResearchSnapshot", "ResearchSnapshotEngine",
    "ExpectedMetric", "Paper", "BenchmarkRegistry",
    "AuditEntry", "ScientificAuditLogger",
    "DatasetVersionManifest", "DatasetVersioningEngine",
    "CircuitReference", "CanonicalCircuitRegistry",
    "BenchmarkReference", "CanonicalBenchmarkRegistry",
    "BenchmarkRunner",
    "InductionHeadsPipeline", "IOIReproductionPipeline", "LogitLensPipeline",
    "GreaterThanCircuitPipeline", "FactualRecallPipeline", "ArithmeticPipeline",
    "SAEReproductionPipeline", "CopyTaskPipeline",
    "IOIPipeline", "GreaterThanPipeline", "SAEPipeline",
]
