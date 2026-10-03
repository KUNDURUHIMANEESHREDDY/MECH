"""Benchmark Tasks — All 8 canonical mechanistic interpretability benchmarks.

Each task defines:
  - Dataset construction
  - Algorithm to run
  - Published reference metrics (from papers)
  - Fidelity computation against the reference
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class BenchmarkTask(str, Enum):
    IOI             = "ioi"
    INDUCTION_HEADS = "induction_heads"
    GREATER_THAN    = "greater_than"
    LOGIT_LENS      = "logit_lens"
    SAE             = "sae"
    COPY_TASK       = "copy_task"
    ARITHMETIC      = "arithmetic"
    FACTUAL_RECALL  = "factual_recall"
    UNIVERSALITY    = "universality"


class ExecutionMode(str, Enum):
    MOCK      = "mock"      # Unit tests, no models, synthetic data
    REFERENCE = "reference" # Real GPT-2 Small, causal patching, CPU/GPU
    PRODUCTION = "production" # Full distributed execution, multiple models


@dataclass
class PublishedReference:
    """Ground-truth metrics from the original paper."""
    source_paper: str
    doi_or_arxiv: str
    metric_name: str
    metric_value: float
    metric_unit: str
    model_family: str           # paper's model family
    algorithm: str


@dataclass
class BenchmarkTaskSpec:
    task_id: BenchmarkTask
    name: str
    description: str
    algorithm: str              # Which interpretability algorithm is used
    dataset_size: int           # Number of prompts
    reference: PublishedReference
    # Metrics to compute
    primary_metric: str         # e.g. "circuit_recovery_fidelity"
    secondary_metrics: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    task_id: BenchmarkTask
    model_id: str
    backend: str
    mode: ExecutionMode
    # Core metrics
    primary_score: float        # 0–100 %
    reference_score: float      # Published reference value
    fidelity_pct: float         # agreement with published metric (0–100)
    # Resource metrics. Optional: these were random draws in every mode, and a
    # resource figure that is invented reads as a measured cost.
    runtime_s: float
    peak_memory_mb: Optional[float]
    # Statistical
    #
    # The interval fields were built as `score*100 +/- half_width` with a fixed
    # half_width, which is a spread around a point estimate rather than an
    # interval estimated from the samples. They now carry how wide they are and
    # whether they were derived from the data at all.
    confidence_interval_low: Optional[float]
    confidence_interval_high: Optional[float]
    confidence_interval_half_width: Optional[float] = None
    confidence_interval_method: Optional[str] = None
    confidence_interval_derived: bool = False
    n_samples: int = 0
    # High-Fidelity Performance Metrics (Phase 39.14)
    # Profiler readings. 0.0 is ambiguous between "measured and zero" and
    # "never measured", so these are Optional and the caller checks.
    flops: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    throughput_tps: Optional[float] = None
    mean_latency_ms: Optional[float] = None
    # Optional/Default Resource metrics
    peak_vram_mb: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    resource_metrics_measured: bool = False
    # Execution Metadata
    git_sha: str = "unknown"
    device_info: str = "cpu"
    precision: str = "fp32"
    # Optional: a count that is not counted must not be a specific integer.
    forward_passes: Optional[int] = None
    patch_count: Optional[int] = None
    patch_count_measured: bool = False
    # 0.0 here means "no patching was run", not "patching never succeeds".
    patch_success_rate: Optional[float] = None
    published_overlap_pct: Optional[float] = None
    # Comparison
    tl_agreement_pct: Optional[float] = None
    sae_lens_agreement_pct: Optional[float] = None
    sae_lens_agreement_measured: bool = False
    sae_lens_agreement_reason: Optional[str] = None
    # Metadata
    notes: str = ""
    run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id.value,
            "model_id": self.model_id,
            "backend": self.backend,
            "mode": self.mode.value,
            "primary_score": round(self.primary_score, 2),
            "reference_score": round(self.reference_score, 2),
            "fidelity_pct": round(self.fidelity_pct, 2),
            "runtime_s": round(self.runtime_s, 3),
            # Rounded only when present; round(None) raises, and a field that
            # is None must stay None through serialisation.
            "peak_memory_mb": (round(self.peak_memory_mb, 1)
                               if self.peak_memory_mb is not None else None),
            "peak_vram_mb": (round(self.peak_vram_mb, 1)
                             if self.peak_vram_mb is not None else None),
            "tokens_per_sec": (round(self.tokens_per_sec, 2)
                               if self.tokens_per_sec is not None else None),
            "resource_metrics_measured": self.resource_metrics_measured,
            "flops": self.flops,
            "gpu_util_pct": self.gpu_util_pct,
            "throughput_tps": self.throughput_tps,
            "mean_latency_ms": self.mean_latency_ms,
            "profiler_measured": self.throughput_tps is not None,
            "ci_low": (round(self.confidence_interval_low, 2)
                       if self.confidence_interval_low is not None else None),
            "ci_high": (round(self.confidence_interval_high, 2)
                        if self.confidence_interval_high is not None else None),
            "n_samples": self.n_samples,
            "git_sha": self.git_sha,
            "device_info": self.device_info,
            "precision": self.precision,
            "forward_passes": self.forward_passes,
            "patch_count": self.patch_count,
            "patch_count_measured": self.patch_count_measured,
            "patch_success_rate": (round(self.patch_success_rate, 2)
                                   if self.patch_success_rate is not None
                                   else None),
            "published_overlap_pct": self.published_overlap_pct,
            "tl_agreement_pct": self.tl_agreement_pct,
            "sae_lens_agreement_pct": self.sae_lens_agreement_pct,
            "sae_lens_agreement_measured": self.sae_lens_agreement_measured,
            "sae_lens_agreement_reason": self.sae_lens_agreement_reason,
            "confidence_interval_derived": self.confidence_interval_derived,
            "confidence_interval_method": self.confidence_interval_method,
            "notes": self.notes,
            "run_id": self.run_id,
        }


# ------------------------------------------------------------------ #
# Published reference metrics (sourced from original papers)          #
# ------------------------------------------------------------------ #

TASK_CATALOGUE: Dict[BenchmarkTask, BenchmarkTaskSpec] = {
    BenchmarkTask.IOI: BenchmarkTaskSpec(
        task_id=BenchmarkTask.IOI,
        name="Indirect Object Identification (IOI)",
        description=(
            "Identify heads responsible for indirect object identification in sentences "
            "of the form 'When Mary and John went to the store, John gave...'."
        ),
        algorithm="ACDC + Path Patching",
        dataset_size=100,
        reference=PublishedReference(
            source_paper="Wang et al. 2022 — Interpretability in the Wild",
            doi_or_arxiv="arxiv:2211.00593",
            metric_name="circuit_recovery_fidelity",
            metric_value=0.97,
            metric_unit="fraction",
            model_family="gpt2",
            algorithm="path_patching",
        ),
        primary_metric="circuit_recovery_fidelity",
        secondary_metrics=["node_count", "edge_count", "logit_diff"],
    ),
    BenchmarkTask.INDUCTION_HEADS: BenchmarkTaskSpec(
        task_id=BenchmarkTask.INDUCTION_HEADS,
        name="Induction Head Sequence Repeater",
        description=(
            "Detect induction heads that copy AB...A → B patterns. "
            "Measure the induction score across all attention heads."
        ),
        algorithm="Activation Patching",
        dataset_size=200,
        reference=PublishedReference(
            source_paper="Olsson et al. 2022 — In-context Learning and Induction Heads",
            doi_or_arxiv="arxiv:2209.11895",
            metric_name="induction_score",
            metric_value=0.95,
            metric_unit="fraction",
            model_family="gpt2",
            algorithm="activation_patching",
        ),
        primary_metric="induction_score",
        secondary_metrics=["head_agreement", "copy_accuracy"],
    ),
    BenchmarkTask.GREATER_THAN: BenchmarkTaskSpec(
        task_id=BenchmarkTask.GREATER_THAN,
        name="Greater-Than Numeric Comparison",
        description=(
            "Identify the circuit that computes 'The war lasted from 1942 to 19XX' "
            "where XX > 42 must be predicted."
        ),
        algorithm="ACDC",
        dataset_size=150,
        reference=PublishedReference(
            source_paper="Hanna et al. 2023 — How Does GPT-2 Compute Greater-Than?",
            doi_or_arxiv="arxiv:2305.00586",
            metric_name="circuit_faithfulness",
            metric_value=0.89,
            metric_unit="fraction",
            model_family="gpt2",
            algorithm="acdc",
        ),
        primary_metric="circuit_faithfulness",
        secondary_metrics=["mlp_attribution", "attention_attribution"],
    ),
    BenchmarkTask.LOGIT_LENS: BenchmarkTaskSpec(
        task_id=BenchmarkTask.LOGIT_LENS,
        name="Logit Lens Layer Attribution",
        description=(
            "Apply logit lens at every residual stream layer and measure "
            "prediction accuracy buildup across depth."
        ),
        algorithm="Logit Lens",
        dataset_size=300,
        reference=PublishedReference(
            source_paper="Nostalgebraist 2020 — Interpreting GPT: the logit lens",
            doi_or_arxiv="lesswrong.com/posts/AcKRB8wDpdaN6v6ru",
            metric_name="final_layer_acc",
            metric_value=0.82,
            metric_unit="fraction",
            model_family="gpt2",
            algorithm="logit_lens",
        ),
        primary_metric="final_layer_acc",
        secondary_metrics=["layer_by_layer_acc", "entropy_reduction"],
    ),
    BenchmarkTask.SAE: BenchmarkTaskSpec(
        task_id=BenchmarkTask.SAE,
        name="SAE Feature Dictionary Recovery",
        description=(
            "Train a Sparse Autoencoder on residual stream activations and measure "
            "reconstruction fidelity and feature monosemanticity."
        ),
        algorithm="Sparse Autoencoder (SAE)",
        dataset_size=500,
        reference=PublishedReference(
            source_paper="Cunningham et al. 2023 — Sparse Autoencoders Find Highly Interpretable Features",
            doi_or_arxiv="arxiv:2309.08600",
            metric_name="reconstruction_r2",
            metric_value=0.91,
            metric_unit="fraction",
            model_family="gpt2",
            algorithm="sae",
        ),
        primary_metric="reconstruction_r2",
        secondary_metrics=["l0_sparsity", "monosemanticity_score", "feature_density"],
    ),
    BenchmarkTask.COPY_TASK: BenchmarkTaskSpec(
        task_id=BenchmarkTask.COPY_TASK,
        name="Copy Task Token Induction",
        description=(
            "Measure how well the model copies a token sequence [A][B]...[A] → [B] "
            "and attribute the computation to specific heads."
        ),
        algorithm="Activation Patching",
        dataset_size=200,
        reference=PublishedReference(
            source_paper="Elhage et al. 2021 — A Mathematical Framework for Transformer Circuits",
            doi_or_arxiv="arxiv:2112.14628",
            metric_name="copy_fidelity",
            metric_value=0.93,
            metric_unit="fraction",
            model_family="gpt2",
            algorithm="activation_patching",
        ),
        primary_metric="copy_fidelity",
        secondary_metrics=["k_composition_score", "q_composition_score"],
    ),
    BenchmarkTask.ARITHMETIC: BenchmarkTaskSpec(
        task_id=BenchmarkTask.ARITHMETIC,
        name="Arithmetic Computation Circuit",
        description=(
            "Identify MLP layers and attention heads responsible for simple arithmetic "
            "(single-digit addition/subtraction) via causal intervention."
        ),
        algorithm="Causal Scrubbing",
        dataset_size=250,
        reference=PublishedReference(
            source_paper="Stolfo et al. 2023 — A Mechanistic Interpretation of Arithmetic in LLMs",
            doi_or_arxiv="arxiv:2305.02301",
            metric_name="arithmetic_faithfulness",
            metric_value=0.84,
            metric_unit="fraction",
            model_family="gpt2",
            algorithm="causal_scrubbing",
        ),
        primary_metric="arithmetic_faithfulness",
        secondary_metrics=["mlp_contribution", "head_contribution"],
    ),
    BenchmarkTask.FACTUAL_RECALL: BenchmarkTaskSpec(
        task_id=BenchmarkTask.FACTUAL_RECALL,
        name="Factual Recall Knowledge Attribution",
        description=(
            "Identify where factual associations (e.g. 'The Eiffel Tower is in...') "
            "are stored using causal mediation analysis."
        ),
        algorithm="Causal Mediation Analysis",
        dataset_size=200,
        reference=PublishedReference(
            source_paper="Meng et al. 2022 — Locating and Editing Factual Associations in GPT",
            doi_or_arxiv="arxiv:2202.05262",
            metric_name="causal_effect_score",
            metric_value=0.79,
            metric_unit="fraction",
            model_family="gpt2",
            algorithm="causal_mediation",
        ),
        primary_metric="causal_effect_score",
        secondary_metrics=["mlp_vs_attn_split", "layer_attribution"],
    ),
    BenchmarkTask.UNIVERSALITY: BenchmarkTaskSpec(
        task_id=BenchmarkTask.UNIVERSALITY,
        name="Cross-Model Feature Universality",
        description=(
            "Measure how well representations align across model architectures "
            "using bipartite matching of SAE features."
        ),
        algorithm="Feature Alignment / Bipartite Matching",
        dataset_size=100,
        reference=PublishedReference(
            source_paper="Chughtai et al. 2023 — A Toy Model of Universality",
            doi_or_arxiv="arxiv:2305.01126",
            metric_name="alignment_fidelity",
            metric_value=0.82,
            metric_unit="fraction",
            model_family="gpt2",
            algorithm="alignment",
        ),
        primary_metric="alignment_fidelity",
        secondary_metrics=["cluster_drift", "semantic_overlap"],
    ),
}


class BenchmarkTaskExecutor:
    """
    Executes a single benchmark task on a model.

    When a real model backend is available (TransformerLens / HuggingFace / Ollama),
    it runs the actual algorithm. Otherwise it uses a deterministic stub that produces
    realistic variance around the published reference metric.
    """

    def __init__(self, rng_seed: int = 42) -> None:
        self._rng = random.Random(rng_seed)

    def execute(
        self,
        task: BenchmarkTaskSpec,
        model_id: str,
        backend: str,
        mode: ExecutionMode = ExecutionMode.MOCK,
        n_samples: Optional[int] = None,
        run_id: str = "",
    ) -> BenchmarkResult:
        from backend.runtime.performance_profiler import PerformanceProfiler

        # Initialize Profiler
        # Parameter counts were hardcoded by a substring test on the model id:
        # anything containing "small" got 0.124B, everything else 0.355B. So
        # gpt2-medium, pythia-1.4b and llama-7b all reported the same FLOPs.
        # Now read from the registry, and refuse rather than guess.
        n_params_b = self._resolve_param_count(model_id)
        if n_params_b is None:
            raise ValueError(
                f"Parameter count for '{model_id}' is unknown, so FLOPs "
                "cannot be estimated. It was previously inferred from whether "
                "the id contained 'small', which gave every non-small model the "
                "same figure."
            )
        profiler = PerformanceProfiler(model_id, n_params_b=n_params_b)
        profiler.start()
        # Token counts are only known if a pipeline reports them. None does yet,
        # so this stays None and the profiler reports no throughput.
        self._tokens_counted: Optional[int] = None

        n = n_samples or task.dataset_size
        ref_val = task.reference.metric_value

        # Simulation/Execution logic based on mode
        # Resource figures.
        #
        # These were all random draws in *every* mode, including the real ones:
        # peak memory gauss(3400, 400), vram 1800.0, tokens/sec gauss(150, 20).
        # A benchmark whose resource column is noise looks like a benchmark
        # that measured resources. None of them are read from the process, so
        # they are now None and the profiler's real readings are used when it
        # has any.
        peak_mem = None
        vram = None
        tokens_sec = None

        if mode == ExecutionMode.MOCK:
            # A fixture, and labelled as one: centred on the published
            # reference so downstream code has something to exercise.
            noise = self._rng.gauss(0, 0.015)
            score = max(0.0, min(1.0, ref_val + noise))
            tl_agree = max(0.85, min(1.0, 0.97 + self._rng.gauss(0, 0.02)))
            patch_success = 0.0
            overlap = None
        else:
            # REFERENCE or PRODUCTION — run the real pipeline.
            score, patch_success, overlap, tl_agree = self._run_pipeline_dispatch(
                task, model_id, mode, n)

        fidelity_pct = (1.0 - abs(score - ref_val) / max(ref_val, 1e-9)) * 100.0

        # Stop the profiler. The caller previously passed `n * 50` -- "estimated
        # tokens = prompts * avg_seq_len" -- so throughput and latency were
        # divided by a token count nobody counted. Without a real count the
        # profiler reports None for both, which is the honest answer.
        perf = profiler.stop(total_tokens=self._tokens_counted)
        runtime_s = perf.duration_s

        # A binomial 95% interval is only valid when `score` is a proportion of
        # n independent binary outcomes. Here `score` is a single aggregate
        # metric returned by a pipeline, so the per-sample outcomes that would
        # justify the formula are not available at this level. The formula was
        # still applied and the result reported as a 95% CI.
        #
        # The condition is now stated instead of assumed: an interval is only
        # emitted for a metric that actually is a proportion of n Bernoulli
        # trials, and `ci_derived` records whether that held.
        ci_is_binomial_proportion = bool(
            ref_val is not None and 0.0 <= score <= 1.0 and n > 0
            and task.primary_metric.endswith(("_rate", "_accuracy", "success_rate"))
        )
        if ci_is_binomial_proportion:
            half_width = 1.96 * (score * (1 - score) / n) ** 0.5 * 100
            ci_low = round(score * 100 - half_width, 2)
            ci_high = round(score * 100 + half_width, 2)
            ci_method = "normal approximation to the binomial proportion"
        else:
            half_width = None
            ci_low = None
            ci_high = None
            ci_method = (
                None
            )

        return BenchmarkResult(
            task_id=task.task_id,
            model_id=model_id,
            backend=backend,
            mode=mode,
            primary_score=round(score * 100, 2),
            reference_score=round(ref_val * 100, 2),
            fidelity_pct=round(fidelity_pct, 2),
            runtime_s=round(runtime_s, 4),
            # `perf.X or abs(vram)` also substituted the random draw whenever
            # the profiler reported 0.0, so a real zero became a fake number.
            peak_memory_mb=(round(abs(peak_mem), 1) if peak_mem is not None
                            else None),
            peak_vram_mb=(round(perf.peak_vram_mb, 1)
                          if perf.peak_vram_mb else None),
            tokens_per_sec=(round(perf.throughput_tps, 2)
                            if perf.throughput_tps else None),
            resource_metrics_measured=bool(perf.throughput_tps),
            flops=perf.total_flops,
            gpu_util_pct=perf.gpu_util_pct,
            throughput_tps=perf.throughput_tps,
            mean_latency_ms=perf.mean_latency_ms,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            confidence_interval_half_width=(round(half_width, 2)
                                            if half_width is not None else None),
            confidence_interval_method=ci_method,
            confidence_interval_derived=ci_is_binomial_proportion,
            n_samples=n,
            tl_agreement_pct=round(tl_agree * 100, 2) if tl_agree else None,
            # Was `self._rng.gauss(0.95, 0.02) * 100` -- a random draw reported
            # as agreement between two lens implementations. Nothing was
            # compared with anything.
            sae_lens_agreement_pct=None,
            sae_lens_agreement_measured=False,
            sae_lens_agreement_reason=(
                "Comparing an SAE-based and a raw LogitLens projection requires "
                "both projections over the same prompt and tokens. No such "
                "comparison is performed here, so no agreement can be reported."
            ),
            notes=f"Backend: {backend} | Mode: {mode.value}",
            run_id=run_id,
            # Was `n * 2` commented "Rough estimate". A forward-pass count is a
            # real cost figure; multiplying the prompt count by a constant and
            # calling it rough still produces a specific integer that reads as
            # counted. It was not counted, so it is now None.
            forward_passes=None,
            # Was `n if "patching" in task.algorithm.lower() else 0`: a patch
            # count inferred from the *name* of the algorithm, with no patching
            # performed. The count was not counted.
            patch_count=None,
            patch_count_measured=False,
            patch_success_rate=patch_success,
            published_overlap_pct=overlap,
        )

    @staticmethod
    def _resolve_param_count(model_id: str) -> Optional[float]:
        """Parameter count in billions, from the model registry, or None.

        The registry keys on ModelFamily, so a family must be named explicitly.
        Guessing the family from the model id is the substring matching this
        replaced, so callers pass the family rather than have it inferred.
        """
        try:
            # Absolute import: `..` fails when this module is loaded as
            # top-level `benchmarking.benchmark_tasks` (which is how pytest and
            # several callers reach it), and the ImportError was being swallowed
            # into a silent None.
            from backend.benchmarking.model_registry import MODEL_CATALOGUE
        except Exception:
            return None
        # ModelFamily is a str-enum whose value is the canonical model id, so
        # an exact match on that value is not substring guessing. Callers pass
        # whatever id they hold; common shorthands are normalised because they
        # are unambiguous rather than heuristic.
        needle = (model_id or "").strip().lower()
        aliases = {"gpt2-small": "gpt2", "gpt-2": "gpt2", "gpt2_small": "gpt2"}
        needle = aliases.get(needle, needle)
        for family, spec in MODEL_CATALOGUE.items():
            if str(family.value).lower() == needle:
                return spec.n_params_b
        return None

    def _run_pipeline_dispatch(
        self, task: BenchmarkTaskSpec, model_id: str, mode: ExecutionMode, n: int
    ) -> Tuple[float, float, Optional[float], Optional[float]]:
        """Dispatch to high-fidelity pipelines in backend/science/reproducibility/.

        The fourth element is cross-implementation agreement, which is Optional:
        no pipeline here is cross-checked against a second implementation, so
        it is None rather than a flattering constant.
        """
        mock_mode = (mode == ExecutionMode.MOCK)

        if task.task_id == BenchmarkTask.IOI:
            from backend.science.reproducibility.ioi_pipeline import IOIReproductionPipeline
            pipe = IOIReproductionPipeline(mock_mode=mock_mode)
            res = pipe.run(n_prompts=n)
            metrics = res["observed_metrics"]
            # Was 0.98, a constant, reported as transformer-lens agreement with
            # the pipeline it had just run. The pipeline computes its own
            # fidelity; it does not cross-check against a second
            # implementation. Agreement between two methods is a separate
            # measurement and none was made.
            return (metrics["circuit_faithfulness"],
                    metrics["patch_success_rate"], None, None)

        if task.task_id == BenchmarkTask.INDUCTION_HEADS:
            from backend.science.reproducibility.induction_heads_pipeline import InductionHeadsPipeline
            pipe = InductionHeadsPipeline(mock_mode=mock_mode)
            res = pipe.run(n_sequences=n)
            metrics = res["observed_metrics"]
            # Was 0.96, for the same reason as the IOI case above.
            return (metrics["induction_score"], 0.0,
                    metrics["published_overlap_pct"], None)

        # Fallback for other tasks not yet fully pipelined
        score, tl = self._run_real(task, model_id, "fallback", n)
        return score, 0.0, None, tl

    # ------------------------------------------------------------------ #
    # Backend implementations                                              #
    # ------------------------------------------------------------------ #

    def _run_real(
        self, task: BenchmarkTaskSpec, model_id: str, backend: str, n: int
    ) -> Tuple[float, Optional[float]]:
        """Attempt real execution; fall back to stub on import/runtime error."""
        try:
            if backend == "transformerlens":
                return self._run_tl(task, model_id, n)
            return self._run_hf(task, model_id, n)
        except Exception as exc:  # noqa: BLE001
            # The real backend raised. Previously this logged a warning and
            # returned `ref + noise` -- the *published reference value* plus
            # random jitter, as a measurement. A reproduction that scores the
            # published number by construction reproduces nothing, and it did
            # so silently: the caller received a plausible score with no
            # indication the real path had failed.
            #
            # It now fails loudly. The caller decides whether to run in mock
            # mode, which is at least visibly a stub.
            logger.error("Real backend %s failed for %s: %s",
                         backend, task.task_id, exc)
            raise RuntimeError(
                f"Real benchmark backend '{backend}' failed for "
                f"{task.task_id}: {exc}. This will not be replaced with a "
                "synthetic score derived from the published reference value, "
                "because that measures nothing. Use ExecutionMode.MOCK "
                "explicitly if you want a fixture."
            ) from exc

    def _run_tl(
        self, task: BenchmarkTaskSpec, model_id: str, n: int
    ) -> Tuple[float, float]:
        import transformer_lens as tl  # noqa: F401
        # Imports the library, then returns the published reference value plus
        # jitter. No hook, no forward pass, no measurement. The import is what
        # made this look like a real execution path.
        raise NotImplementedError(
            "The transformer_lens backend is not implemented: importing the "
            "library is not running it. This previously returned "
            "reference.metric_value + noise, which reproduces the published "
            "number by construction."
        )

    def _run_hf(
        self, task: BenchmarkTaskSpec, model_id: str, n: int
    ) -> Tuple[float, float]:
        import transformers  # noqa: F401
        # Same problem as the transformer_lens path: import, then return the
        # reference value with noise.
        raise NotImplementedError(
            "The transformers backend is not implemented: importing the "
            "library is not running it. This previously returned "
            "reference.metric_value + noise."
        )

    def _run_ollama(
        self, task: BenchmarkTaskSpec, model_id: str, n: int
    ) -> Tuple[float, Optional[float]]:
        # The comment said fidelity is "estimated from token-level outputs",
        # which is not an estimate -- it is the reference value with noise.
        raise NotImplementedError(
            "The ollama backend is not implemented. Token-level outputs cannot "
            "substitute for the activation-level measurements these tasks "
            "require, so no score is returned."
        )
