"""Benchmark Tasks — All 8 canonical mechanistic interpretability benchmarks.

Each task defines:
  - Dataset construction
  - Algorithm to run
  - Published reference metrics (from papers)
  - Fidelity computation against the reference
"""

from __future__ import annotations

import math
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
    # What the interval bounds. May not be the primary score.
    confidence_interval_target: Optional[str] = None
    confidence_interval_derived: bool = False
    n_samples: int = 0
    #: How many independent seeds the score rests on.
    #:
    #: `n_samples` counts prompts; it says nothing about how many times the
    #: measurement was repeated. A result from one seed and a result pooled over
    #: five are very different claims, and `n_samples=100` looks identical in
    #: both -- which is how "a single 100-prompt draw is still one draw" stayed
    #: invisible. None means the seed count was not established, not that it
    #: was one.
    n_seeds: Optional[int] = None
    #: Which seeds produced the pooled value, when pooling happened.
    seeds: Optional[List[int]] = None
    #: The per-seed values behind a pooled score. Kept so a reader can see
    #: whether a tight interval came from seeds that agree or seeds that happen
    #: to cancel out; a mean alone reads the same either way.
    per_seed_values: Optional[List[float]] = None
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
    # What actually ran. `backend` is caller-supplied and was previously
    # ignored by every path, so a caller could not tell which code executed.
    backend_effective: Optional[str] = None
    # True when the score came from the fixture rather than a measurement.
    is_fixture: bool = False
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
            "backend_effective": self.backend_effective,
            "is_fixture": self.is_fixture,
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
            "n_seeds": self.n_seeds,
            "seeds": list(self.seeds) if self.seeds is not None else None,
            "per_seed_values": (list(self.per_seed_values)
                                if self.per_seed_values is not None else None),
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
            "confidence_interval_target": self.confidence_interval_target,
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
            outcome = {
                "score": self._fixture_score(task),
                "fixture": True,
                "patch_success_rate": None,
                "overlap_pct": None,
                "trials": None,
                "correct": None,
                "tokens": None,
                "backend_effective": "mock fixture",
            }
        else:
            # REFERENCE or PRODUCTION -- run the task's real pipeline.
            outcome = self._run_pipeline_dispatch(task, model_id, mode, n)

        score = outcome["score"]
        if score is None:
            raise RuntimeError(
                f"The {outcome.get('backend_effective')} pipeline returned no "
                f"primary score for {task.task_id!r}, so no result can be "
                f"reported. This is not replaced with a reference-derived "
                f"value."
            )
        patch_success = outcome.get("patch_success_rate")
        overlap = outcome.get("overlap_pct")
        is_fixture = bool(outcome.get("fixture"))
        # Cross-implementation agreement is never measured here; it used to be
        # a constant (0.98 / 0.96 / 0.97 / 0.99 depending on the path).
        tl_agree: Optional[float] = None
        self._tokens_counted = outcome.get("tokens")

        fidelity_pct = (1.0 - abs(score - ref_val) / max(ref_val, 1e-9)) * 100.0

        # Stop the profiler with a token count the pipeline actually produced,
        # so throughput and latency rest on a real denominator. It used to be
        # `n * 50` -- "estimated tokens = prompts * avg_seq_len".
        perf = profiler.stop(total_tokens=self._tokens_counted)
        runtime_s = perf.duration_s

        # A binomial interval needs the per-trial outcomes, which the pipelines
        # now return. `correct`/`trials` come from real boolean predictions;
        # the interval is computed from them and `ci_derived` records whether
        # they were available.
        correct = outcome.get("correct")
        trials = outcome.get("trials")
        ci_is_binomial_proportion = bool(
            not is_fixture
            and isinstance(correct, int)
            and isinstance(trials, int)
            and trials > 0
            and 0 <= correct <= trials
        )
        if ci_is_binomial_proportion:
            # Wilson score interval, from the pipeline's real correct/trials.
            #
            # The normal approximation was used before, on the *aggregate*
            # score rather than on trial counts, and it is unreliable exactly
            # where these benchmarks sit: at p=0.90 the approximation runs past
            # the boundary, and a single-copy control at p=0.0 yields a
            # zero-width interval that implies false precision. Wilson stays
            # inside [0,1] at both extremes.
            proportion = correct / trials
            z = 1.959963984540054  # 95%
            denominator = 1.0 + (z * z) / trials
            centre = proportion + (z * z) / (2 * trials)
            spread = z * math.sqrt(
                (proportion * (1 - proportion) / trials)
                + (z * z) / (4 * trials * trials)
            )
            ci_low = round(max(0.0, (centre - spread) / denominator) * 100, 2)
            ci_high = round(min(1.0, (centre + spread) / denominator) * 100, 2)
            half_width = round((ci_high - ci_low) / 2, 2)
            ci_method = (
                f"Wilson score interval from {correct}/{trials} observed "
                f"outcomes"
            )
            # Name the quantity the interval covers. For IOI this is the same
            # family as the primary score; for induction the primary score is a
            # mean attention fraction, while the interval covers the
            # behavioural accuracy. Leaving that unstated would present a
            # narrow interval as though it bounded the headline number.
            ci_target = outcome.get("ci_target") or "primary_score"
        else:
            half_width = None
            ci_low = None
            ci_high = None
            ci_method = None
            ci_target = None

        # Seed provenance. The Wilson interval above is computed from the
        # pipeline's per-prompt outcomes, which bounds within-draw precision.
        # Whether the score itself was repeated across seeds is a separate fact,
        # and it is recorded separately: `n_samples` counting prompts must not
        # stand in for it.
        seeds = outcome.get("seeds")
        n_seeds = outcome.get("n_seeds")
        per_seed_values = outcome.get("per_seed_values")
        if n_seeds is not None:
            try:
                n_seeds = int(n_seeds)
            except (TypeError, ValueError):
                n_seeds = None
            # An interval over a single seed cannot be a seed-level interval, so
            # say so on the interval rather than leaving `derived` implying one.
            if n_seeds == 1 and ci_is_binomial_proportion:
                ci_method += (
                    f". NOTE: from one seed only (n_seeds=1), so this bounds "
                    f"precision within that draw and says nothing about "
                    f"seed-to-seed variation."
                )

        return BenchmarkResult(
            task_id=task.task_id,
            model_id=model_id,
            backend=backend,
            backend_effective=outcome.get("backend_effective"),
            is_fixture=is_fixture,
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
            confidence_interval_target=ci_target,
            confidence_interval_derived=ci_is_binomial_proportion,
            n_samples=n,
            n_seeds=n_seeds,
            seeds=seeds,
            per_seed_values=per_seed_values,
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
    ) -> Dict[str, Any]:
        """Run the task's real reproduction pipeline.

        Returns the measured primary score plus whatever the pipeline actually
        reported. Raises for a task with no implemented pipeline rather than
        substituting a value: six of the eight catalogue tasks have no
        measurement implementation, and inventing one would be exactly the
        failure this file was rewritten to remove.

        `backend` is intentionally not consulted. There is one measurement path
        -- the reproduction pipelines -- and they all run the same model. The
        parameter is retained in the signature because it is part of the public
        API and is recorded in the result, but a caller cannot select between
        it. `backend_effective` on the result says what actually ran.
        """
        mock_mode = (mode == ExecutionMode.MOCK)

        if task.task_id == BenchmarkTask.IOI:
            from backend.science.reproducibility.ioi_pipeline import (
                IOIReproductionPipeline,
            )
            result = IOIReproductionPipeline(mock_mode=mock_mode).run(
                n_prompts=n)
            if mock_mode:
                return {
                    "score": self._fixture_score(task),
                    "fixture": True,
                    "patch_success_rate": None,
                    "overlap_pct": None,
                    "trials": None,
                    "correct": None,
                    "tokens": None,
                    "backend_effective": "mock fixture",
                }

            metrics = result["observed_metrics"]
            traces = result.get("raw_traces") or []
            # patch_success is a per-prompt boolean produced by the pipeline,
            # so it is a real binomial sample and the interval can be computed
            # from it rather than from the aggregate score.
            successes = [
                t["patch_success"] for t in traces
                if isinstance(t, dict) and isinstance(t.get("patch_success"), bool)
            ]
            return {
                "score": metrics.get("circuit_faithfulness"),
                "fixture": False,
                "patch_success_rate": metrics.get("patch_success_rate"),
                "overlap_pct": None,
                "trials": len(successes),
                "correct": sum(1 for s in successes if s),
                "tokens": self._count_ioi_tokens(traces),
                "backend_effective": "IOIReproductionPipeline",
            }

        if task.task_id == BenchmarkTask.INDUCTION_HEADS:
            from backend.science.reproducibility.induction_heads_pipeline import (
                InductionHeadsPipeline,
            )
            if mock_mode:
                return {
                    "score": self._fixture_score(task),
                    "fixture": True,
                    "patch_success_rate": None,
                    "overlap_pct": None,
                    "trials": None,
                    "correct": None,
                    "tokens": None,
                    "backend_effective": "mock fixture",
                }

            result = InductionHeadsPipeline(mock_mode=False).run(
                n_sequences=n, seq_len=8)
            metrics = result["observed_metrics"]
            behaviour = result.get("behaviour") or {}
            trials = behaviour.get("repeated_block_n")
            correct = behaviour.get("repeated_block_correct")
            return {
                "score": metrics.get("induction_score"),
                "fixture": False,
                "patch_success_rate": None,
                "overlap_pct": metrics.get("published_overlap_pct"),
                "trials": trials,
                "correct": correct,
                "tokens": self._count_induction_tokens(n),
                "backend_effective": "InductionHeadsPipeline",
                # The behavioural accuracy is a real proportion too, and it is
                # the metric with a meaningful number of trials.
                "behaviour_accuracy": behaviour.get("repeated_block_accuracy"),
                "behaviour_trials": trials,
                "behaviour_correct": correct,
                # The primary score here is a mean attention fraction, not a
                # proportion of trials, so the interval covers the behavioural
                # accuracy instead. Saying so prevents a narrow interval being
                # read as a bound on the headline number.
                "ci_target": "behavioural_accuracy",
                "mechanism": (result.get("mechanism") or {}).get("mechanism"),
            }

        raise NotImplementedError(
            f"No measurement pipeline is implemented for task "
            f"{task.task_id!r}. Tasks with a real implementation: IOI "
            f"(IOIReproductionPipeline), INDUCTION_HEADS "
            f"(InductionHeadsPipeline). This previously fell through to a "
            f"stub returning the published reference value plus noise, which "
            f"reproduces the paper by construction."
        )

    def _fixture_score(self, task: BenchmarkTaskSpec) -> float:
        """A value centred on the published reference. Explicitly a fixture."""
        return max(0.0, min(1.0, task.reference.metric_value
                            + self._rng.gauss(0, 0.015)))

    @staticmethod
    def _count_ioi_tokens(traces: List[Dict[str, Any]]) -> Optional[int]:
        """Real token count for the IOI prompts, or None if unavailable."""
        try:
            from backend.services import gpt2_engine
            tokenizer = getattr(gpt2_engine, "_tokenizer", None)
            if tokenizer is None:
                return None
            prompts = [t.get("prompt") for t in traces
                       if isinstance(t, dict) and t.get("prompt")]
            if not prompts:
                return None
            return sum(
                len(tokenizer.encode(p)) for p in prompts
            )
        except Exception:
            return None

    @staticmethod
    def _count_induction_tokens(n_sequences: int) -> Optional[int]:
        """Token count implied by the sequences the pipeline actually built."""
        # 8 token block + 1 separator + the repeated block, per sequence. The
        # block length is a caller-visible parameter of the pipeline, not a
        # guess at average prompt length.
        try:
            return int(n_sequences) * (8 + 1 + 8)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ #
    # Backend implementations                                              #
    # ------------------------------------------------------------------ #

