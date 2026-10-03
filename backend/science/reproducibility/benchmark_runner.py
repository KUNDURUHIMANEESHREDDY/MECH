"""Continuous Benchmark Runner.

Orchestrates the continuous integration suite that automatically evaluates
landmark interpretability tasks to detect regressions.
"""

from typing import Any, Dict, List
import datetime as _dt
import inspect
import time
import traceback

import torch

from .reproducibility_report import ReproducibilityReportEngine
from ..models.adapter_base import LiveUnavailable
from .ioi_pipeline import IOIReproductionPipeline
from .induction_heads_pipeline import InductionHeadsPipeline
from .greater_than_pipeline import GreaterThanCircuitPipeline
from .copy_task_pipeline import CopyTaskPipeline
from .arithmetic_pipeline import ArithmeticPipeline
from .factual_recall_pipeline import FactualRecallPipeline
from ..models.model_manager import ModelManager

class BenchmarkRunner:
    """Runner for executing all benchmark reproduction pipelines."""
    
    def __init__(self, mock_mode: bool = False) -> None:
        self.model_manager = ModelManager()
        self.report_engine = ReproducibilityReportEngine()

        # Was hardcoded `True`, so every call to `run_all` built its pipelines in
        # mock mode and could only ever produce fixtures -- while presenting them
        # through the same report path as real measurements, under a class
        # documented as "the continuous integration suite". Defaulting to False
        # matches every pipeline in this package and lets a caller opt into
        # fixtures explicitly, which is the direction of travel: fixtures are for
        # tests, not for the runner that claims to validate.
        self.mock_mode = mock_mode

        self.pipelines = {
            "ioi": IOIReproductionPipeline(mock_mode=self.mock_mode),
            "induction_heads": InductionHeadsPipeline(mock_mode=self.mock_mode),
            "greater_than": GreaterThanCircuitPipeline(mock_mode=self.mock_mode),
            "copy_task": CopyTaskPipeline(self.model_manager),
            "arithmetic": ArithmeticPipeline(self.model_manager),
            "factual_recall": FactualRecallPipeline(self.model_manager),
        }

    def run_all(self, model_id: str = "gpt2-small", seed: int = 42) -> Dict[str, Any]:
        """Run all pipelines and generate comprehensive reproducibility reports."""
        results = {}
        for paper_id, pipeline in self.pipelines.items():
            print(f"Running benchmark: {paper_id}")
            start_time = time.time()
            peak_vram = None
            # A fixture is not a pass. `status = "PASS"` was the default for any
            # run that did not raise, so a mock-mode run -- which by definition
            # measured nothing -- reported PASS. Callers reading this dict could
            # not tell a completed measurement from a completed fabrication.
            status = "FIXTURE" if self.mock_mode else "PASS"
            error_msg = None
            
            try:
                # Some pipelines might take model_id, IOIReproductionPipeline currently has it hardcoded inside adapter
                if paper_id == "ioi":
                    results_dict = pipeline.run(n_prompts=10, seed=seed)
                    metrics = results_dict.get("observed_metrics", {})
                else:
                    # Inspect the signature instead of calling and catching
                    # TypeError. The previous `except TypeError` fallback was
                    # both wrong and dangerous: it converted a genuine TypeError
                    # raised *inside* a measurement into a second attempt with a
                    # different signature, so a real bug could be reported as a
                    # signature mismatch and silently retried.
                    params = inspect.signature(pipeline.run).parameters
                    if "seed" in params:
                        results_dict = pipeline.run(seed=seed)
                    elif "model_id" in params:
                        results_dict = pipeline.run(model_id=model_id)
                    else:
                        results_dict = pipeline.run()

                    metrics = results_dict.get("observed_metrics", {}) if isinstance(results_dict, dict) else results_dict

                # Peak VRAM was previously `6.7 if "gpt2" in model_id else
                # 12.4` -- a constant chosen by model name and stored under a key
                # called peak_vram_gb, i.e. presented as an observed quantity.
                # Either it is measured with torch.cuda.max_memory_allocated or
                # it is absent; a made-up number is worse than None.
                #
                # In mock mode nothing ran on the GPU, so the honest value is
                # None, not the 0.0 that max_memory_allocated reports. Emitting
                # 0.0 would read as "measured, and the model needs no memory".
                if self.mock_mode or not torch.cuda.is_available():
                    peak_vram = None
                else:
                    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
                    torch.cuda.reset_peak_memory_stats()

                report = self.report_engine.generate_report(
                    paper_id=paper_id,
                    pipeline_name=pipeline.__class__.__name__,
                    model_id=model_id,
                    # Was the literal "mock_dataset_manifest_v1" -- a fake
                    # dataset identity stamped onto every generated report. A
                    # manifest id is provenance; inventing one here meant each
                    # report claimed to be traceable to a dataset that never
                    # existed.
                    dataset_manifest_id=None,
                    observed_metrics=metrics,
                    # Was ["Milestone A Validation Run"], a fixed narrative that
                    # said nothing about what was actually measured and implied
                    # a validation that had not happened.
                    explanation_of_diffs=[],
                )

            except LiveUnavailable as e:
                # Not a crash: the benchmark has no implementation, or the model
                # does not perform the task. Labelling this ERROR conflated "we
                # did not measure this" with "the measurement broke", which is
                # the same distinction the validation scheduler draws between
                # NOT_RUN and a regression. Four of the six pipelines here now
                # raise by design, so the distinction is load-bearing.
                print(f"Not measured ({paper_id}): {e}")
                status = "NOT_RUN"
                error_msg = str(e)
                metrics = {}
                report = {}
                peak_vram = None
            except Exception as e:
                print(f"Error running {paper_id}: {e}")
                status = "ERROR"
                error_msg = traceback.format_exc()
                metrics = {}
                report = {}
                peak_vram = None
                
            end_time = time.time()
            
            results[paper_id] = {
                "status": status,
                "error": error_msg,
                "runtime_sec": end_time - start_time,
                "peak_vram_gb": peak_vram,
                "metrics": metrics,
                "report": report
            }
        
        return {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "model_id": model_id,
            "reports": results
        }
