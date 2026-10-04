"""`backend.science` -- statistics, validation and reproducibility.

`BenchmarkingOrchestrator` used to be exported here. It was removed rather than
rewired, for reasons that are worth recording so the deletion is not mistaken for
an oversight and re-reversed.

It looked like a working campaign runner. It could not run at all, and if it
could have, it would have corrupted the provenance of the file it wrote to.

Dead on arrival
---------------
`run_full_campaign` imported four classes from `backend.reproductions`:

    from ..reproductions.ioi_reproduction import IOIReproduction
    from ..reproductions.induction_heads import InductionHeadsReproduction
    from ..reproductions.sae_reproduction import SparseAutoencoderReproduction
    from ..reproductions.acdc_reproduction import ACDCReproduction

`backend/reproductions/` does not exist, and none of those four classes exist
anywhere in the repository. The imports were function-local, so
`import backend.science` and `import run_benchmarks` both succeeded -- the
failure only surfaced when someone actually ran the campaign, as
`ModuleNotFoundError` on the first line of the loop. An import that cannot
resolve is not an entry point.

Unsound even if the imports had resolved
----------------------------------------
`_record_result` appended entries shaped like:

    {"timestamp": ..., "model": ..., "benchmark": ...,
     "reproduction_successful": ..., "quality_score": ...,
     "p_value": ..., "effect_size": ...}

into `backend/science/benchmark_database.json` -- a file in which *every*
existing record is explicitly marked:

    "fixture": true, "measured": false, "provenance": "reference",
    "validation_eligible": false, "publication_eligible": false,
    "fixture_notice": "HAND-WRITTEN FIXTURE, NOT A RESULT."

The appended records carried none of those markers. So a run that measured
nothing, or measured something and crashed halfway, would have left entries in a
reference file that a reader -- or the evidence policy -- could not distinguish
from the hand-typed fixtures around them. That is the specific failure this
project exists to prevent, sitting in the one place that writes benchmark data.

Its model list was also fiction: `["gpt2-small", "gpt2-medium", "gemma-2b",
"llama-3-8b", "qwen-7b"]`. Only `gpt2` is in the local HuggingFace cache, and
the loop would have attempted all five regardless, appending a row per
(attempted) model per benchmark.

Where the work actually lives
----------------------------
`backend/science/reproducibility/BenchmarkRunner` runs the six real pipelines
(`ioi`, `induction_heads`, `greater_than`, `copy_task`, `arithmetic`,
`factual_recall`), distinguishes `NOT_RUN` from `ERROR`, and carries per-sample
traces forward instead of only aggregates. `run_benchmarks.py` at the repository
root is the entry point. `run_reproducibility_demo.py` is the loud-banner
demonstration of the artifact and signing path.
"""

from .statistics.statistical_validator import StatisticalValidator
from .statistics.statistical_protocol import StatisticalProtocol
from .phase_gatekeeper import PhaseGatekeeper
from .research_portal import ResearchPortal

__all__ = [
    "StatisticalValidator",
    "StatisticalProtocol",
    "PhaseGatekeeper",
    "ResearchPortal",
]
