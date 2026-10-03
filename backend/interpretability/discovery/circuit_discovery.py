"""Automated Circuit Discovery Engine."""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Dict, List, Optional

from backend.agents.evidence_policy import provenance_of
from backend.science.models.adapter_base import ModelAdapter
from .algorithms import get_algorithm
from backend.research_datasets.dataset_manager import DatasetManager


class CircuitDiscoveryEngine:
    """Discovers circuits using registered discovery algorithms.

    Returns an explicit "unavailable" record rather than a synthetic circuit when
    no measurement was possible. See `_unavailable` for what was removed.
    """

    def __init__(self, adapter: Optional[ModelAdapter] = None) -> None:
        self.adapter = adapter
        # Point at the research datasets directory relative to this file.
        # Renamed from "datasets" to "research_datasets": a backend/datasets/
        # package shadowed HuggingFace's `datasets` whenever backend/ was on
        # sys.path, which is how the desktop app launches the backend.
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.dataset_manager = DatasetManager(
            os.path.join(base_dir, "research_datasets"))

    def _unavailable(
        self,
        dataset_name: str,
        prompt_id: str,
        algorithm_name: str,
        reason: str,
        model_id: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """A record saying the circuit was not discovered, and why.

        This replaces a fallback that, whenever no adapter was connected,
        returned a complete synthetic circuit: a `Neuron` node labelled
        `L8_N402 (IOI)`, edges weighted 0.88 and 0.95 with confidences 0.94 and
        0.98, `confidence: 0.95`, `runtime_ms: 150.0` and `graph.score: 0.945` --
        which was the arithmetic mean of the two invented confidences, so even the
        summary statistic was derived from the fabrication rather than measured.
        `provenance` was `{}`, so nothing anywhere marked it as synthetic.

        That mattered because the shape was indistinguishable from a real result.
        A caller checking `if result.get("confidence")` got 0.95; a caller
        rendering `result["graph"]` drew a plausible IOI circuit with a named
        neuron in it. The `L8_N402` node is the sharpest edge: it is a specific
        claim about a specific neuron, invented.

        The record keeps the same keys as a real report -- `statistics`,
        `evidence`, `graph`, `artifacts` -- so consumers do not need to
        special-case it, but every score is None, the graph is empty, and
        provenance says "unavailable" in the form the evidence policy can read
        (a plain string; a dict is resolved via source/kind/type/status and would
        resolve to "unavailable" anyway, which is the right answer but by accident).
        """
        record: Dict[str, Any] = {
            "status": "unavailable",
            "algorithm": algorithm_name,
            "dataset_id": f"{dataset_name}_{prompt_id}",
            "model_id": model_id,
            "runtime_ms": None,
            "statistics": {},
            "evidence": {},
            "confidence": None,
            "graph": {"nodes": [], "edges": [], "score": None},
            "artifacts": [],
            # Plain string: see the note in _unavailable's docstring.
            "provenance": "unavailable",
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": reason,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        }
        record.update(extra)
        return record

    def discover_circuit(self, dataset_name: str = "ioi", prompt_id: str = "ioi_0001", algorithm_name: str = "acdc") -> Dict[str, Any]:
        """Discovers a circuit using the specified registered algorithm and dataset."""
        if not self.adapter:
            return self._unavailable(
                dataset_name, prompt_id, algorithm_name,
                reason=(
                    "No model adapter is connected, so no circuit was discovered. "
                    "This previously returned a synthetic circuit with a named "
                    "neuron, invented edge weights and confidences, and "
                    "confidence 0.95."
                ),
            )

        model_id = getattr(getattr(self.adapter, "spec", None), "model_id", None)

        try:
            prompts = self.dataset_manager.load(dataset_name)
        except Exception as exc:
            # Previously `{"error": str(exc)}` -- a bare dict that a caller
            # checking for `confidence` would treat as a result with no score,
            # rather than as a failure.
            return self._unavailable(
                dataset_name, prompt_id, algorithm_name, model_id=model_id,
                reason=f"Dataset '{dataset_name}' could not be loaded: "
                       f"{type(exc).__name__}: {exc}",
            )

        if not prompts:
            return self._unavailable(
                dataset_name, prompt_id, algorithm_name, model_id=model_id,
                reason=f"Dataset '{dataset_name}' is empty.",
            )

        dataset_item = next(
            (p for p in prompts if isinstance(p, dict) and p.get("id") == prompt_id),
            prompts[0],
        )

        try:
            engine = get_algorithm(algorithm_name, self.adapter)
        except Exception as exc:
            return self._unavailable(
                dataset_name, prompt_id, algorithm_name, model_id=model_id,
                reason=f"Algorithm '{algorithm_name}' is not registered: "
                       f"{type(exc).__name__}: {exc}",
            )

        try:
            report = engine.run(dataset=dataset_item)
        except Exception as exc:
            # A raising algorithm is a measurement that did not happen, not an
            # error to swallow into an unlabelled dict. The traceback goes in
            # `error` for debugging; `reason` is the honest summary.
            return self._unavailable(
                dataset_name, prompt_id, algorithm_name, model_id=model_id,
                reason=f"{algorithm_name} raised during discovery: "
                       f"{type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )

        result = report.to_dict()
        # Execution and evidence are different states and are reported separately.
        # `status` says whether the algorithm ran to completion; `measured` says
        # whether it took measurements from loaded weights. An earlier version set
        # `status` from `measured`, so a run that measured 144 heads and retained a
        # 113-node circuit reported "unavailable" purely because the whole-circuit
        # fidelity needs `io_id`/`subject_id`, which the bundled dataset omits.
        provenance = result.get("provenance")
        measured = bool(
            isinstance(provenance, dict) and provenance.get("measured")
        ) or provenance_of(result) == "live"
        result.setdefault("status", "completed")
        result.setdefault("measured", measured)
        # Eligibility stays False unless something explicitly opts in. A measured
        # sweep is not the same as a scientifically adequate one, and asserting
        # eligibility here would be the compressed claim the evidence policy is
        # meant to prevent.
        result.setdefault("validation_eligible", False)
        result.setdefault("publication_eligible", False)
        return result
