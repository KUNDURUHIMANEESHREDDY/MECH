"""Society Executor agent (capability: exec).

Runs models and experiments. All heavy work goes through
backend.services.gpt2_engine (thread-safe, live weights) via
asyncio.to_thread so FastAPI stays responsive — same pattern as
main.py::_preload_gpt2_engine.

Also wraps:
- backend.science.reproducibility.*_pipeline.IOIReproductionPipeline etc.
- backend.runtime.orchestration.execution_orchestrator.ExecutionOrchestrator

Legacy shim note: backend/gpt2_service.py should become thin wrappers over
gpt2_engine; this agent calls gpt2_engine directly (the survivor per the
audit). Every method degrades to {"status": "unavailable"|"error", ...}
when torch/transformers is absent — never raises, never hangs the UI.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

PIPELINE_MODULES = {
    "ioi": ("backend.science.reproducibility.ioi_pipeline", "IOIReproductionPipeline"),
    "induction_heads": (
        "backend.science.reproducibility.induction_heads_pipeline",
        "InductionHeadsPipeline",
    ),
    "greater_than": (
        "backend.science.reproducibility.greater_than_pipeline",
        "GreaterThanCircuitPipeline",
    ),
    "logit_lens": (
        "backend.science.reproducibility.logit_lens_pipeline",
        "LogitLensPipeline",
    ),
    "sparse_autoencoders": (
        "backend.science.reproducibility.sae_pipeline",
        "SAEReproductionPipeline",
    ),
}


def _jsonable(obj: Any) -> Any:
    """Coerce legacy engine output (sets, tuples, datetimes) to plain JSON.

    The IOI pipeline leaks e.g. discovered_nodes as a set of head labels —
    real circuit data, just not HTTP-safe. Normalize at this boundary so
    every downstream consumer (SSE, status endpoint, KG writes) is safe.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        items = [_jsonable(v) for v in obj]
        try:
            return sorted(items, key=repr)
        except Exception:
            return items
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()  # type: ignore[no-any-return]
        except Exception:
            pass
    return str(obj)


class Executor:
    """Executes model loads, prompts, patches, pipelines, orchestrated runs."""

    capability = "exec"

    # -- engine helpers -------------------------------------------------
    def _engine(self):  # type: ignore[no-untyped-def]
        from backend.services import gpt2_engine
        return gpt2_engine

    async def _call(self, name: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            engine = self._engine()
        except Exception as exc:
            return {"status": "unavailable", "reason": f"gpt2_engine import failed: {exc}"}
        if not engine.is_available():
            return {"status": "unavailable",
                    "reason": "torch/transformers not installed — seeded fallback in use"}
        try:
            fn = getattr(engine, name)
            res = await asyncio.to_thread(fn, *args, **kwargs)
            return res if isinstance(res, dict) else {"status": "ok", "result": res}
        except Exception as exc:
            return {"status": "error", "op": name, "error": str(exc)[:500]}

    # -- model ops ------------------------------------------------------
    async def ensure_model(self, model_name: str = "gpt2") -> Dict[str, Any]:
        res = await self._call("load")
        if res.get("status") == "loaded":
            res["model_name"] = model_name
        return res

    async def run_prompt(self, prompt: str) -> Dict[str, Any]:
        return await self._call("run_prompt", prompt)

    async def infer(self, prompt: str) -> Dict[str, Any]:
        return await self._call("infer", prompt)

    async def activations(self, layer: int = 5) -> Dict[str, Any]:
        return await self._call("activations", layer)

    async def patch_head(self, layer: int, head: int,
                           pos_token: str = " Paris",
                           neg_token: str = " London",
                           prompt: str = "The capital of France is") -> Dict[str, Any]:
        # patch_head reads _cache["prompt"] — populate it first.
        await self._call("run_prompt", prompt)
        return await self._call("patch_head", layer, head,
                                pos_token, neg_token)

    async def patch_neuron(self, layer: int, neuron_index: int,
                           patch_value: float = 3.5) -> Dict[str, Any]:
        return await self._call("patch_neuron", layer, neuron_index, patch_value)

    async def ioi(self, **kwargs: Any) -> Dict[str, Any]:
        return await self._call("ioi", **kwargs)

    # -- pipelines & orchestration (sync, CPU-cheap dispatch) ------------
    def reproduce(self, paper_id: str = "ioi") -> Dict[str, Any]:
        mod_name, cls_name = PIPELINE_MODULES.get(
            paper_id, PIPELINE_MODULES["ioi"])
        try:
            import importlib
            import inspect as _inspect
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            try:
                params = _inspect.signature(cls.__init__).parameters
                pipeline = cls(mock_mode=True) if "mock_mode" in params else cls()
            except Exception:
                pipeline = cls()
            res = pipeline.run()
            return {"status": "completed", "paper_id": paper_id,
                    "result": _jsonable(res)}
        except Exception as exc:
            return {"status": "error", "paper_id": paper_id,
                    "error": str(exc)[:500]}

    def orchestrate(self, experiment_id: str, goal: str,
                    strategy: str = "Balanced") -> Dict[str, Any]:
        try:
            from backend.runtime.orchestration.execution_orchestrator import (
                ExecutionOrchestrator,
            )
            orch = ExecutionOrchestrator()
            return orch.submit_and_orchestrate(
                experiment_id=experiment_id, goal=goal, strategy=strategy)
        except Exception as exc:
            return {"status": "error", "error": str(exc)[:500]}
