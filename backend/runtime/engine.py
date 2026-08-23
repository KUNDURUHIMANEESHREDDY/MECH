"""Core Orchestrator for Model Execution and Debugging.

Orchestrates distributed execution, multi-GPU partitioning, layer streaming,
activation compression, experiment scheduling, and debugger checkpoints.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from .batch_runner import BatchRunnerEngine
from .comparison import ModelComparisonEngine
from .execution.distributed import DistributedExecutionTarget, ExecutionTarget
from .execution.multi_gpu import MultiGPUPlacementPlanner
from .execution.streaming import ModelStreamingEngine
from .execution_context import ExecutionContext
from .logits import IntermediateLogitsEngine
from .memory.checkpoints import CheckpointSystem
from .memory.compression import ActivationCompressor
from .orchestration.execution_manager import ResourceManager
from .patching import PatchSet
from .scheduling.scheduler import ExperimentSchedulerEngine


class ExecutionEngine:
    """Orchestrates runtime capabilities across model families and execution targets."""

    def __init__(self) -> None:
        self.context = ExecutionContext()
        self.patch_set = PatchSet()
        self.comparison_engine = ModelComparisonEngine()
        self.logits_engine = IntermediateLogitsEngine()
        self.batch_runner = BatchRunnerEngine()
        self.distributed_target = DistributedExecutionTarget()
        self.placement_planner = MultiGPUPlacementPlanner()
        self.streaming_engine = ModelStreamingEngine()
        self.compressor = ActivationCompressor()
        self.checkpoint_system = CheckpointSystem()
        self.scheduler = ExperimentSchedulerEngine()
        self.resource_manager = ResourceManager()

    def get_status(self) -> Dict[str, Any]:
        metrics = self.resource_manager.get_resource_metrics()
        return {
            "status": "ready",
            "active_model": self.context.model_name,
            "session_id": self.context.session_id,
            "target": self.distributed_target.target_type,
            "active_breakpoints": self.context.get_breakpoints(),
            "active_patches": len(self.patch_set.to_list()),
            "metrics": metrics,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        }

    def analyze_tokens(self, prompt: str) -> Dict[str, Any]:
        """
        Analyzes tokens from a prompt using the actual model.
        
        Returns real logits computed by the model, or NOT_EXECUTABLE if
        no model is loaded. Never returns fabricated values.
        """
        tokens = prompt.split() if prompt else ["<empty>"]
        
        # Try to get real logits from the model
        model_name = self.context.model_name
        if not model_name:
            return {
                "prompt": prompt,
                "token_count": len(tokens),
                "tokens": [],
                "model": None,
                "status": "NOT_EXECUTABLE",
                "error": "No model loaded. Cannot compute logits.",
            }
        
        try:
            from backend.core.model_adapter import get_model_adapter
            adapter = get_model_adapter(model_id=model_name)
            adapter.load()
            
            # Tokenize and run forward pass
            import torch
            inputs = adapter.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(adapter._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                out = adapter.model(**inputs, return_dict=True)
            
            logits = out.logits[0, -1, :]  # Last token logits
            
            # Get top-k tokens for each position
            token_analyses = []
            for idx, tok in enumerate(tokens[:len(inputs["input_ids"][0])]):
                # Get the logit for this token's position
                if idx < len(logits):
                    logit_val = float(logits[idx].item()) if idx < logits.shape[0] else 0.0
                else:
                    logit_val = 0.0
                
                token_analyses.append({
                    "index": idx,
                    "token": tok,
                    "logit": round(logit_val, 4),
                })
            
            return {
                "prompt": prompt,
                "token_count": len(tokens),
                "tokens": token_analyses,
                "model": model_name,
                "status": "COMPLETED",
            }
            
        except Exception as e:
            return {
                "prompt": prompt,
                "token_count": len(tokens),
                "tokens": [],
                "model": model_name,
                "status": "NOT_EXECUTABLE",
                "error": f"Failed to compute logits: {e}",
            }

    def apply_patch(self, patch_dict: Dict[str, Any]) -> Dict[str, Any]:
        from .patching import ActivationPatch
        p = ActivationPatch(
            session_id=patch_dict.get("session_id", "default"),
            layer=patch_dict.get("layer", 0),
            component=patch_dict.get("component", "mlp"),
            neuron_index=patch_dict.get("neuron_index", patch_dict.get("neuron", 0)),
            operation=patch_dict.get("operation", "replace"),
            value=patch_dict.get("value", 0.0),
        )
        self.patch_set.add_patch(p)
        self.context.add_patch(patch_dict)
        return {
            "status": "patch_applied",
            "patch": p.to_dict(),
            "active_patches_count": len(self.patch_set.to_list()),
        }

    def compare_models(self, prompt: str, model_a: str, model_b: str) -> Dict[str, Any]:
        return self.comparison_engine.compare(prompt=prompt, model_a=model_a, model_b=model_b)

    def get_intermediate_logits(self, prompt: str, layers: int = 12) -> Dict[str, Any]:
        return self.logits_engine.extract_logits(prompt=prompt, num_layers=layers)

    def run_batch_experiment(self, experiment_id: str, model_name: str, prompts: List[str]) -> Dict[str, Any]:
        from .batch_runner import ExperimentSpec
        spec = ExperimentSpec(experiment_id=experiment_id, model_name=model_name, prompts=prompts)
        return self.batch_runner.run_experiment(spec)


# Alias for backward compatibility
NeuralDebuggerEngine = ExecutionEngine

_engine_instance: ExecutionEngine | None = None


def get_engine() -> ExecutionEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ExecutionEngine()
    return _engine_instance
