"""LLM Interaction Layer.

Enables MECH to work with LLMs through the model layer, including local-model
setups. Provides prompt sending, response capture, behavior inspection,
controlled experiments, and cross-input/model comparison.
"""

from .engine import InteractionEngine
from .models import (
    InteractionRequest,
    InteractionResponse,
    InteractionRecord,
    ExperimentSpec,
    ExperimentResult,
    ComparisonSpec,
    ComparisonResult,
    InspectionSpec,
    InspectionResult,
)
from .experiments import ExperimentRunner
from .comparison import InteractionComparisonEngine
from .inspector import BehaviorInspector

__all__ = [
    "InteractionEngine",
    "InteractionRequest",
    "InteractionResponse",
    "InteractionRecord",
    "ExperimentSpec",
    "ExperimentResult",
    "ComparisonSpec",
    "ComparisonResult",
    "InspectionSpec",
    "InspectionResult",
    "ExperimentRunner",
    "InteractionComparisonEngine",
    "BehaviorInspector",
]