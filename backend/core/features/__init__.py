"""Feature Flag / Experiment Config Module for MECH Platform."""

from backend.core.features.flags import (
    FeatureFlag,
    FeatureState,
    get_builtin_flags,
    get_flag_by_name,
    BUILTIN_FLAGS,
)
from backend.core.features.experiments import (
    Experiment,
    ExperimentVariant,
    ExperimentStatus,
    AssignmentStrategy,
    get_builtin_experiments,
    get_experiment_by_name,
    BUILTIN_EXPERIMENTS,
)
from backend.core.features.rollout import (
    RolloutManager,
    RolloutRule,
    get_rollout_manager,
    reset_rollout_manager,
)

__all__ = [
    # Flags
    "FeatureFlag",
    "FeatureState",
    "get_builtin_flags",
    "get_flag_by_name",
    "BUILTIN_FLAGS",
    # Experiments
    "Experiment",
    "ExperimentVariant",
    "ExperimentStatus",
    "AssignmentStrategy",
    "get_builtin_experiments",
    "get_experiment_by_name",
    "BUILTIN_EXPERIMENTS",
    # Rollout
    "RolloutManager",
    "RolloutRule",
    "get_rollout_manager",
    "reset_rollout_manager",
]