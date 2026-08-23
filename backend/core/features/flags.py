"""Feature Flag Definitions for MECH Platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from backend.core.config import get_settings


class FeatureState(Enum):
    """Feature flag states."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    ROLLOUT = "rollout"  # Percentage-based rollout


@dataclass
class FeatureFlag:
    """Feature flag definition."""

    name: str
    description: str
    state: FeatureState = FeatureState.DISABLED
    rollout_percentage: float = 0.0  # 0.0 to 100.0
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_enabled_for(self, user_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if feature is enabled for a user/context."""
        if self.state == FeatureState.ENABLED:
            return True
        if self.state == FeatureState.DISABLED:
            return False
        if self.state == FeatureState.ROLLOUT:
            return self._check_rollout(user_id, context)
        return False

    def _check_rollout(self, user_id: Optional[str], context: Optional[Dict[str, Any]]) -> bool:
        """Check percentage-based rollout."""
        import hashlib

        # Use user_id or context to create consistent hash
        identifier = user_id or str(context) if context else "anonymous"
        hash_val = int(hashlib.md5(f"{self.name}:{identifier}".encode()).hexdigest()[:8], 16)
        percentage = (hash_val % 10000) / 100.0  # 0.00 to 99.99
        return percentage < self.rollout_percentage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "state": self.state.value,
            "rollout_percentage": self.rollout_percentage,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "metadata": self.metadata,
        }


# Built-in feature flags
BUILTIN_FLAGS = [
    FeatureFlag(
        name="new_dispatcher",
        description="Use new API dispatcher v2",
        state=FeatureState.ROLLOUT,
        rollout_percentage=50.0,
        tags=["api", "dispatcher"],
    ),
    FeatureFlag(
        name="async_runtime",
        description="Enable async runtime execution",
        state=FeatureState.ENABLED,
        tags=["runtime", "async"],
    ),
    FeatureFlag(
        name="model_integrity_gate",
        description="Enforce model integrity gate before experiments",
        state=FeatureState.ENABLED,
        tags=["validation", "integrity"],
    ),
    FeatureFlag(
        name="behavioral_validation",
        description="Run behavioral sanity checks on model load",
        state=FeatureState.ENABLED,
        tags=["validation", "behavioral"],
    ),
    FeatureFlag(
        name="redis_rate_limiting",
        description="Use Redis for distributed rate limiting",
        state=FeatureState.ROLLOUT,
        rollout_percentage=100.0,  # Enabled when REDIS_URL is set
        dependencies=["redis_available"],
        tags=["rate_limiting", "redis"],
    ),
    FeatureFlag(
        name="out_of_core_runtime",
        description="Enable out-of-core runtime for large models",
        state=FeatureState.ROLLOUT,
        rollout_percentage=25.0,
        tags=["runtime", "memory"],
    ),
    FeatureFlag(
        name="circuit_verification",
        description="Enable ACDC circuit verification pipeline",
        state=FeatureState.ENABLED,
        tags=["circuits", "verification"],
    ),
    FeatureFlag(
        name="cross_model_analysis",
        description="Enable cross-model circuit comparison",
        state=FeatureState.ROLLOUT,
        rollout_percentage=50.0,
        tags=["comparative", "cross_model"],
    ),
    FeatureFlag(
        name="hallucination_pipeline",
        description="Enable hallucination competition experiment",
        state=FeatureState.ENABLED,
        tags=["causal", "hallucination"],
    ),
    FeatureFlag(
        name="semantic_falsification",
        description="Enable semantic falsification probes",
        state=FeatureState.ENABLED,
        tags=["verification", "falsification"],
    ),
    FeatureFlag(
        name="scientific_validation_suite",
        description="Enable 5-pillar scientific validation",
        state=FeatureState.ENABLED,
        tags=["validation", "scientific"],
    ),
    FeatureFlag(
        name="live_intervention",
        description="Enable live tensor intervention",
        state=FeatureState.ROLLOUT,
        rollout_percentage=75.0,
        tags=["causal", "live"],
    ),
    FeatureFlag(
        name="backup_circuit_discovery",
        description="Enable redundant backup circuit discovery",
        state=FeatureState.ENABLED,
        tags=["redundancy", "backup"],
    ),
    FeatureFlag(
        name="unified_registry",
        description="Use unified registry for all capabilities",
        state=FeatureState.ROLLOUT,
        rollout_percentage=100.0,
        tags=["registry", "unified"],
    ),
    FeatureFlag(
        name="dynamic_prompt_sampling",
        description="Enable dynamic prompt sampling on startup",
        state=FeatureState.ENABLED,
        tags=["sampling", "startup"],
    ),
    FeatureFlag(
        name="plugin_system",
        description="Enable new plugin system for tools",
        state=FeatureState.ENABLED,
        tags=["plugins", "tools"],
    ),
    FeatureFlag(
        name="capability_definitions",
        description="Use new capability definition system",
        state=FeatureState.ENABLED,
        tags=["capabilities", "definitions"],
    ),
    FeatureFlag(
        name="factory_system",
        description="Use factory system for engines/runtimes",
        state=FeatureState.ENABLED,
        tags=["factories", "engines"],
    ),
    FeatureFlag(
        name="experiment_tracking",
        description="Enable experiment tracking and A/B testing",
        state=FeatureState.DISABLED,
        tags=["experiments", "tracking"],
    ),
    FeatureFlag(
        name="distributed_execution",
        description="Enable distributed cluster execution",
        state=FeatureState.DISABLED,
        tags=["distributed", "cluster"],
    ),
    FeatureFlag(
        name="multi_precision",
        description="Enable multi-precision runtime support",
        state=FeatureState.ROLLOUT,
        rollout_percentage=50.0,
        tags=["runtime", "precision"],
    ),
]


def get_builtin_flags() -> List[FeatureFlag]:
    """Get all built-in feature flags."""
    return BUILTIN_FLAGS.copy()


def get_flag_by_name(name: str) -> Optional[FeatureFlag]:
    """Get a feature flag by name."""
    for flag in BUILTIN_FLAGS:
        if flag.name == name:
            return flag
    return None