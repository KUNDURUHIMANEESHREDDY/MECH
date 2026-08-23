"""A/B Experiment Configuration for MECH Platform."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MECH.features.experiments")


class ExperimentStatus(Enum):
    """Experiment status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class AssignmentStrategy(Enum):
    """User assignment strategy."""

    RANDOM = "random"
    USER_ID = "user_id"
    CONTEXT = "context"
    STICKY = "sticky"  # Consistent assignment based on user_id


@dataclass
class ExperimentVariant:
    """Experiment variant (control/treatment)."""

    name: str
    description: str
    config: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Traffic weight (normalized)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "weight": self.weight,
        }


@dataclass
class Experiment:
    """A/B experiment definition."""

    name: str
    description: str
    variants: List[ExperimentVariant]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    assignment_strategy: AssignmentStrategy = AssignmentStrategy.USER_ID
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    targeting_rules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        # Normalize weights
        total_weight = sum(v.weight for v in self.variants)
        if total_weight > 0:
            for v in self.variants:
                v.weight = v.weight / total_weight

    def get_variant_for(self, user_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Optional[ExperimentVariant]:
        """Get assigned variant for user/context."""
        if self.status != ExperimentStatus.RUNNING:
            return None

        # Check targeting rules
        if not self._matches_targeting(user_id, context):
            return None

        # Check date range
        now = datetime.utcnow()
        if self.start_date and now < self.start_date:
            return None
        if self.end_date and now > self.end_date:
            return None

        # Assign variant
        return self._assign_variant(user_id, context)

    def _matches_targeting(self, user_id: Optional[str], context: Optional[Dict[str, Any]]) -> bool:
        """Check if user/context matches targeting rules."""
        if not self.targeting_rules:
            return True

        # Simple targeting: check if context contains required keys
        for key, value in self.targeting_rules.items():
            if context is None or context.get(key) != value:
                return False
        return True

    def _assign_variant(self, user_id: Optional[str], context: Optional[Dict[str, Any]]) -> Optional[ExperimentVariant]:
        """Assign variant based on strategy."""
        if not self.variants:
            return None

        if self.assignment_strategy == AssignmentStrategy.RANDOM:
            import random
            rand = random.random()
        elif self.assignment_strategy == AssignmentStrategy.USER_ID:
            if not user_id:
                return self.variants[0]  # Default to first variant
            # Consistent hash based on user_id + experiment name
            hash_val = int(hashlib.md5(f"{self.name}:{user_id}".encode()).hexdigest()[:8], 16)
            rand = (hash_val % 10000) / 10000.0
        elif self.assignment_strategy == AssignmentStrategy.CONTEXT:
            if not context:
                return self.variants[0]
            ctx_str = str(sorted(context.items()))
            hash_val = int(hashlib.md5(f"{self.name}:{ctx_str}".encode()).hexdigest()[:8], 16)
            rand = (hash_val % 10000) / 10000.0
        elif self.assignment_strategy == AssignmentStrategy.STICKY:
            # Same as USER_ID but falls back to context
            identifier = user_id or str(context) if context else "anonymous"
            hash_val = int(hashlib.md5(f"{self.name}:{identifier}".encode()).hexdigest()[:8], 16)
            rand = (hash_val % 10000) / 10000.0
        else:
            return self.variants[0]

        # Weighted random selection
        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if rand < cumulative:
                return variant

        return self.variants[-1]  # Fallback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "status": self.status.value,
            "assignment_strategy": self.assignment_strategy.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "targeting_rules": self.targeting_rules,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# Built-in experiments
BUILTIN_EXPERIMENTS = [
    Experiment(
        name="dispatcher_v2_rollout",
        description="Gradual rollout of new API dispatcher v2",
        variants=[
            ExperimentVariant(name="control", description="Legacy dispatcher", config={"dispatcher_version": "v1"}, weight=0.5),
            ExperimentVariant(name="treatment", description="New dispatcher v2", config={"dispatcher_version": "v2"}, weight=0.5),
        ],
        status=ExperimentStatus.RUNNING,
        assignment_strategy=AssignmentStrategy.USER_ID,
    ),
    Experiment(
        name="runtime_memory_optimization",
        description="Test out-of-core runtime vs in-memory for large models",
        variants=[
            ExperimentVariant(name="control", description="In-memory runtime", config={"runtime_type": "in_memory"}, weight=0.7),
            ExperimentVariant(name="treatment", description="Out-of-core runtime", config={"runtime_type": "out_of_core"}, weight=0.3),
        ],
        status=ExperimentStatus.DRAFT,
        assignment_strategy=AssignmentStrategy.CONTEXT,
        targeting_rules={"model_size": "large"},
    ),
    Experiment(
        name="circuit_verification_threshold",
        description="Test different ACDC faithfulness thresholds",
        variants=[
            ExperimentVariant(name="control", description="Standard threshold (0.05)", config={"acdc_threshold": 0.05}, weight=0.5),
            ExperimentVariant(name="treatment", description="Stricter threshold (0.01)", config={"acdc_threshold": 0.01}, weight=0.5),
        ],
        status=ExperimentStatus.DRAFT,
        assignment_strategy=AssignmentStrategy.USER_ID,
    ),
    Experiment(
        name="rate_limit_algorithm",
        description="Compare rate limiting algorithms",
        variants=[
            ExperimentVariant(name="control", description="Sliding window", config={"rate_limit_algorithm": "sliding_window"}, weight=0.5),
            ExperimentVariant(name="treatment", description="Token bucket", config={"rate_limit_algorithm": "token_bucket"}, weight=0.5),
        ],
        status=ExperimentStatus.DRAFT,
        assignment_strategy=AssignmentStrategy.USER_ID,
    ),
]


def get_builtin_experiments() -> List[Experiment]:
    """Get all built-in experiments."""
    return BUILTIN_EXPERIMENTS.copy()


def get_experiment_by_name(name: str) -> Optional[Experiment]:
    """Get an experiment by name."""
    for exp in BUILTIN_EXPERIMENTS:
        if exp.name == name:
            return exp
    return None