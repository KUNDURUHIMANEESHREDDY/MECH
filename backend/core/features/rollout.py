"""Gradual Rollout Logic for MECH Platform."""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from backend.core.features.flags import FeatureFlag, FeatureState, get_builtin_flags, get_flag_by_name
from backend.core.features.experiments import Experiment, ExperimentStatus, get_builtin_experiments, get_experiment_by_name

logger = logging.getLogger("MECH.features.rollout")


@dataclass
class RolloutRule:
    """Rule for gradual rollout."""

    feature_name: str
    percentage: float  # 0.0 to 100.0
    conditions: Dict[str, Any] = field(default_factory=dict)  # e.g., {"model_size": "large"}
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    def matches(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if rule matches context."""
        if not self.conditions:
            return True

        if context is None:
            return False

        for key, value in self.conditions.items():
            if context.get(key) != value:
                return False
        return True

    def is_active(self) -> bool:
        """Check if rule is active based on dates."""
        now = datetime.utcnow()
        if self.start_date and now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False
        return True


class RolloutManager:
    """Manages feature flag rollouts and experiment assignments."""

    def __init__(self) -> None:
        self._flags: Dict[str, FeatureFlag] = {}
        self._experiments: Dict[str, Experiment] = {}
        self._rollout_rules: List[RolloutRule] = []
        self._assignments: Dict[str, Dict[str, str]] = {}  # feature -> user -> variant
        self._lock = threading.RLock()
        self._initialized = False

    def initialize(self, flags: Optional[List[FeatureFlag]] = None, experiments: Optional[List[Experiment]] = None) -> None:
        """Initialize with flags and experiments."""
        with self._lock:
            if self._initialized:
                return

            # Load built-in flags
            for flag in flags or get_builtin_flags():
                self._flags[flag.name] = flag

            # Load built-in experiments
            for exp in experiments or get_builtin_experiments():
                self._experiments[exp.name] = exp

            self._initialized = True
            logger.info("Rollout manager initialized: %d flags, %d experiments", len(self._flags), len(self._experiments))

    def is_enabled(self, feature_name: str, user_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature is enabled for a user/context."""
        with self._lock:
            # Check rollout rules first
            for rule in self._rollout_rules:
                if rule.feature_name == feature_name and rule.matches(context) and rule.is_active():
                    return self._check_percentage(feature_name, rule.percentage, user_id, context)

            # Check feature flag
            flag = self._flags.get(feature_name)
            if flag:
                return flag.is_enabled_for(user_id, context)

            return False

    def _check_percentage(self, feature_name: str, percentage: float, user_id: Optional[str], context: Optional[Dict[str, Any]]) -> bool:
        """Check percentage-based rollout with sticky assignment."""
        identifier = user_id or str(context) if context else "anonymous"
        hash_val = int(hashlib.md5(f"{feature_name}:{identifier}".encode()).hexdigest()[:8], 16)
        user_percentage = (hash_val % 10000) / 100.0
        return user_percentage < percentage

    def get_variant(self, experiment_name: str, user_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get assigned variant for an experiment."""
        with self._lock:
            experiment = self._experiments.get(experiment_name)
            if not experiment:
                return None

            variant = experiment.get_variant_for(user_id, context)
            if variant:
                # Cache assignment for stickiness
                if experiment.assignment_strategy in ["sticky", "user_id"]:
                    self._assignments.setdefault(experiment_name, {})[user_id or "anonymous"] = variant.name
                return variant.name
            return None

    def get_config(self, feature_name: str, user_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get merged configuration for a feature/experiment."""
        config = {}

        # Check experiments first (they can override feature config)
        for exp in self._experiments.values():
            if exp.status == ExperimentStatus.RUNNING:
                variant = exp.get_variant_for(user_id, context)
                if variant:
                    config.update(variant.config)

        # Apply feature flag config
        flag = self._flags.get(feature_name)
        if flag:
            config.setdefault("feature_enabled", flag.is_enabled_for(user_id, context))

        return config

    def add_rollout_rule(self, rule: RolloutRule) -> None:
        """Add a rollout rule."""
        with self._lock:
            self._rollout_rules.append(rule)
            logger.info("Added rollout rule for %s: %.1f%%", rule.feature_name, rule.percentage)

    def remove_rollout_rule(self, feature_name: str) -> bool:
        """Remove rollout rules for a feature."""
        with self._lock:
            initial_len = len(self._rollout_rules)
            self._rollout_rules = [r for r in self._rollout_rules if r.feature_name != feature_name]
            return len(self._rollout_rules) < initial_len

    def set_flag_state(self, feature_name: str, state: FeatureState, rollout_percentage: float = 0.0) -> bool:
        """Update feature flag state."""
        with self._lock:
            flag = self._flags.get(feature_name)
            if not flag:
                return False
            flag.state = state
            flag.rollout_percentage = rollout_percentage
            logger.info("Updated flag %s: state=%s, rollout=%.1f%%", feature_name, state.value, rollout_percentage)
            return True

    def get_flag(self, feature_name: str) -> Optional[FeatureFlag]:
        """Get feature flag by name."""
        with self._lock:
            return self._flags.get(feature_name)

    def list_flags(self) -> List[FeatureFlag]:
        """List all feature flags."""
        with self._lock:
            return list(self._flags.values())

    def list_experiments(self) -> List[Experiment]:
        """List all experiments."""
        with self._lock:
            return list(self._experiments.values())

    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get experiment by name."""
        with self._lock:
            return self._experiments.get(name)

    def get_all_configs(self, user_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get all feature configs for a user/context."""
        configs = {}
        for flag in self._flags.values():
            configs[flag.name] = {
                "enabled": flag.is_enabled_for(user_id, context),
                "state": flag.state.value,
                "rollout_percentage": flag.rollout_percentage,
            }
        for exp in self._experiments.values():
            if exp.status == ExperimentStatus.RUNNING:
                variant = exp.get_variant_for(user_id, context)
                if variant:
                    configs[f"experiment:{exp.name}"] = {
                        "variant": variant.name,
                        "config": variant.config,
                    }
        return configs


# Global instance
_rollout_manager: Optional[RolloutManager] = None
_init_lock = threading.Lock()


def get_rollout_manager() -> RolloutManager:
    """Get the global rollout manager."""
    global _rollout_manager
    if _rollout_manager is None:
        with _init_lock:
            if _rollout_manager is None:
                _rollout_manager = RolloutManager()
                _rollout_manager.initialize()
    return _rollout_manager


def reset_rollout_manager() -> None:
    """Reset the global rollout manager (for testing)."""
    global _rollout_manager
    with _init_lock:
        _rollout_manager = None