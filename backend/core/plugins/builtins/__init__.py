"""Built-in plugins package for MECH Platform."""

from backend.core.plugins.builtins.logit_lens import LogitLensPlugin
from backend.core.plugins.builtins.activation_patching import ActivationPatchingPlugin
from backend.core.plugins.builtins.sae_features import SAEFeaturesPlugin
from backend.core.plugins.builtins.circuit_discovery import CircuitDiscoveryPlugin
from backend.core.plugins.builtins.path_patching import PathPatchingPlugin
from backend.core.plugins.builtins.hallucination_experiment import HallucinationExperimentPlugin
from backend.core.plugins.builtins.semantic_falsification import SemanticFalsificationPlugin
from backend.core.plugins.builtins.circuit_metrics import CircuitMetricsPlugin
from backend.core.plugins.builtins.live_intervention import LiveInterventionPlugin
from backend.core.plugins.builtins.scientific_validation import ScientificValidationPlugin
from backend.core.plugins.builtins.cross_model_universality import CrossModelUniversalityPlugin
from backend.core.plugins.builtins.backup_circuits import BackupCircuitsPlugin

__all__ = [
    "LogitLensPlugin",
    "ActivationPatchingPlugin",
    "SAEFeaturesPlugin",
    "CircuitDiscoveryPlugin",
    "PathPatchingPlugin",
    "HallucinationExperimentPlugin",
    "SemanticFalsificationPlugin",
    "CircuitMetricsPlugin",
    "LiveInterventionPlugin",
    "ScientificValidationPlugin",
    "CrossModelUniversalityPlugin",
    "BackupCircuitsPlugin",
]


def get_all_plugins():
    """Get all built-in plugin instances."""
    return [
        LogitLensPlugin(),
        ActivationPatchingPlugin(),
        SAEFeaturesPlugin(),
        CircuitDiscoveryPlugin(),
        PathPatchingPlugin(),
        HallucinationExperimentPlugin(),
        SemanticFalsificationPlugin(),
        CircuitMetricsPlugin(),
        LiveInterventionPlugin(),
        ScientificValidationPlugin(),
        CrossModelUniversalityPlugin(),
        BackupCircuitsPlugin(),
    ]