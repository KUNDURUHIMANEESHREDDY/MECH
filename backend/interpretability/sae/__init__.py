"""Unified Sparse Autoencoder (SAE) Subsystem for MECH.

Provides:
- Unified SAE Interface (SAEInterface, SAEMetadata, SAEReconstructionResult, SAEOriginState, SAEProvenance)
- Central SAE Registry (SAERegistry, default_sae_registry)
- Adapters (NativeMECHSAE, SAELensAdapter, SparseAutoencoderAdapter, GenericPyTorchSAEAdapter)
- Loaders (NativeSAELoader, SAELensLoader, HuggingFaceSAELoader, SAELoader)
- Attribution (SAEDirectLogitAttributor, SAEDLAResult)
- Analysis (FeatureActivationAnalyzer, FeatureSparsityTracker, FeatureInterpretabilityAnalyzer, FeatureDashboardEngine)
- Training (SAETrainer, SAETrainingConfig, SAELossCalculator, DeadFeatureResampler)
- Causal Steering (SAECausalInterventionEngine, SAECausalInterventionResult)
- Validation (SAEValidator, SAEValidationAudit, ScientificSAEReportEngine, SAEEmpiricalValidationReport)
"""

from .cache import SAECache, PersistentActivationCache
from .feature_dictionary import FeatureDictionary, SAEFeature
from .inspector import SAEInspector
from .loader import SAELoader, SAEConfig, SAE

from .sae_interface import (
    SAEInterface,
    SAEMetadata,
    SAEArchitectureType,
    SAEBackendSource,
    SAEOriginState,
    SAEProvenance,
    FeatureActivationSummary,
    SAEReconstructionResult,
)
from .sae_registry import SAERegistry, default_sae_registry
from .sae_adapter import (
    NativeMECHSAE,
    GenericPyTorchSAEAdapter,
    SAELensAdapter,
    SparseAutoencoderAdapter,
)
from .loaders import (
    NativeSAELoader,
    SAELensLoader,
    HuggingFaceSAELoader,
)
from .attribution import (
    SAEDirectLogitAttributor,
    SAEDLAResult,
)
from .analysis import (
    FeatureActivationAnalyzer,
    FeatureSparsityTracker,
    FeatureInterpretabilityAnalyzer,
    FeatureDashboardEngine,
)
from .training import (
    SAETrainer,
    SAETrainingConfig,
    SAELossCalculator,
    SAELossOutput,
    DeadFeatureResampler,
)
from .causal import (
    SAECausalInterventionEngine,
    SAECausalInterventionResult,
)
from .validation import (
    SAEValidator,
    SAEValidationAudit,
    ScientificSAEReportEngine,
    SAEEmpiricalValidationReport,
    SAEBaseError,
    SAELoadError,
    SAEIncompatibleError,
    SAECorruptedError,
    SAEExecutionError,
)
from .live_sae_engine import LiveSAEEngine, LiveSparseAutoencoder

__all__ = [
    # Legacy / compatibility
    "SAELoader",
    "SAEConfig",
    "SAE",
    "FeatureDictionary",
    "SAEFeature",
    "SAEInspector",
    "SAECache",
    "PersistentActivationCache",
    # Unified core
    "SAEInterface",
    "SAEMetadata",
    "SAEArchitectureType",
    "SAEBackendSource",
    "SAEOriginState",
    "SAEProvenance",
    "FeatureActivationSummary",
    "SAEReconstructionResult",
    "SAERegistry",
    "default_sae_registry",
    # Adapters
    "NativeMECHSAE",
    "GenericPyTorchSAEAdapter",
    "SAELensAdapter",
    "SparseAutoencoderAdapter",
    # Loaders
    "NativeSAELoader",
    "SAELensLoader",
    "HuggingFaceSAELoader",
    # Attribution
    "SAEDirectLogitAttributor",
    "SAEDLAResult",
    # Analysis
    "FeatureActivationAnalyzer",
    "FeatureSparsityTracker",
    "FeatureInterpretabilityAnalyzer",
    "FeatureDashboardEngine",
    # Training
    "SAETrainer",
    "SAETrainingConfig",
    "SAELossCalculator",
    "SAELossOutput",
    "DeadFeatureResampler",
    # Causal Steering
    "SAECausalInterventionEngine",
    "SAECausalInterventionResult",
    # Validation & Diagnostics
    "SAEValidator",
    "SAEValidationAudit",
    "ScientificSAEReportEngine",
    "SAEEmpiricalValidationReport",
    "SAEBaseError",
    "SAELoadError",
    "SAEIncompatibleError",
    "SAECorruptedError",
    "SAEExecutionError",
    # Live engine
    "LiveSAEEngine",
    "LiveSparseAutoencoder",
]
