"""Built-in Capability Definitions for MECH Platform."""

from __future__ import annotations

from backend.core.capabilities.registry import CapabilityRegistry, get_capability_registry
from backend.core.capabilities.schema import CapabilitySpec


# Built-in capability specifications
BUILTIN_CAPABILITIES = [
    CapabilitySpec(
        name="logit_lens",
        description="Decomposes intermediate residual stream states into token vocabulary projections per layer",
        version="1.0.0",
        category="localization",
        provides_tools=["run_logit_lens"],
        tags=["localization", "residual_stream", "vocabulary_projection"],
    ),
    CapabilitySpec(
        name="causal_tracing",
        description="Causal intervention techniques for tracing information flow through model components",
        version="1.0.0",
        category="causal",
        provides_tools=["run_activation_patching", "run_path_patching", "run_live_tensor_intervention"],
        dependencies=["patching"],
        tags=["causal", "intervention", "attribution", "mediation"],
    ),
    CapabilitySpec(
        name="patching",
        description="Activation patching and causal mediation analysis",
        version="1.0.0",
        category="causal",
        provides_tools=["run_activation_patching", "run_path_patching"],
        tags=["patching", "activation_patching", "path_patching"],
    ),
    CapabilitySpec(
        name="sae",
        description="Sparse Autoencoder feature discovery and dictionary learning",
        version="1.0.0",
        category="dictionary_learning",
        provides_tools=["extract_sae_features"],
        tags=["sae", "dictionary_learning", "features", "monosemantic"],
    ),
    CapabilitySpec(
        name="acdc_circuits",
        description="Automated Circuit Discovery (ACDC) for extracting minimal subnetwork graphs",
        version="1.0.0",
        category="circuits",
        provides_tools=["discover_circuit"],
        tags=["circuits", "acdc", "discovery", "subnetwork"],
    ),
    CapabilitySpec(
        name="circuits",
        description="Circuit analysis and manipulation utilities",
        version="1.0.0",
        category="circuits",
        dependencies=["acdc_circuits"],
        tags=["circuits", "analysis"],
    ),
    CapabilitySpec(
        name="verification",
        description="Scientific validation and falsification of mechanistic claims",
        version="1.0.0",
        category="verification",
        provides_tools=["run_semantic_falsification_probe", "run_scientific_circuit_validation"],
        tags=["verification", "falsification", "scientific", "validation"],
    ),
    CapabilitySpec(
        name="falsification",
        description="Semantic falsification probes for component role verification",
        version="1.0.0",
        category="verification",
        provides_tools=["run_semantic_falsification_probe"],
        dependencies=["verification"],
        tags=["falsification", "semantic", "probes"],
    ),
    CapabilitySpec(
        name="scientific_validation",
        description="5-pillar scientific validation suite for circuit claims",
        version="1.0.0",
        category="verification",
        provides_tools=["run_scientific_circuit_validation"],
        dependencies=["verification"],
        tags=["validation", "scientific", "rigor", "statistics"],
    ),
    CapabilitySpec(
        name="metrics",
        description="ACDC standard circuit metrics (Faithfulness, Completeness, Minimality)",
        version="1.0.0",
        category="verification",
        provides_tools=["evaluate_circuit_metrics"],
        tags=["metrics", "acdc", "faithfulness", "completeness", "minimality"],
    ),
    CapabilitySpec(
        name="comparative",
        description="Cross-model comparison and alignment analysis",
        version="1.0.0",
        category="comparative",
        provides_tools=["run_cross_model_universality_sweep"],
        tags=["comparative", "cross_model", "alignment"],
    ),
    CapabilitySpec(
        name="cross_model",
        description="Cross-model circuit universality and transfer evaluation",
        version="1.0.0",
        category="comparative",
        dependencies=["comparative"],
        tags=["cross_model", "universality", "transfer"],
    ),
    CapabilitySpec(
        name="universality",
        description="Circuit universality across model families",
        version="1.0.0",
        category="comparative",
        dependencies=["cross_model"],
        tags=["universality", "conservation", "primitives"],
    ),
    CapabilitySpec(
        name="redundancy",
        description="Redundant backup head discovery and self-repair analysis",
        version="1.0.0",
        category="redundancy",
        provides_tools=["discover_backup_redundant_circuits"],
        tags=["redundancy", "backup_heads", "self_repair", "knockout"],
    ),
    CapabilitySpec(
        name="backup_heads",
        description="Wang et al. (2022) backup head mechanism discovery",
        version="1.0.0",
        category="redundancy",
        dependencies=["redundancy"],
        provides_tools=["discover_backup_redundant_circuits"],
        tags=["backup", "wang2022", "compensatory"],
    ),
    CapabilitySpec(
        name="hallucination",
        description="Causal competition experiment for hallucination circuit discovery",
        version="1.0.0",
        category="causal",
        provides_tools=["run_hallucination_experiment"],
        dependencies=["patching"],
        tags=["hallucination", "competition", "causal", "pipeline"],
    ),
    CapabilitySpec(
        name="live_intervention",
        description="Real-time forward pass intervention with PyTorch hooks",
        version="1.0.0",
        category="causal",
        provides_tools=["run_live_tensor_intervention"],
        tags=["live", "hooks", "real_time", "intervention"],
    ),
    CapabilitySpec(
        name="llama",
        description="LLaMA model family support",
        version="1.0.0",
        category="general",
        required_models=["llama", "llama-2", "llama-3"],
        tags=["model", "llama"],
    ),
    CapabilitySpec(
        name="gemma",
        description="Gemma model family support",
        version="1.0.0",
        category="general",
        required_models=["gemma", "gemma-2"],
        tags=["model", "gemma"],
    ),
    CapabilitySpec(
        name="qwen",
        description="Qwen model family support",
        version="1.0.0",
        category="general",
        required_models=["qwen", "qwen-2"],
        tags=["model", "qwen"],
    ),
    CapabilitySpec(
        name="mistral",
        description="Mistral model family support",
        version="1.0.0",
        category="general",
        required_models=["mistral"],
        tags=["model", "mistral"],
    ),
    CapabilitySpec(
        name="deepseek",
        description="DeepSeek model family support",
        version="1.0.0",
        category="general",
        required_models=["deepseek"],
        tags=["model", "deepseek"],
    ),
    CapabilitySpec(
        name="visualization",
        description="Visualization tools (bertviz, circuitsvis, heatmaps)",
        version="1.0.0",
        category="general",
        provides_tools=[],
        tags=["visualization", "bertviz", "circuitsvis", "heatmaps"],
    ),
    CapabilitySpec(
        name="reproducibility",
        description="Experiment reproducibility and replication tools",
        version="1.0.0",
        category="general",
        tags=["reproducibility", "replication", "experiments"],
    ),
    CapabilitySpec(
        name="cluster",
        description="Distributed/cluster computing support",
        version="1.0.0",
        category="general",
        tags=["cluster", "distributed", "scaling"],
    ),
    CapabilitySpec(
        name="cloud",
        description="Cloud execution and remote model support",
        version="1.0.0",
        category="general",
        tags=["cloud", "remote", "api"],
    ),
    CapabilitySpec(
        name="ai_research_assistant",
        description="AI-powered research assistance and hypothesis generation",
        version="1.0.0",
        category="general",
        tags=["ai_assistant", "research", "hypothesis"],
    ),
]


def register_builtin_capabilities(registry: Optional[CapabilityRegistry] = None) -> CapabilityRegistry:
    """Register all built-in capabilities."""
    reg = registry or get_capability_registry()
    for cap in BUILTIN_CAPABILITIES:
        reg.register(cap)
    return reg


def get_builtin_capability(name: str) -> Optional[CapabilitySpec]:
    """Get a built-in capability by name."""
    for cap in BUILTIN_CAPABILITIES:
        if cap.name == name:
            return cap
    return None


def list_builtin_capabilities() -> List[CapabilitySpec]:
    """List all built-in capabilities."""
    return BUILTIN_CAPABILITIES.copy()