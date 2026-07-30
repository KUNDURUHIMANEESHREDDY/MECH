export const experimentTemplates = {
  listTemplates() {
    return [
      { id: 'tmpl_ioi', name: 'Indirect Object Identification', category: 'CircuitDiscovery', defaultModel: 'GPT-2 Small' },
      { id: 'tmpl_induction', name: 'Induction Head Detection', category: 'AttentionPattern', defaultModel: 'GPT-2 Small' },
      { id: 'tmpl_sae_probing', name: 'SAE Feature Probing', category: 'FeatureAnalysis', defaultModel: 'Gemma-2B' },
      { id: 'tmpl_causal_tracing', name: 'Activation Patching & Causal Tracing', category: 'CausalIntervention', defaultModel: 'Llama-3-8B' },
    ];
  },
};
