const api = typeof window !== 'undefined' ? window.appApi : undefined;

/**
 * VisualizationService - Fetches, transforms, and normalizes DTO payloads
 * from backend API surface for UI panel consumption.
 */
class VisualizationService {
  async fetchSAEFeature(featureId) {
    if (api && api.pythonCall) {
      return api.pythonCall('interpretability/features/inspect', { feature_id: featureId });
    }
    return {
      feature_id: featureId,
      label: `SAE Feature #${featureId}`,
      activation: 4.12,
      connected_neurons: [
        { layer: 8, neuron: 402, weight: 0.85 },
        { layer: 9, neuron: 112, weight: 0.62 },
      ],
      dataset_examples: [
        'John gave a book to Mary',
        'Alice sent a letter to Bob',
      ],
      statistics: { firing_freq: 0.042, max_activation: 4.12, mean_activation: 1.25 },
    };
  }

  async fetchLogitLens(prompt, layer = 11, method = 'logit_lens') {
    const route = method === 'tuned_lens' ? 'interpretability/projections/tuned_lens' : 'interpretability/projections/logit_lens';
    if (api && api.pythonCall) {
      return api.pythonCall(route, { prompt, layer });
    }
    return {
      method: method === 'tuned_lens' ? 'TunedLens' : 'LogitLens',
      prompt,
      layer,
      top_token: ' Paris',
      top_logit: 14.8,
      entropy: 0.65,
      top_k_tokens: [
        { token: ' Paris', logit: 14.8, probability: 0.82 },
        { token: ' France', logit: 12.1, probability: 0.12 },
      ],
    };
  }

  async fetchModelComparison(prompt, modelA = 'GPT-2 Small', modelB = 'Pythia 160M') {
    if (api && api.pythonCall) {
      return api.pythonCall('runtime/compare', { prompt, model_a: modelA, model_b: modelB });
    }
    return {
      prompt,
      model_a: modelA,
      model_b: modelB,
      activations: { cosine_similarity: 0.875, kl_divergence: 0.142 },
      predictions: { top_token_match: true, model_a_top: { token: ' Paris', prob: 0.82 }, model_b_top: { token: ' Paris', prob: 0.79 } },
      attention: { head_alignment_score: 0.91 },
      residuals: { layer_norm_ratio: 1.04 },
    };
  }
}

export const visualizationService = new VisualizationService();
