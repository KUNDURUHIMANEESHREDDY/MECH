/**
 * Unified Scientific Data Model for MECH Platform (TypeScript Definitions).
 *
 * Used by AIResearchAssistantView, RealTimeDAGVisualizer, NeuronPanel,
 * EvidenceFusionView, KnowledgeGraphPanel, and Experiment Notebooks.
 */

export type EvidenceLevel = 'OBSERVED' | 'CANDIDATE' | 'SUPPORTED' | 'CAUSALLY_VERIFIED';

export type MechanismType = 
  | 'attention_routing' 
  | 'mlp_projection' 
  | 'residual_stream' 
  | 'ov_circuit' 
  | 'qk_circuit';

export type TransitionType = 
  | 'emergence' 
  | 'divergence' 
  | 'stabilization' 
  | 'suppression' 
  | 'stable';

export interface SubstrateAnchor {
  layer: number;
  neuron_idx: number;
  weight_norm: number;
  is_polysemantic: boolean;
  role_description: string;
}

export interface FeatureEvidence {
  feature_id: string;
  layer: number;
  latent_idx: number;
  semantic_label: string;
  specificity: number;
  consistency: number;
  cross_prompt_stability: number;
  activation_strength: number;
  causal_effect?: number | null;
  confidence_score: number;
  linear_logit_delta: Record<string, number>; // Δz_i ≈ a_i * (W_U d_i)
  substrate_anchors: SubstrateAnchor[];
  evidence_level: EvidenceLevel;
  is_causally_mediating: boolean;
}

export interface LogitLensPrediction {
  token: string;
  probability: number;
  logit: number;
}

export interface FeatureLogitProjection {
  feature_id: string;
  activation: number;
  target_token: string;
  projected_delta_logit: number;
}

export interface LogitLensTransition {
  layer: number;
  top_token: string;
  probability: number;
  delta_probability: number;
  rank: number;
  is_predictive_transition: boolean;
  transition_type: TransitionType;
  predictions: LogitLensPrediction[];
  active_feature_projections: FeatureLogitProjection[];
}

export interface CircuitPathwayEdge {
  edge_id: string;
  source_id: string;
  target_id: string;
  mechanism_type: MechanismType;
  evidence_level: EvidenceLevel;
  attention_routing_score: number;
  attribution_score: number;
  causal_mediation_effect?: number | null;
  value_flow_description: string;
  query_relation_description: string;
  falsification_report?: string | null;
}

export interface UnifiedScientificReport {
  investigation_id: string;
  model_id: string;
  clean_prompt: string;
  corrupted_prompt?: string | null;
  target_token: string;
  predictive_divergence_layer: number;
  logit_lens_trajectory: LogitLensTransition[];
  candidate_features: FeatureEvidence[];
  circuit_nodes: any[];
  circuit_edges: CircuitPathwayEdge[];
  falsified_hypotheses: any[];
  summary_verdict: string;
}
