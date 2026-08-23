/**
 * Unified Scientific Data Model for MECH Platform.
 * 
 * Strict epistemological distinction between four knowledge types:
 * OBSERVATION - directly measured
 * INFERENCE - concluded from observations
 * CAUSAL EVIDENCE - intervention demonstrates measurable effect
 * CLAIM - researcher-level conclusion built from evidence
 */

// ── Epistemology Levels ──────────────────────────────────────────────

/** Something directly measured (e.g. "L8H3 attends strongly to token position 4"). */
export type KnowledgeType = 'OBSERVATION' | 'INFERENCE' | 'CAUSAL_EVIDENCE' | 'CLAIM';

export type EvidenceLevel = 'OBSERVED' | 'CANDIDATE' | 'SUPPORTED' | 'CAUSALLY_VERIFIED' | 'FALSIFIED';

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

export interface FeatureLogitProjection {
  feature_id: string;
  activation: number;
  target_token: string;
  projected_delta_logit: number;
  methodology: string;  // e.g. "linear_logit_attribution", "sae_probing"
  sample_size: number;
}

export interface FeatureEvidence {
  feature_id: string;
  layer: number;
  latent_idx: number;
  semantic_label?: string;  // candidate interpretation, not ground truth
  knowledge_type: KnowledgeType;  // OBSERVATION | INFERENCE | CAUSAL_EVIDENCE | CLAIM
  evidence_level: EvidenceLevel;
  methodology: string;  // how was this measured/computed?
  dataset_version: string;
  model_version: string;
  sample_size: number;
  uncertainty?: number;  // quantified uncertainty, not confidence_score
  causal_effect?: number | null;  // only for CAUSAL_EVIDENCE
  linear_logit_delta?: Record<string, number>; // Δz_i ≈ a_i * (W_U d_i), only if measured
  substrate_anchors?: SubstrateAnchor[];
  // NOT ALLOWED: specificity, consistency, cross_prompt_stability, confidence_score
  // These must have defensible definitions before use.
}

export interface LogitLensPrediction {
  token: string;
  probability: number;
  logit: number;
  knowledge_type: KnowledgeType;
  methodology: string;  // e.g. "logit_lens_forward_pass"
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
  knowledge_type: KnowledgeType;
  methodology: string;  // e.g. "logit_lens_trajectory_computation"
}

export interface ControlMeasurement {
  control_name: string;
  control_type: string;  // e.g. "random_head", "neighboring_head", "unrelated_layer"
  component_id: string;
  layer: number;
  index: number;
  selection_rationale: string;  // why was this control chosen?
  delta_logit: number;
  delta_prob: number;
  methodology: string;  // e.g. "ablation_mean_over_5_samples", "activation_patching_3_controls"
}

export interface ContinuousCausalResult {
  prompt_id: string;
  prompt: string;
  target_token: string;
  target_component: string;
  component_type: string;
  layer: number;
  index: number;
  intervention_type: string;
  knowledge_type: KnowledgeType;  // OBSERVATION | INFERENCE | CAUSAL_EVIDENCE | CLAIM
  evidence_level: EvidenceLevel;
  methodology: string;  // e.g. "activation_patching_4_controls", "ablation_mean"
  dataset_version: string;
  model_version: string;
  sample_size: number;
  uncertainty?: number;  // e.g. std_delta_logit / sqrt(N), not "94%"
  causal_effect?: number | null;  // only for CAUSAL_EVIDENCE
  // NOT ALLOWED: robustness scores, specificity ratios without definitions
  controls?: ControlMeasurement[];
  mean_control_delta_logit?: number;
  max_control_delta_logit?: number;
  evidence_tier: EvidenceLevel | 'FALSIFIED';
  verdict: string;
}

export interface EdgeEvidenceObject {
  logit_lens_stage: string;
  sae_association?: string;
  knowledge_type: KnowledgeType;  // OBSERVATION | INFERENCE | CAUSAL_EVIDENCE | CLAIM
  attention_routing_score?: number;  // only if OBSERVATION with methodology
  attribution_score?: number;  // only if OBSERVATION with methodology
  causal_effect?: number | null;  // only for CAUSAL_EVIDENCE
  robust_specificity?: number | null;  // must have defensible definition
  control_results?: ControlMeasurement[];
  cross_prompt_replicated?: boolean;
  epistemic_scope: string;
  methodology: string;  // how was this edge's score computed?
}

export interface CircuitPathwayEdge {
  edge_id: string;
  source_id: string;
  target_id: string;
  mechanism_type: MechanismType;
  evidence_level: EvidenceLevel;
  knowledge_type: KnowledgeType;  // OBSERVATION | INFERENCE | CAUSAL_EVIDENCE | CLAIM
  attention_routing_score?: number;  // only if OBSERVATION with methodology
  attribution_score?: number;  // only if OBSERVATION with methodology
  causal_mediation_effect?: number | null;  // only for CAUSAL_EVIDENCE
  value_flow_description: string;
  query_relation_description: string;
  evidence_object?: EdgeEvidenceObject;
  falsification_report?: string | null;
}

export interface CircuitCompositionReport {
  pathway_id: string;
  edges: CircuitPathwayEdge[];
  composed_causal_tier: EvidenceLevel;
  weakest_link_edge_id: string;
  weakest_link_evidence_tier: EvidenceLevel;
  composition_principle: string;  // e.g. "end_to_end_patching", "residual_stream_flow"
  end_to_end_path_patching_effect?: number | null;  // only if CAUSAL_EVIDENCE with methodology
  epistemic_scope: string;  // e.g. "single-prompt", "cross-prompt"
}

export interface PathStepMeasurement {
  step_name: string;
  intervention_target: string;
  intervention_type: string;
  clean_logit: number;
  intervened_logit: number;
  delta_logit: number;
  clean_prob: number;
  intervened_prob: number;
  delta_prob: number;
  description: string;
}

export interface NullPathDistribution {
  control_path_count: number;
  control_path_deltas: number[];
  mean_null_delta: number;
  median_null_delta: number;
  max_null_delta: number;
  observed_path_percentile: number;
  empirical_p_value: number;
}

export interface MediationRescueResult {
  source_node: string;
  mediator_node: string;
  target_token: string;
  clean_source_activation: number;
  clean_mediator_activation: number;
  ablated_source_logit: number;
  ablated_mediator_logit: number;
  rescued_logit: number;
  rescue_delta_recovery: number;
  rescue_fraction: number;
  formal_mediation_status: 'CONFIRMED_CAUSAL_MEDIATOR' | 'PARTIAL_RESCUE' | 'BYSTANDER_NON_MEDIATING';
  rescue_verdict: string;
}

export interface PathwayVerificationReport {
  pathway_id: string;
  prompt: string;
  target_token: string;
  node_chain: string[];
  edge_chain: string[];
  step_measurements: PathStepMeasurement[];
  path_contribution_fraction: number;
  mediation_rescue: MediationRescueResult;
  null_distribution: NullPathDistribution;
  total_prompt_effect: number;
  node_effect: number;
  edge_effects: number[];
  composite_path_effect: number;
  direct_bypass_effect: number;
  path_specificity_ratio: number;
  edge_causal_status: string;
  path_causal_status: 'END_TO_END_VERIFIED' | 'PARTIALLY_MEDIATED' | 'NON_MEDIATING' | 'FALSIFIED';
  path_verdict: string;
  epistemic_scope: string[];
}

export interface FalsificationEntry {
  pathway: string;
  mechanism_type: string;
  knowledge_type: KnowledgeType;  // should be CAUSAL_EVIDENCE or FALSIFIED
  attention_routing_score?: number;  // only if observed
  causal_effect: number;  // must have measurement provenance
  falsification_verdict: string;
  methodology: string;  // how was falsification attempted?
}

export interface CrossPromptCausalReport {
  target_component: string;
  component_type: string;
  layer: number;
  index: number;
  prompt_count: number;
  mean_delta_logit: number;
  median_delta_logit: number;
  std_delta_logit: number;
  iqr_delta_logit: number;
  ci_95_lower: number;
  ci_95_upper: number;
  expected_sign_rate: number;
  mean_delta_prob: number;
  cross_prompt_stability?: number;  // only if method defensibly defined
  mediation_fraction?: number;  // only if method defensibly defined
  mean_specificity_ratio?: number;  // only if method defensibly defined
  overall_evidence_tier: EvidenceLevel | 'FALSIFIED';
  knowledge_type: KnowledgeType;  // should be CAUSAL_EVIDENCE or CLAIM
  prompt_evaluations: ContinuousCausalResult[];
  falsification_summary: string;
  promotion_reasons?: string[];  // must reference actual experiments/evidence
  evidence_scope?: string[];  // e.g. ["IOI", "indirect-object"]
  remaining_limitations?: string[];  // e.g. ["single-prompt only", "no cross-digit generalization"]
}

export interface UnifiedScientificReport {
  investigation_id: string;
  model_id: string;
  clean_prompt: string;
  corrupted_prompt?: string | null;
  target_token: string;
  predictive_divergence_layer: number;
  maximum_predictive_gain_layer?: number;
  max_delta_probability?: number;
  target_probability_trajectory?: number[];
  target_rank_trajectory?: number[];
  target_logit_trajectory?: number[];
  logit_lens_trajectory: LogitLensTransition[];
  candidate_features: FeatureEvidence[];
  circuit_nodes: any[];
  circuit_edges: CircuitPathwayEdge[];
  circuit_composition?: CircuitCompositionReport;
  falsified_hypotheses: FalsificationEntry[];
  summary_verdict: string;
  cross_prompt_causal_report?: CrossPromptCausalReport;
  // Provenance, not fabricated
  manifest_id?: string;
  provenance_hash?: string;
  evidence_level: EvidenceLevel;
  statistical_caveat: string;  // mandatory human-readable caveat
}

/* ── Phase 11: Immutable Experiment & Reproducibility Types ──────────────── */

export interface ModelIdentity {
  model_id: string;
  architecture: string;
  parameter_count: number;
  weights_hash: string;
  revision_or_commit: string;
  tokenizer_hash: string;
  config_hash: string;
  source: string;
  precision: string;
  execution_strategy: string;
}

export interface ExecutionEnvironment {
  python_version: string;
  pytorch_version: string;
  transformers_version: string;
  cuda_version?: string | null;
  gpu_name?: string | null;
  gpu_compute_capability?: string | null;
  os_platform: string;
  os_release: string;
  mech_version: string;
  git_commit_sha: string;
  dependency_lock_hash: string;
  deterministic_mode: boolean;
}

export interface ExperimentSpecification {
  clean_prompt: string;
  target_token: string;
  corrupted_prompt?: string | null;
  distractor_token?: string | null;
  target_component: string;
  component_type: string;
  layer: number;
  component_index: number;
  intervention_type: string;
  ablation_scale: number;
  random_seed: number;
  control_battery_spec?: any[];
  cross_prompt_suite?: [string, string][];
  knowledge_type?: KnowledgeType;  // tracks the type of knowledge this experiment produces
  methodology: string;  // detailed description of how intervention was executed
}

export interface ProvenanceChain {
  logit_lens_divergence_layer?: number;
  maximum_predictive_gain_layer?: number;
  active_sae_candidates?: string[];
  dense_substrate_anchors?: string[];
  linear_projection_delta?: Record<string, number>;
  edge_causal_effect?: number;
  four_control_results?: any[];
  cross_prompt_stability?: number;  // only if method defensibly defined
  mediation_rescue_fraction?: number;  // only if method defensibly defined
  null_distribution_percentile?: number;
  null_distribution_p_value?: number;
  final_evidence_tier: EvidenceLevel | 'FALSIFIED';
  epistemic_scope: string[];  // e.g. ["single-prompt", "cross-prompt"]
  methodology: string;  // how was the provenance computed?
  knowledge_type: KnowledgeType;  // tracks the type of knowledge established
}

export interface ImmutableExperimentRun {
  run_id: string;
  parent_run_id?: string | null;
  experiment_type: 'ORIGINAL' | 'REPRODUCTION' | 'REPLICATION';
  title: string;
  timestamp_utc: string;
  model: ModelIdentity;
  environment: ExecutionEnvironment;
  specification: ExperimentSpecification;
  provenance_chain: ProvenanceChain;
  measurements: Record<string, any>;
  verdict: string;
  manifest_sha256: string;
  knowledge_type: KnowledgeType;  // the type of knowledge this run produces
  status: 'COMPLETED' | 'FAILED' | 'CANCELLED' | 'REPRODUCIBLE' | 'NOT_REPRODUCIBLE';
  execution_status: ExperimentExecutionStatus;  // PENDING, RUNNING, COMPLETED, FAILED, NOT_EXECUTABLE
  used_mock_data: boolean;  // True if results are from mock/synthetic data, not live model
}

export type ExperimentExecutionStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'NOT_EXECUTABLE';

export interface ComponentToleranceResult {
  metric_name: string;
  expected_value: any;
  observed_value: any;
  abs_diff?: number | null;
  rel_diff?: number | null;
  tolerance_threshold: string;
  passed: boolean;
  status: 'EXACT' | 'WITHIN_TOLERANCE' | 'MISMATCH';
  details: string;
}

export interface ReproductionComparisonReport {
  original_run_id: string;
  reproduction_run_id: string;
  timestamp_utc: string;
  overall_reproduced: boolean;
  numerical_tolerance_verdict: string;
  component_comparisons: ComponentToleranceResult[];
  execution_duration_ms: number;
  model_matched: boolean;
  environment_matched: boolean;
}

