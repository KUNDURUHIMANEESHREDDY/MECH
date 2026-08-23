/**
 * Research Pipeline Adapter for MECH Platform.
 *
 * Normalizes backend investigation responses into the canonical 6-stage
 * scientific discovery pipeline format for UI rendering.
 */

import { UnifiedScientificReport, FeatureEvidence, CircuitPathwayEdge } from '../types/scientificTypes';

export interface FormattedPipelineData {
  investigationId: string;
  modelName: string;
  cleanPrompt: string;
  corruptedPrompt: string;
  targetToken: string;
  divergenceLayer: number;
  logitLensTrajectory: any[];
  candidateFeatures: FeatureEvidence[];
  circuitNodes: any[];
  circuitEdges: CircuitPathwayEdge[];
  falsifiedHypotheses: any[];
  verdict: string;
}

export function adaptScientificReport(report: UnifiedScientificReport): FormattedPipelineData {
  return {
    investigationId: report.investigation_id,
    modelName: report.model_id,
    cleanPrompt: report.clean_prompt,
    corruptedPrompt: report.corrupted_prompt || '',
    targetToken: report.target_token,
    divergenceLayer: report.predictive_divergence_layer,
    logitLensTrajectory: report.logit_lens_trajectory,
    candidateFeatures: report.candidate_features,
    circuitNodes: report.circuit_nodes,
    circuitEdges: report.circuit_edges,
    falsifiedHypotheses: report.falsified_hypotheses,
    verdict: report.summary_verdict,
  };
}
