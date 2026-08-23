import { 
  UnifiedScientificReport, 
  ContinuousCausalResult, 
  CrossPromptCausalReport,
  PathwayVerificationReport,
  ImmutableExperimentRun,
  ExperimentExecutionStatus,
  ReproductionComparisonReport
} from '../types/scientificTypes';

const API_BASE = 'http://127.0.0.1:8000/api/v1/science';

export interface TrajectoryParams {
  prompt: string;
  model_id?: string;
  target_token?: string;
  distractor_token?: string;
}

export interface InvestigationParams {
  clean_prompt: string;
  corrupted_prompt?: string;
  target_token?: string;
  model_id?: string;
}

export interface EvaluateCausalityParams {
  prompt: string;
  target_token: string;
  layer: number;
  component_type: string;
  component_index: number;
  prompt_id?: string;
  ablation_scale?: number;
  model_id?: string;
}

export interface EvaluateCrossPromptParams {
  prompts: [string, string][];
  layer: number;
  component_type: string;
  component_index: number;
  ablation_scale?: number;
  model_id?: string;
}

export interface VerifyPathwayParams {
  edge_id: string;
  model_id?: string;
  clean_prompt?: string;
  corrupted_prompt?: string;
  target_token?: string;
}

export interface VerifyFullPathwayParams {
  pathway_id?: string;
  clean_prompt: string;
  target_token: string;
  node_chain?: string[];
  edge_chain?: string[];
  layer?: number;
  model_id?: string;
}

export const scienceApi = {
  async fetchPredictiveTrajectory(params: TrajectoryParams) {
    const res = await fetch(`${API_BASE}/predictive-trajectory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: params.prompt,
        model_id: params.model_id || 'gpt2',
        target_token: params.target_token || ' Paris',
        distractor_token: params.distractor_token || ' London',
      }),
    });
    if (!res.ok) throw new Error(`Logit Lens trajectory fetch failed: ${res.statusText}`);
    return res.json();
  },

  async fetchLayerFeatures(layer: number = 0, model_id: string = 'gpt2') {
    const res = await fetch(`${API_BASE}/layer-features/${layer}?model_id=${model_id}`);
    if (!res.ok) throw new Error(`Layer features fetch failed: ${res.statusText}`);
    const env = await res.json();
    // The science endpoints return a fail-closed ScientificResponseEnvelope.
    // Unwrap it: an UNEXECUTED result carries no payload and MUST be surfaced
    // as an empty/absent feature set — never as fabricated candidates.
    if (env?.result && Array.isArray(env.result.result)) {
      return env.result.result;
    }
    return [];
  },

  async fetchUnifiedInvestigationReport(params: InvestigationParams): Promise<UnifiedScientificReport> {
    const res = await fetch(`${API_BASE}/unified-investigation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        clean_prompt: params.clean_prompt,
        corrupted_prompt: params.corrupted_prompt,
        target_token: params.target_token || ' Paris',
        model_id: params.model_id || 'gpt2',
      }),
    });
    if (!res.ok) throw new Error(`Unified investigation report fetch failed: ${res.statusText}`);
    return res.json();
  },

  async evaluateCausality(params: EvaluateCausalityParams): Promise<ContinuousCausalResult> {
    const res = await fetch(`${API_BASE}/evaluate-causality`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Causal evaluation failed: ${res.statusText}`);
    return res.json();
  },

  async evaluateCrossPromptCausality(params: EvaluateCrossPromptParams): Promise<CrossPromptCausalReport> {
    const res = await fetch(`${API_BASE}/evaluate-cross-prompt-causality`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Cross-prompt causal evaluation failed: ${res.statusText}`);
    return res.json();
  },

  async verifyPathway(params: VerifyPathwayParams): Promise<{
    edge_id: string;
    previous_evidence_level: string;
    new_evidence_level: string;
    causal_mediation_effect: number;
    is_verified: boolean;
    falsification_verdict: string;
  }> {
    const res = await fetch(`${API_BASE}/verify-pathway`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Pathway verification failed: ${res.statusText}`);
    return res.json();
  },

  async verifyFullPathway(params: VerifyFullPathwayParams): Promise<PathwayVerificationReport> {
    const res = await fetch(`${API_BASE}/verify-full-pathway`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Full pathway verification failed: ${res.statusText}`);
    return res.json();
  },

  /* ── Phase 11: Immutable Experiment & Reproducibility APIs ─────────── */

  async saveExperiment(params: any): Promise<ImmutableExperimentRun> {
    const res = await fetch(`${API_BASE}/experiments/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Save experiment failed: ${res.statusText}`);
    const envelope = await res.json();
    if (!envelope || !envelope.result) {
      throw new Error(envelope?.statistical_caveat || 'Experiment is UNEXECUTED (no model loaded).');
    }
    return envelope.result;
  },

  async listExperiments(experimentType?: string): Promise<ImmutableExperimentRun[]> {
    const query = experimentType ? `?experiment_type=${encodeURIComponent(experimentType)}` : '';
    const res = await fetch(`${API_BASE}/experiments/list${query}`);
    if (!res.ok) throw new Error(`List experiments failed: ${res.statusText}`);
    return res.json();
  },

  async getExperiment(runId: string): Promise<ImmutableExperimentRun> {
    const res = await fetch(`${API_BASE}/experiments/${encodeURIComponent(runId)}`);
    if (!res.ok) throw new Error(`Get experiment failed: ${res.statusText}`);
    return res.json();
  },

  async getExperimentLineage(runId: string): Promise<ImmutableExperimentRun[]> {
    const res = await fetch(`${API_BASE}/experiments/${encodeURIComponent(runId)}/lineage`);
    if (!res.ok) throw new Error(`Get experiment lineage failed: ${res.statusText}`);
    return res.json();
  },

  async reproduceExperiment(runId: string): Promise<{
    reproduction_run: ImmutableExperimentRun;
    tolerance_report: ReproductionComparisonReport;
  }> {
    const res = await fetch(`${API_BASE}/experiments/${encodeURIComponent(runId)}/reproduce`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Reproduce experiment failed: ${res.statusText}`);
    return res.json();
  },

  async replicateExperiment(
    runId: string,
    params: { new_prompt: string; new_target_token: string; title?: string }
  ): Promise<ImmutableExperimentRun> {
    const res = await fetch(`${API_BASE}/experiments/${encodeURIComponent(runId)}/replicate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Replicate experiment failed: ${res.statusText}`);
    return res.json();
  },

  async getExperimentArchive(runId: string): Promise<Record<string, string>> {
    const res = await fetch(`${API_BASE}/experiments/${encodeURIComponent(runId)}/archive`);
    if (!res.ok) throw new Error(`Get experiment archive failed: ${res.statusText}`);
    return res.json();
  },

  async verifyExperimentIntegrity(runId: string): Promise<{
    run_id: string;
    is_valid: boolean;
    message: string;
  }> {
    const res = await fetch(`${API_BASE}/experiments/${encodeURIComponent(runId)}/verify-integrity`);
    if (!res.ok) throw new Error(`Verify experiment integrity failed: ${res.statusText}`);
    return res.json();
  },
};

