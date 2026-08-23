import { create } from 'zustand';
import { api } from '../../services/api';
import { KnowledgeType, EvidenceLevel, ExperimentExecutionStatus } from '../../science/types/scientificTypes';
import { useSelectionStore, ResearchComponent } from './selection';
import { computeKnowledgeTypes } from '../utils/evidenceUtils';

export interface Investigation {
  id: string;
  title: string;
  research_question: string;
  model_id: string;
  dataset_id: string;
  current_hypothesis_id?: string;
  status: string;
  tags: string[];
  created_at: number;
  updated_at: number;
}

export interface Hypothesis {
  id: string;
  investigation_id: string;
  title: string;
  statement: string;
  target_component: string;
  prediction: string;
  expected_evidence: string;
  falsification_condition: string;
  status: 'UNTESTED' | 'SUPPORTED' | 'PARTIALLY_SUPPORTED' | 'CONTRADICTED' | 'FALSIFIED' | 'INCONCLUSIVE';
  evidence_count_supporting: number;
  evidence_count_contradicting: number;
  confidence_score?: number;
  created_at: number;
  updated_at: number;
}

export interface ExperimentRun {
  id: string;
  experiment_id: string;
  investigation_id: string;
  timestamp: number;
  model_id: string;
  execution_time_ms: number;
  status: string;
  execution_status: ExperimentExecutionStatus;
  used_mock_data: boolean;
  baseline_target_prob: number;
  intervened_target_prob: number;
  delta_target_prob: number;
  baseline_logit: number;
  intervened_logit: number;
  delta_logit: number;
  control_delta_logit?: number;
  effect_size_cohens_d?: number;
  top_predicted_tokens_clean: Array<{ token: string; probability: number; logit: number }>;
  top_predicted_tokens_intervened: Array<{ token: string; probability: number; logit: number }>;
  is_reproducible: boolean;
  manifest_id?: string;
  provenance_hash?: string;
  logs: string[];

}
export interface EvidenceRecord {
  id: string;
  investigation_id: string;
  hypothesis_id?: string;
  experiment_run_id?: string;
  source_type: string;
  claim: string;
  evidence_level: string;
  supports_hypothesis: boolean;
  metric_name: string;
  metric_value: number;
  baseline_value?: number;
  control_value?: number;
  sample_size: number;
  statistical_details: Record<string, any>;
  provenance_chain: string[];
  created_at: number;
}

export interface Mechanism {
  id: string;
  investigation_id: string;
  name: string;
  description: string;
  nodes: Array<{ id: string; component_type: string; label: string; layer?: number; head?: number; neuron?: number; feature_id?: string; evidence_level: string }>;
  edges: Array<{ id: string; source_node_id: string; target_node_id: string; mechanism_type: string; causal_effect?: number; is_causally_verified: boolean; evidence_ids: string[] }>;
  is_evidence_backed: boolean;
  weakest_link_tier: string;
  created_at: number;
  updated_at: number;
}

export interface ResearchStoreState {
  investigations: Investigation[];
  activeInvestigationId: string | null;
  activeInvestigation: Investigation | null;
  
  hypotheses: Hypothesis[];
  activeHypothesisId: string | null;
  activeHypothesis: Hypothesis | null;

  runs: ExperimentRun[];
  evidence: EvidenceRecord[];
  evidenceMatrix: any[];
  mechanisms: Mechanism[];
  artifacts: any[];
  jobs: any[];
  
  isDemoMode: boolean;
  loading: boolean;
  error: string | null;

  // Scientific epistemology tracking
  knowledgeTypes: KnowledgeType[];  // OBSERVATION | INFERENCE | CAUSAL_EVIDENCE | CLAIM
  evidenceStatus: EvidenceLevel;    // OBSERVED | CANDIDATE | SUPPORTED | CAUSALLY_VERIFIED | FALSIFIED
  lastRun: ExperimentRun | null;
  reproducibility: {is_reproducible: boolean; details: string} | null;

  // Actions
  loadInvestigations: () => Promise<void>;
  selectInvestigation: (id: string) => Promise<void>;
  createInvestigation: (data: Partial<Investigation>) => Promise<Investigation>;
  deleteInvestigation: (id: string) => Promise<void>;

  loadHypotheses: (invId?: string) => Promise<void>;
  selectHypothesis: (id: string) => void;
  createHypothesis: (data: Partial<Hypothesis>) => Promise<Hypothesis>;
  deleteHypothesis: (id: string) => Promise<void>;
  evaluateHypothesis: (id: string) => Promise<any>;

  runCausalExperiment: (exp: Record<string, any>) => Promise<ExperimentRun>;
  loadRuns: (invId?: string) => Promise<void>;
  loadEvidence: (invId?: string) => Promise<void>;
  loadEvidenceMatrix: (invId?: string) => Promise<void>;
  loadMechanisms: (invId?: string) => Promise<void>;
  saveMechanism: (data: Partial<Mechanism>) => Promise<Mechanism>;
  deleteMechanism: (id: string) => Promise<void>;
  generateReport: (invId?: string) => Promise<string>;

  setDemoMode: (val: boolean) => void;
  
  /** @deprecated Use useSelectionStore.setSelectedResearchComponent instead */
  selectComponent: (comp: ResearchComponent | null) => void;
  /** @deprecated Use useSelectionStore.setSelectedTokenIndex instead */
  selectToken: (idx: number | null) => void;
}

export const useResearchStore = create<ResearchStoreState>((set, get) => ({
  investigations: [],
  activeInvestigationId: null,
  activeInvestigation: null,
  hypotheses: [],
  activeHypothesisId: null,
  activeHypothesis: null,
  runs: [],
  evidence: [],
  evidenceMatrix: [],
  mechanisms: [],
  artifacts: [],
  jobs: [],
  knowledgeTypes: [] as KnowledgeType[],
  evidenceStatus: 'OBSERVED' as EvidenceLevel,
  lastRun: null as ExperimentRun | null,
  reproducibility: null as {is_reproducible: boolean; details: string} | null,
  isDemoMode: false,
  loading: false,
  error: null,

  loadInvestigations: async () => {
    set({ loading: true, error: null });
    try {
      const res = await api.listInvestigations();
      const items = res.investigations || [];
      const currentActiveId = get().activeInvestigationId || (items.length > 0 ? items[0].id : null);
      const activeInv = items.find((i: Investigation) => i.id === currentActiveId) || (items.length > 0 ? items[0] : null);
      
      set({
        investigations: items,
        activeInvestigationId: activeInv?.id || null,
        activeInvestigation: activeInv || null,
        loading: false,
      });

      if (activeInv) {
        await get().selectInvestigation(activeInv.id);
      }
    } catch (err: any) {
      set({ error: err.message || 'Failed to load investigations', loading: false });
    }
  },

  selectInvestigation: async (id: string) => {
    set({ loading: true, error: null });
    try {
      const res = await api.getInvestigation(id);
      const inv = res.investigation;
      const hyps = res.hypotheses || [];
      const activeHyp = hyps.find((h: Hypothesis) => h.id === inv.current_hypothesis_id) || (hyps.length > 0 ? hyps[0] : null);

      const evidence = res.evidence || [];
      const { knowledgeTypes, evidenceStatus } = computeKnowledgeTypes(evidence);

      const runs = res.runs || [];
      const lastRun = runs.length > 0 ? runs[runs.length - 1] : null;

      const allHaveManifest = runs.every((r: any) => r.manifest_id && r.provenance_hash);
      const reproducibilityInfo = {
        is_reproducible: allHaveManifest && runs.length > 0,
        details: allHaveManifest ? 'All runs have provenance hashes' : 'Some runs missing provenance',
      };

      set({
        activeInvestigationId: id,
        activeInvestigation: inv,
        hypotheses: hyps,
        activeHypothesisId: activeHyp?.id || null,
        activeHypothesis: activeHyp || null,
        runs: runs || [],
        evidence: evidence || [],
        knowledgeTypes,
        evidenceStatus,
        lastRun,
        reproducibility: reproducibilityInfo,
        mechanisms: res.mechanisms || [],
        loading: false,
      });

      await get().loadEvidenceMatrix(id);
    } catch (err: any) {
      set({ error: err.message || 'Failed to fetch investigation details', loading: false });
    }
  },

  createInvestigation: async (data: Partial<Investigation>) => {
    set({ loading: true, error: null });
    try {
      const res = await api.createInvestigation(data as Record<string, unknown>);
      const newInv = res.investigation;
      await get().loadInvestigations();
      await get().selectInvestigation(newInv.id);
      set({ loading: false });
      return newInv;
    } catch (err: any) {
      set({ error: err.message, loading: false });
      throw err;
    }
  },

  deleteInvestigation: async (id: string) => {
    try {
      await api.deleteInvestigation(id);
      await get().loadInvestigations();
    } catch (err: any) {
      set({ error: err.message });
    }
  },

  loadHypotheses: async (invId?: string) => {
    const targetId = invId || get().activeInvestigationId;
    if (!targetId) return;
    try {
      const res = await api.listHypotheses(targetId);
      const items = res.hypotheses || [];
      const activeHyp = items.find((h: Hypothesis) => h.id === get().activeHypothesisId) || (items.length > 0 ? items[0] : null);
      set({ hypotheses: items, activeHypothesis: activeHyp, activeHypothesisId: activeHyp?.id || null });
    } catch (err: any) {
      set({ error: err.message });
    }
  },

  selectHypothesis: (id: string) => {
    const hyp = get().hypotheses.find((h) => h.id === id) || null;
    set({ activeHypothesisId: id, activeHypothesis: hyp });
    if (hyp && hyp.target_component) {
      const comp = hyp.target_component;
      if (comp.startsWith('L') && comp.includes('H')) {
        const parts = comp.split('H');
        const layer = parseInt(parts[0].replace('L', ''), 10);
        const head = parseInt(parts[1], 10);
        useSelectionStore.getState().setSelectedResearchComponent({ name: comp, layer, head, componentType: 'head' });
      }
    }
  },

  createHypothesis: async (data: Partial<Hypothesis>) => {
    const invId = data.investigation_id || get().activeInvestigationId;
    if (!invId) throw new Error('No active investigation.');
    const payload = { ...data, investigation_id: invId };
    const res = await api.createHypothesis(payload as Record<string, unknown>);
    await get().loadHypotheses(invId);
    set({ activeHypothesisId: res.hypothesis.id, activeHypothesis: res.hypothesis });
    return res.hypothesis;
  },

  deleteHypothesis: async (id: string) => {
    await api.deleteHypothesis(id);
    await get().loadHypotheses();
  },

  evaluateHypothesis: async (id: string) => {
    const invId = get().activeInvestigationId;
    if (!invId) return;
    const res = await api.evaluateHypothesis(id, invId);
    await get().loadHypotheses(invId);
    await get().loadEvidence(invId);
    await get().loadEvidenceMatrix(invId);
    
    const evidence = get().evidence;
    const { knowledgeTypes, evidenceStatus } = computeKnowledgeTypes(evidence);

    set({ knowledgeTypes, evidenceStatus });
    return res;
  },

  runCausalExperiment: async (exp: Record<string, any>) => {
    set({ loading: true, error: null });
    const invId = exp.investigation_id || get().activeInvestigationId;
    const hypId = exp.hypothesis_id || get().activeHypothesisId;
    const payload: Record<string, any> = {
      ...exp,
      investigation_id: invId,
      hypothesis_id: hypId,
    };
    try {
      const res = await api.runCausalExperiment(payload);
      // Agent 1 fails closed with a UNEXECUTED_EXPERIMENT envelope when the
      // experiment was never run on live weights (e.g. model unavailable,
      // validation error, epistemic gate). Surface that as a NOT_EXECUTABLE
      // run instead of crashing on an undefined run object.
      if (res.status === 'UNEXECUTED_EXPERIMENT' || !res.run) {
        const reason = (res as any).error || 'Experiment was not executed on live model weights.';
        set({ error: reason, loading: false });
        throw new Error(reason);
      }
      const run = res.run;
      // Map the API response to ExperimentRun with execution status awareness
      const experimentRun: ExperimentRun = {
        id: run.id,
        experiment_id: run.experiment_id || payload.experiment_id || '',
        investigation_id: invId || '',
        timestamp: run.timestamp || Date.now() / 1000,
        model_id: run.model_id || 'gpt2',
        execution_time_ms: run.execution_time_ms || 0,
        status: run.status || 'COMPLETED',
        execution_status: (run.execution_status || run.status || 'COMPLETED') as ExperimentExecutionStatus,
        used_mock_data: run.used_mock_data || false,
        baseline_target_prob: run.baseline_target_prob || 0,
        intervened_target_prob: run.intervened_target_prob || 0,
        delta_target_prob: run.delta_target_prob || 0,
        baseline_logit: run.baseline_logit || 0,
        intervened_logit: run.intervened_logit || 0,
        delta_logit: run.delta_logit || 0,
        control_delta_logit: run.control_delta_logit,
        effect_size_cohens_d: run.effect_size_cohens_d,
        top_predicted_tokens_clean: run.top_predicted_tokens_clean || [],
        top_predicted_tokens_intervened: run.top_predicted_tokens_intervened || [],
        is_reproducible: run.is_reproducible !== undefined ? run.is_reproducible : true,
        manifest_id: run.manifest_id,
        provenance_hash: run.provenance_hash,
        logs: run.logs || [],
      };
      set((state) => ({
        runs: [experimentRun, ...state.runs],
        loading: false,
      }));
      if (invId) {
        await get().loadEvidence(invId);
        await get().loadHypotheses(invId);
        await get().loadEvidenceMatrix(invId);
      }
      return experimentRun;
    } catch (err: any) {
      set({ error: err.message, loading: false });
      throw err;
    }
  },

  loadRuns: async (invId?: string) => {
    try {
      const targetId = invId || get().activeInvestigationId;
      const res = await api.listRuns(undefined, targetId || undefined);
      const runs: ExperimentRun[] = (res.runs || []).map((run: any) => ({
        id: run.id,
        experiment_id: run.experiment_id || '',
        investigation_id: run.investigation_id || '',
        timestamp: run.timestamp || Date.now() / 1000,
        model_id: run.model_id || 'gpt2',
        execution_time_ms: run.execution_time_ms || 0,
        status: run.status || 'COMPLETED',
        execution_status: (run.execution_status || run.status || 'COMPLETED') as ExperimentExecutionStatus,
        used_mock_data: run.used_mock_data,
        baseline_target_prob: run.baseline_target_prob || 0,
        intervened_target_prob: run.intervened_target_prob || 0,
        delta_target_prob: run.delta_target_prob || 0,
        baseline_logit: run.baseline_logit || 0,
        intervened_logit: run.intervened_logit || 0,
        delta_logit: run.delta_logit || 0,
        control_delta_logit: run.control_delta_logit,
        effect_size_cohens_d: run.effect_size_cohens_d,
        top_predicted_tokens_clean: run.top_predicted_tokens_clean || [],
        top_predicted_tokens_intervened: run.top_predicted_tokens_intervened || [],
        is_reproducible: run.is_reproducible !== undefined ? run.is_reproducible : true,
        manifest_id: run.manifest_id,
        provenance_hash: run.provenance_hash,
        logs: run.logs || [],
  }));
      set({ runs, error: null });
    } catch (err: any) {
      set({ error: err.message || 'Failed to load runs' });
    }
  },

  loadEvidence: async (invId?: string) => {
    try {
      const targetId = invId || get().activeInvestigationId;
      if (!targetId) return;
      const res = await api.listEvidence(targetId);
      const evidence = res.evidence || [];
    
      const { knowledgeTypes, evidenceStatus } = computeKnowledgeTypes(evidence);

      set({ evidence, knowledgeTypes, evidenceStatus, error: null });
    } catch (err: any) {
      set({ error: err.message || 'Failed to load evidence' });
    }
  },

  loadEvidenceMatrix: async (invId?: string) => {
    const targetId = invId || get().activeInvestigationId;
    if (!targetId) return;
    try {
      const res = await api.getEvidenceMatrix(targetId);
      set({ evidenceMatrix: res.matrix || [], error: null });
    } catch { }
  },

  loadMechanisms: async (invId?: string) => {
    try {
      const targetId = invId || get().activeInvestigationId;
      if (!targetId) return;
      const res = await api.listMechanisms(targetId);
      set({ mechanisms: res.mechanisms || [], error: null });
    } catch (err: any) {
      set({ error: err.message || 'Failed to load mechanisms' });
    }
  },

  saveMechanism: async (data: Partial<Mechanism>) => {
    const invId = data.investigation_id || get().activeInvestigationId;
    const payload = { ...data, investigation_id: invId };
    const res = await api.createMechanism(payload as Record<string, unknown>);
    await get().loadMechanisms(invId ?? undefined);
    return res.mechanism;
  },

  deleteMechanism: async (id: string) => {
    await api.deleteMechanism(id);
    await get().loadMechanisms();
  },

  generateReport: async (invId?: string) => {
    const targetId = invId || get().activeInvestigationId;
    if (!targetId) throw new Error('No active investigation.');
    const res = await api.generateResearchReport(targetId);
    return res.report_markdown;
  },

  selectComponent: (comp) => {
    useSelectionStore.getState().setSelectedResearchComponent(comp);
  },

  selectToken: (idx) => {
    useSelectionStore.getState().setSelectedTokenIndex(idx);
  },

  setDemoMode: (val) => {
    set({ isDemoMode: val });
  },
}));
