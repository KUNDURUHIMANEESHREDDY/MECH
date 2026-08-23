// Allow direct HTTP mode for web browser / standalone backend.
// In Electron, the sidecar bridge via `window.desktopApi.httpRequest` is used instead.
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const isElectron = typeof window !== 'undefined' && !!window.desktopApi;

let cachedApiKey: string | null = null;

async function getApiKeyFromSidecar(): Promise<string | null> {
  if (typeof window === 'undefined') return null;
  if (window.desktopApi?.getApiKey) {
    try {
      return await window.desktopApi.getApiKey();
    } catch {
      return null;
    }
  }
  return null;
}

export async function getApiKey(): Promise<string> {
  if (cachedApiKey) return cachedApiKey;

  // 1. Try Electron sidecar key
  const sidecarKey = await getApiKeyFromSidecar();
  if (sidecarKey) {
    cachedApiKey = sidecarKey;
    return cachedApiKey;
  }

  // 2. Try env var in web / direct HTTP mode
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_KEY) {
    cachedApiKey = import.meta.env.VITE_API_KEY;
    return cachedApiKey;
  }

  // 3. No auth configured (sidecar will reject or backend auth disabled in dev)
  return '';
}

async function httpRequest<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  // In Electron, route through the sidecar bridge
  if (isElectron && window.desktopApi?.httpRequest) {
    const method = options.method || 'GET';
    const body = options.body ? JSON.parse(options.body as string) : undefined;
    const result = await window.desktopApi.httpRequest({
      method,
      path,
      body,
    });
    return result as T;
  }

  // Direct HTTP mode (web browser or standalone backend)
  const apiKey = await getApiKey();
  const headers: Record<string, string> = {};
  if (options.headers) {
    Object.assign(headers, options.headers as Record<string, string>);
  }
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }

  const res = await fetch(`${BACKEND_URL}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status} for ${path}: ${res.statusText}`);
  }

  // Handle empty responses
  const text = await res.text();
  if (!text) return {} as T;
  return JSON.parse(text) as T;
}

async function get<T>(path: string): Promise<T> {
  return httpRequest<T>(path, { method: 'GET' });
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  return httpRequest<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
}

async function del<T>(path: string): Promise<T> {
  return httpRequest<T>(path, { method: 'DELETE' });
}

export const api = {
  pythonPing: () => get<{ status: string }>('/api/status'),
  pythonCall: (method: string, params: Record<string, unknown>) =>
    post<Record<string, unknown>>(`/api/${method}`, params),

  listModels: () => get<{ models: string[] }>('/api/models'),
  loadModel: (name: string) => post<{ status: string }>('/api/models/load', { model_name: name }),
  getModelInfo: (name: string) => get<Record<string, unknown>>(`/api/models/${name}`),
  listSkills: () =>
    get<{
      skills: Array<{
        id: string;
        name: string;
        description: string;
        origin?: string;
        license?: string;
        path: string;
        resources?: Record<string, number>;
      }>;
      count: number;
      source: string;
    }>('/api/skills'),
  infer: (prompt: string, model?: string) =>
    post<Record<string, unknown>>('/api/infer', { prompt, model_name: model }),

  getSessionProbes: () =>
    get<{ status: string; probes: Array<{ probe_id: string; category: string; clean_prompt: string; target_token: string; corrupted_prompt: string; distractor_token: string }>; count: number }>('/api/session/probes'),
  refreshSessionProbes: () =>
    post<{ status: string; probes: Array<{ probe_id: string; category: string; clean_prompt: string; target_token: string; corrupted_prompt: string; distractor_token: string }>; count: number }>('/api/session/probes/refresh'),

  gpt2Load: (model?: string) =>
    post<{ status: string; model_name: string }>('/api/gpt2/load', { model_name: model ?? 'gpt2' }),
  gpt2RunPrompt: (prompt: string) =>
    post<Record<string, unknown>>('/api/gpt2/run_prompt', { prompt }),
  gpt2GetActivations: (layer?: number) =>
    post<Record<string, unknown>>('/api/gpt2/activations', { layer }),
  gpt2AttentionHead: (layer: number, head: number) =>
    post<Record<string, unknown>>('/api/gpt2/attention_head', { layer, head }),
  gpt2PatchHead: (layer: number, head: number, posToken: string, negToken: string) =>
    post<Record<string, unknown>>('/api/gpt2/patch_head', { layer, head, pos_token: posToken, neg_token: negToken }),
  gpt2RunIoi: (ioName: string, subjName: string) =>
    post<Record<string, unknown>>('/api/gpt2/ioi', { io_name: ioName, subj_name: subjName }),
  gpt2Architecture: () =>
    post<Record<string, unknown>>('/api/gpt2/architecture', {}),
  gpt2Layer: (layer: number) =>
    post<Record<string, unknown>>('/api/gpt2/layer', { layer }),
  gpt2Neuron: (layer: number, neuronIndex: number, component?: string, topKWeights?: number) =>
    post<Record<string, unknown>>('/api/gpt2/neuron', { layer, neuron_index: neuronIndex, component, top_k_weights: topKWeights }),
  gpt2Neurons: (layer: number, component?: string, page?: number, pageSize?: number, sortBy?: string, order?: string) =>
    post<Record<string, unknown>>('/api/gpt2/neurons', { layer, component, page, page_size: pageSize, sort_by: sortBy, order }),

  listExperiments: async (): Promise<{id: string; name: string; description: string; execution_status: string; used_mock_data: boolean; [key: string]: any}[]> => {
    const res = await get<{ experiments: any[] }>('/api/experiments');
    if (!res || !res.experiments) return [];
    return res.experiments.map((exp: any) => ({
      id: exp.id || exp.item_id,
      name: exp.name || '',
      description: exp.description || '',
      execution_status: exp.execution_status || 'PENDING',
      used_mock_data: exp.used_mock_data || false,
    }));
  },
  createExperiment: (data: Record<string, unknown>) => post<{id: string; execution_status: string; used_mock_data: boolean}>('/api/experiments', data),
  saveExperiment: (data: unknown) => post<{id: string; execution_status: string; used_mock_data: boolean}>('/api/experiments', data as Record<string, unknown>),
  deleteExperiment: (id: string) => del<{ status: string }>(`/api/experiments/${id}`),

  listSessions: async (): Promise<unknown[]> => {
    const res = await get<{ sessions?: unknown[] }>('/api/sessions');
    return Array.isArray(res) ? res : Array.isArray(res?.sessions) ? res.sessions : [];
  },
  createSession: (data: Record<string, unknown>) => post<unknown>('/api/sessions', data),
  saveSession: (data: unknown) => post<unknown>('/api/sessions', data as Record<string, unknown>),
  deleteSession: (id: string) => del<{ status: string }>(`/api/sessions/${id}`),

  listBenchmarks: async (): Promise<string[]> => {
    const res = await get<{ benchmarks?: string[] }>('/api/benchmarks');
    return Array.isArray(res) ? res : Array.isArray(res?.benchmarks) ? res.benchmarks : [];
  },
  runBenchmark: (name: string) => post<unknown>('/api/benchmarks/run', { benchmark_name: name }),

  getAppLogs: () => get<{ logs: any[] }>('/api/logs'),
  getBuildLogs: () => get<{ logs: any[] }>('/api/build/logs'),
  onBuildEvent: () => () => {},
  startBuild: () => post<{ status: string }>('/api/build/start'),
  clearBuildLogs: () => post<{ status: string }>('/api/build/clear'),

  listRecentFiles: () => get<{ files: any[] }>('/api/recent'),
  addRecentFile: (data: Record<string, unknown>) => post<unknown>('/api/recent', data),
  clearRecentFiles: () => post<{ status: string }>('/api/recent/clear'),
  showInFolder: () => {},

  listProjects: () => get<{ projects: any[] }>('/api/projects'),
  addProject: (data: Record<string, unknown>) => post<unknown>('/api/projects', data),
  removeProject: (id: string) => del<{ status: string }>(`/api/projects/${id}`),

  analyzeTokens: (prompt: string) =>
    post<Record<string, unknown>>('/api/runtime/analyze_tokens', { prompt }),

  // AI Research Assistant Layer
  getAssistantTools: () => get<{ tools: any[]; count: number }>('/api/assistant/tools'),
  investigateWithAssistant: (payload: {
    goal: string;
    model_name?: string;
    clean_prompt?: string;
    corrupted_prompt?: string;
    target_token?: string;
  }) => post<Record<string, unknown>>('/api/assistant/investigate', payload),
  executeAssistantTool: (toolName: string, params: Record<string, unknown> = {}) =>
    post<Record<string, unknown>>('/api/assistant/execute-tool', { tool_name: toolName, params }),
  getAssistantHistory: () => get<{ history: any[] }>('/api/assistant/history'),
  interveneWithAssistant: (payload: {
    prompt: string;
    target_token?: string;
    node_interventions?: Record<string, number>;
  }) => post<Record<string, unknown>>('/api/assistant/intervene', payload),
  runCrossModelSweep: (payload: {
    circuit_nodes?: any[];
    circuit_edges?: any[];
    reference_model?: string;
    target_models?: string[];
    task_name?: string;
  }) => post<Record<string, unknown>>('/api/assistant/cross-model-sweep', payload),
  // ── Research OS Endpoints ──────────────────────────────────────────
  listInvestigations: () => get<{ investigations: any[]; count: number }>('/api/v1/research/investigations'),
  getInvestigation: (id: string) => get<{ investigation: any; hypotheses: any[]; runs: any[]; evidence: any[]; mechanisms: any[] }>(`/api/v1/research/investigations/${id}`),
  createInvestigation: (data: Record<string, unknown>) => post<{ status: string; investigation: any }>('/api/v1/research/investigations', data),
  deleteInvestigation: (id: string) => del<{ status: string; id: string }>(`/api/v1/research/investigations/${id}`),

  listHypotheses: (investigationId?: string) =>
    get<{ hypotheses: any[]; count: number }>(`/api/v1/research/hypotheses${investigationId ? `?investigation_id=${investigationId}` : ''}`),
  createHypothesis: (data: Record<string, unknown>) => post<{ status: string; hypothesis: any }>('/api/v1/research/hypotheses', data),
  deleteHypothesis: (id: string) => del<{ status: string; id: string }>(`/api/v1/research/hypotheses/${id}`),
  evaluateHypothesis: (id: string, investigationId: string) =>
    post<Record<string, unknown>>(`/api/v1/research/hypotheses/${id}/evaluate?investigation_id=${investigationId}`),

  runCausalExperiment: (payload: Record<string, unknown>) =>
    post<{ status: string; run: { id: string; execution_status: string; used_mock_data: boolean; [key: string]: any } }>('/api/v1/research/experiments/run', payload),
  listRuns: (experimentId?: string, investigationId?: string) => {
    const params = new URLSearchParams();
    if (experimentId) params.append('experiment_id', experimentId);
    if (investigationId) params.append('investigation_id', investigationId);
    const qs = params.toString();
    return get<{ runs: Array<{ id: string; execution_status: string; used_mock_data: boolean; [key: string]: any }>; count: number }>(`/api/v1/research/runs${qs ? `?${qs}` : ''}`);
  },

  listEvidence: (investigationId?: string, hypothesisId?: string) => {
    const params = new URLSearchParams();
    if (investigationId) params.append('investigation_id', investigationId);
    if (hypothesisId) params.append('hypothesis_id', hypothesisId);
    const qs = params.toString();
    return get<{ evidence: any[]; count: number }>(`/api/v1/research/evidence${qs ? `?${qs}` : ''}`);
  },
  createEvidence: (data: Record<string, unknown>) => post<{ status: string; evidence: any }>('/api/v1/research/evidence', data),
  getEvidenceMatrix: (investigationId: string) =>
    get<{ matrix: any[]; count: number }>(`/api/v1/research/evidence/matrix?investigation_id=${investigationId}`),

  listMechanisms: (investigationId?: string) =>
    get<{ mechanisms: any[]; count: number }>(`/api/v1/research/mechanisms${investigationId ? `?investigation_id=${investigationId}` : ''}`),
  createMechanism: (data: Record<string, unknown>) => post<{ status: string; mechanism: any }>('/api/v1/research/mechanisms', data),
  deleteMechanism: (id: string) => del<{ status: string; id: string }>(`/api/v1/research/mechanisms/${id}`),

  listArtifacts: (investigationId?: string) =>
    get<{ artifacts: any[]; count: number }>(`/api/v1/research/artifacts${investigationId ? `?investigation_id=${investigationId}` : ''}`),
  generateResearchReport: (investigationId: string) =>
    post<{ status: string; report_markdown: string; investigation_id: string }>('/api/v1/research/artifacts/generate-report', { investigation_id: investigationId }),

  listJobs: (status?: string) =>
    get<{ jobs: any[]; count: number }>(`/api/v1/research/jobs${status ? `?status=${status}` : ''}`),
  createJob: (data: Record<string, unknown>) => post<{ status: string; job: any }>('/api/v1/research/jobs', data),
};




