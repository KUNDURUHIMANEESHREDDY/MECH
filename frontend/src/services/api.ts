const BASE = 'http://localhost:8000';

let cachedApiKey: string | null = null;

async function getApiKey(): Promise<string> {
  if (cachedApiKey) return cachedApiKey;
  try {
    if (typeof window !== 'undefined' && window.desktopApi?.getApiKey) {
      cachedApiKey = await window.desktopApi.getApiKey();
      return cachedApiKey ?? '';
    }
  } catch { }
  return '';
}

async function get<T>(path: string): Promise<T> {
  const apiKey = await getApiKey();
  const res = await fetch(`${BASE}${path}`, {
    headers: apiKey ? { 'X-API-Key': apiKey } : undefined,
  });
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const apiKey = await getApiKey();
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(apiKey ? { 'X-API-Key': apiKey } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json();
}

async function del<T>(path: string): Promise<T> {
  const apiKey = await getApiKey();
  const res = await fetch(`${BASE}${path}`, {
    method: 'DELETE',
    headers: apiKey ? { 'X-API-Key': apiKey } : undefined,
  });
  if (!res.ok) throw new Error(`DELETE ${path} failed: ${res.status}`);
  return res.json();
}

export const api = {
  pythonPing: () => get<{ status: string }>('/api/status'),
  pythonCall: (method: string, params: Record<string, unknown>) =>
    post<Record<string, unknown>>(`/api/${method}`, params),

  listModels: () => get<{ models: string[] }>('/api/models'),
  loadModel: (name: string) => post<{ status: string }>('/api/models/load', { model_name: name }),
  getModelInfo: (name: string) => get<Record<string, unknown>>(`/api/models/${name}`),
  infer: (prompt: string, model?: string) =>
    post<Record<string, unknown>>('/api/infer', { prompt, model_name: model }),

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

  listExperiments: async (): Promise<unknown[]> => {
    const res = await get<{ experiments?: unknown[] }>('/api/experiments');
    return Array.isArray(res) ? res : Array.isArray(res?.experiments) ? res.experiments : [];
  },
  createExperiment: (data: Record<string, unknown>) => post<unknown>('/api/experiments', data),
  saveExperiment: (data: unknown) => post<unknown>('/api/experiments', data as Record<string, unknown>),
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
};
