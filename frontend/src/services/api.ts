const BASE = 'http://localhost:8000';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json();
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`DELETE ${path} failed: ${res.status}`);
  return res.json();
}

const noopResolve = (val?: any) => Promise.resolve(val ?? []);

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

  getAppLogs: () => noopResolve(),
  getBuildLogs: () => noopResolve(),
  onBuildEvent: () => () => {},
  startBuild: () => noopResolve(),
  clearBuildLogs: () => noopResolve(),

  listRecentFiles: () => noopResolve(),
  addRecentFile: () => noopResolve(),
  clearRecentFiles: () => noopResolve(),
  showInFolder: () => {},

  listProjects: () => noopResolve(),
  addProject: () => noopResolve(),
  removeProject: () => noopResolve(),
};
