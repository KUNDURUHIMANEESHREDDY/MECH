type AnyFunction = (...args: any[]) => any;
type LocalBridge = Record<string, AnyFunction | undefined>;

type RuntimeGlobals = typeof globalThis & {
  __MECH_API_BASE__?: string;
  appApi?: LocalBridge;
};

const runtime = globalThis as RuntimeGlobals;
const browserUsesOriginProxy = typeof window !== 'undefined' && window.location.protocol !== 'file:';
const configuredOrigin = runtime.__MECH_API_BASE__?.replace(/\/+$/, '');

export const API_ORIGIN = configuredOrigin || (browserUsesOriginProxy ? '' : 'http://localhost:8000');
export const API_BASE = `${API_ORIGIN}/api`;

export function apiUrl(path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${API_ORIGIN}${normalized}`;
}

const BASE = API_ORIGIN;
const REQUEST_TIMEOUT_MS = 30_000;

function localBridge(): LocalBridge | null {
  return runtime.appApi ?? null;
}

function localUnavailable(name: string): Error {
  return new Error(`Local application bridge unavailable for ${name}.`);
}

async function localCall<T>(name: string, ...args: any[]): Promise<T> {
  const method = localBridge()?.[name];
  if (typeof method !== 'function') throw localUnavailable(name);
  return (await method(...args)) as T;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${BASE}${path}`, { ...init, signal: controller.signal });
    if (!response.ok) throw new Error(`${init.method ?? 'GET'} ${path} failed: ${response.status}`);
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(`${init.method ?? 'GET'} ${path} timed out after ${REQUEST_TIMEOUT_MS} ms`);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function get<T>(path: string): Promise<T> {
  return request<T>(path, { method: 'GET' });
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

async function del<T>(path: string): Promise<T> {
  return request<T>(path, { method: 'DELETE' });
}

export const api = {
  baseUrl: BASE,

  pythonPing: () => get<{ status: string }>('/api/status'),
  pythonCall: (method: string, params: Record<string, unknown>) =>
    post<Record<string, unknown>>(`/api/${method}`, params),

  listModels: () => get<{ models: string[] }>('/api/models'),
  loadModel: (name: string) => post<{ status: string; model_name?: string }>('/api/models/load', { model_name: name }),
  getModelInfo: (name: string) => get<Record<string, unknown>>(`/api/models/${name}`),
  infer: (prompt: string, model?: string) =>
    post<Record<string, unknown>>('/api/infer', { prompt, model_name: model }),

  gpt2Load: (model?: string) =>
    post<{ status: string; model_name: string }>('/api/gpt2/load', { model_name: model ?? 'gpt2' }),
  gpt2RunPrompt: (prompt: string) =>
    post<Record<string, unknown>>('/api/gpt2/run_prompt', { prompt }),
  gpt2GetActivations: (layer?: number) =>
    post<Record<string, unknown>>('/api/gpt2/activations', { layer }),
  gpt2LayerActivations: (payload: Record<string, unknown>) =>
    post<Record<string, unknown>>('/api/gpt2/layer_activations', payload),
  gpt2LogitLensAll: (payload: Record<string, unknown>) =>
    post<Record<string, unknown>>('/api/gpt2/logit_lens_all', payload),
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

  getAppLogs: () => localCall<unknown[]>('getAppLogs'),
  getBuildLogs: () => localCall<unknown[]>('getBuildLogs'),
  onBuildEvent: (listener: (...args: any[]) => void) => {
    const method = localBridge()?.onBuildEvent;
    if (typeof method !== 'function') return () => undefined;
    return method(listener);
  },
  startBuild: (...args: any[]) => localCall<unknown>('startBuild', ...args),
  clearBuildLogs: () => localCall<unknown>('clearBuildLogs'),

  listRecentFiles: () => localCall<unknown[]>('listRecentFiles'),
  addRecentFile: (...args: any[]) => localCall<unknown>('addRecentFile', ...args),
  clearRecentFiles: () => localCall<unknown>('clearRecentFiles'),
  showInFolder: (...args: any[]) => localCall<unknown>('showInFolder', ...args),

  listProjects: () => localCall<unknown[]>('listProjects'),
  addProject: (...args: any[]) => localCall<unknown>('addProject', ...args),
  removeProject: (...args: any[]) => localCall<unknown>('removeProject', ...args),
};
