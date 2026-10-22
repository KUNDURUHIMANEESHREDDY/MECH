import type {
  InferenceResponse,
  UnavailableResponse,
} from '../types';

type AnyFunction = (...args: any[]) => any;
type LocalBridge = Record<string, AnyFunction | undefined>;

type ApiProxyResponse = { status: number; body: unknown };

export type ApiProxyRequest = {
  method?: string;
  path: string;
  body?: unknown;
  query?: Record<string, string | number | boolean>;
  headers?: Record<string, string>;
};

type ApiProxy = (request: ApiProxyRequest) => Promise<ApiProxyResponse>;

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

/**
 * The Electron bridge, when this renderer is the desktop app.
 *
 * Its presence is what makes this a desktop renderer: the preload exposes
 * `apiRequest` and nothing that returns the bearer token.
 */
function apiProxy(): ApiProxy | null {
  const method = runtime.appApi?.apiRequest;
  return typeof method === 'function' ? (method as ApiProxy) : null;
}

export function isDesktop(): boolean {
  return apiProxy() !== null;
}

/**
 * No credential is attached by the renderer.
 *
 * Desktop requests are authenticated by the Electron main process through the
 * `api:request` proxy. The browser build is authenticated by the HttpOnly
 * session cookie issued by `POST /session`, which page JavaScript cannot read.
 * There is therefore no token to resolve and no header to build here; the
 * helper stays because call sites spread it alongside their own headers.
 */
export function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return { ...extra };
}

export async function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const proxy = apiProxy();
  if (proxy) {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
    let url: URL | null = null;
    try {
      url = new URL(raw, API_ORIGIN || 'http://localhost:8000');
    } catch {
      url = null;
    }
    if (url && /^\/(?:api|health)(?:\/|$)/.test(url.pathname)) {
      const query: Record<string, string> = {};
      url.searchParams.forEach((value, key) => { query[key] = value; });
      const { status, body } = await proxy({
        method: init.method ?? 'GET',
        path: url.pathname,
        query: Object.keys(query).length ? query : undefined,
        body: parseJsonBody(init.body),
        headers: init.headers as Record<string, string> | undefined,
      });
      return jsonResponse(status, body);
    }
  }
  const headers = {
    ...authHeaders(),
    ...(init.headers as Record<string, string> | undefined),
  };
  return fetch(input, { credentials: 'same-origin', ...init, headers });
}

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

function parseJsonBody(body: BodyInit | null | undefined): unknown {
  if (typeof body !== 'string' || !body) return undefined;
  try {
    return JSON.parse(body);
  } catch {
    return undefined;
  }
}

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body ?? null), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function splitPath(path: string): { pathname: string; query?: Record<string, string> } {
  const [pathname, search = ''] = path.split('?');
  if (!search) return { pathname };
  const query: Record<string, string> = {};
  new URLSearchParams(search).forEach((value, key) => { query[key] = value; });
  return { pathname, query };
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const label = `${init.method ?? 'GET'} ${path}`;

  // Desktop: authenticated by the main process, never by this renderer.
  const proxy = apiProxy();
  if (proxy) {
    const { pathname, query } = splitPath(path);
    const { status, body } = await proxy({
      method: init.method ?? 'GET',
      path: pathname,
      query,
      body: parseJsonBody(init.body),
      headers: init.headers as Record<string, string> | undefined,
    });
    if (status === 401) throw new Error(`${label} failed: 401 (missing or invalid bearer token)`);
    if (status < 200 || status >= 300) throw new Error(`${label} failed: ${status}`);
    return body as T;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const headers = { ...authHeaders(), ...(init.headers as Record<string, string> | undefined) };
    // `same-origin` is what carries the session cookie in the browser build.
    const response = await fetch(`${BASE}${path}`, {
      credentials: 'same-origin',
      ...init,
      headers,
      signal: controller.signal,
    });
    if (response.status === 401) throw new Error(`${init.method ?? 'GET'} ${path} failed: 401 (missing or invalid bearer token)`);
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

async function put<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
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
  loadModel: (name: string) =>
    post<{ status: string; model_name?: string } & Partial<UnavailableResponse>>(
      '/api/models/load',
      { model_name: name },
    ),
  getModelInfo: (name: string) => get<Record<string, unknown>>(`/api/models/${name}`),
  infer: (prompt: string, model?: string) =>
    post<Partial<InferenceResponse> & Partial<UnavailableResponse>>(
      '/api/infer',
      { prompt, model_name: model },
    ),

  gpt2Load: (model?: string) =>
    post<{ status: string; model_name?: string } & Partial<UnavailableResponse>>(
      '/api/gpt2/load',
      { model_name: model ?? 'gpt2' },
    ),
  gpt2RunPrompt: (prompt: string) =>
    post<Record<string, unknown>>('/api/gpt2/run_prompt', { prompt }),
  gpt2FreshPrompt: () =>
    post<Record<string, unknown>>('/api/gpt2/fresh_prompt', {}),
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
  gpt2Steer: (payload: Record<string, unknown>) =>
    post<Record<string, unknown>>('/api/gpt2/steer', payload),
  gpt2Architecture: () =>
    post<Record<string, unknown>>('/api/gpt2/architecture', {}),
  gpt2Layer: (layer: number) =>
    post<Record<string, unknown>>('/api/gpt2/layer', { layer }),
  gpt2Neuron: (layer: number, neuronIndex: number, component?: string, topKWeights?: number) =>
    post<Record<string, unknown>>('/api/gpt2/neuron', { layer, neuron_index: neuronIndex, component, top_k_weights: topKWeights }),
  gpt2Neurons: (layer: number, component?: string, page?: number, pageSize?: number, sortBy?: string, order?: string) =>
    post<Record<string, unknown>>('/api/gpt2/neurons', { layer, component, page, page_size: pageSize, sort_by: sortBy, order }),

  saeStatus: () =>
    get<Record<string, unknown>>('/api/sae/status'),
  saeInspect: (payload: Record<string, unknown>) =>
    post<Record<string, unknown>>('/api/sae/inspect', payload),
  saeTrain: (payload: Record<string, unknown>) =>
    post<Record<string, unknown>>('/api/sae/train', payload),
  saeRun: (runId: string) =>
    get<Record<string, unknown>>(`/api/sae/runs/${runId}`),

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
  },  createSession: (data: Record<string, unknown>) => post<unknown>('/api/sessions', data),
  saveSession: (data: unknown) => post<unknown>('/api/sessions', data as Record<string, unknown>),
  deleteSession: (id: string) => del<{ status: string }>(`/api/sessions/${id}`),

  listBenchmarks: async (): Promise<string[]> => {
    const res = await get<{ benchmarks?: string[] }>('/api/benchmarks');
    return Array.isArray(res) ? res : Array.isArray(res?.benchmarks) ? res.benchmarks : [];
  },
  runBenchmark: (name: string) => post<unknown>('/api/benchmarks/run', { benchmark_name: name }),
  // Persisted run history. `benchmark` and `limit` are omitted entirely when
  // not supplied so the route's own defaults apply rather than a guessed one.
  benchmarkHistory: (name?: string, limit?: number) => {
    const query = new URLSearchParams();
    if (name) query.set('benchmark', name);
    if (typeof limit === 'number') query.set('limit', String(limit));
    const suffix = query.toString();
    return get<Record<string, unknown>>(
      `/api/benchmarks/history${suffix ? `?${suffix}` : ''}`,
    );
  },
  compareBenchmarkRuns: (runA: string, runB: string) =>
    get<Record<string, unknown>>(
      `/api/benchmarks/compare?${new URLSearchParams({ run_a: runA, run_b: runB }).toString()}`,
    ),

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

  getBackendSettings: () => get<Record<string, unknown>>('/api/settings'),
  updateBackendSettings: (settings: Record<string, unknown>) =>
    put<Record<string, unknown>>('/api/settings', settings),
  listBackendProjects: () => get<Record<string, unknown>>('/api/projects'),
  addBackendProject: (project: Record<string, unknown>) =>
    post<Record<string, unknown>>('/api/projects', project),
  listBackendRecentFiles: () => get<Record<string, unknown>>('/api/recent-files'),
  addBackendRecentFile: (file: Record<string, unknown>) =>
    post<Record<string, unknown>>('/api/recent-files', file),
  getBackendLogs: (limit?: number) =>
    get<Record<string, unknown>>(`/api/logs${limit ? `?limit=${limit}` : ''}`),
  getBackendBuild: () => get<Record<string, unknown>>('/api/build'),
  startBackendBuild: () => post<Record<string, unknown>>('/api/build', {}),
  getBackendPlugins: () => get<Record<string, unknown>>('/api/plugins'),
  installBackendPlugin: (path: string) =>
    post<Record<string, unknown>>('/api/plugins/install', { path }),
  enableBackendPlugin: (name: string) =>
    post<Record<string, unknown>>(`/api/plugins/${name}/enable`, {}),
  disableBackendPlugin: (name: string) =>
    post<Record<string, unknown>>(`/api/plugins/${name}/disable`, {}),
  uninstallBackendPlugin: (name: string) =>
    del<Record<string, unknown>>(`/api/plugins/${name}`),
};
