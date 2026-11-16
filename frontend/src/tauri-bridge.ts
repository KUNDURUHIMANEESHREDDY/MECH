import { invoke } from '@tauri-apps/api/core';

type ApiRequest = {
  method?: string;
  path: string;
  body?: unknown;
  query?: Record<string, string | number | boolean>;
  headers?: Record<string, string>;
};

type TauriWindow = Window & {
  __TAURI_INTERNALS__?: unknown;
  __MECH_API_BASE__?: string;
  appApi?: Record<string, (...args: any[]) => any>;
};

const host = window as TauriWindow;

// The renderer has no Node access and never receives the bearer token. Every
// control-plane request crosses the typed Rust command boundary instead.
if (host.__TAURI_INTERNALS__) {
  host.__MECH_API_BASE__ = 'http://127.0.0.1:8000';
  const storeList = (table: string) => invoke('storage_list', { table });
  const storeUpsert = (table: string, record: unknown) => invoke('storage_upsert', { table, record });
  const storeRemove = (table: string, id: string) => invoke('storage_remove', { table, id });
  const backendCall = (method: string, payload: unknown = {}) => invoke('backend_call', { method, payload });

  host.appApi = {
    apiRequest: (request: ApiRequest) => invoke('api_request', { request }),
    getBackendStatus: () => invoke('backend_status'),

    getSettings: () => invoke('settings_get'),
    setSettings: (patch: unknown) => invoke('settings_set', { patch }),
    resetSettings: () => invoke('settings_reset'),

    listProjects: () => storeList('projects'),
    addProject: (project: Record<string, unknown>) => {
      const now = new Date().toISOString();
      return storeUpsert('projects', { ...project, createdAt: project.createdAt ?? now, updatedAt: now });
    },
    removeProject: (id: string) => storeRemove('projects', id),
    listRecentFiles: () => storeList('recent_files'),
    addRecentFile: (file: { path: string; label?: string }) => storeUpsert('recent_files', {
      id: `rec_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      path: file.path,
      label: file.label ?? file.path.split(/[\\/]/).pop() ?? file.path,
      openedAt: new Date().toISOString(),
    }),
    clearRecentFiles: () => invoke('storage_clear', { table: 'recent_files' }),

    listSessions: () => storeList('sessions'),
    saveSession: (session: Record<string, unknown>) => storeUpsert('sessions', session),
    removeSession: (id: string) => storeRemove('sessions', id),
    listExperiments: () => storeList('experiments'),
    saveExperiment: (experiment: Record<string, unknown>) => storeUpsert('experiments', experiment),
    removeExperiment: (id: string) => storeRemove('experiments', id),

    pythonPing: async () => {
      try { return { ok: true, data: await backendCall('ping', {}) }; }
      catch (error) { return { ok: false, error: String(error) }; }
    },
    pythonCall: (method: string, payload: unknown) => backendCall(method, payload),
    getRuntimeStatus: () => backendCall('runtime/status', {}),
    analyzeTokens: (prompt: string) => backendCall('runtime/analyze_tokens', { prompt }),
    gpt2Load: () => backendCall('gpt2/load', {}),
    gpt2RunPrompt: (prompt: string) => backendCall('gpt2/run_prompt', { prompt }),
    gpt2GetActivations: (layer: number) => backendCall('gpt2/activations', { layer }),
    gpt2AttentionHead: (layer: number, head: number) => backendCall('gpt2/attention_head', { layer, head }),
    gpt2PatchHead: (layer: number, head: number, posToken: string, negToken: string) =>
      backendCall('gpt2/patch_head', { layer, head, pos_token: posToken, neg_token: negToken }),
    gpt2RunIoi: (ioName: string, subjName: string) => backendCall('gpt2/ioi', { io_name: ioName, subj_name: subjName }),
    getAppLogs: () => invoke('storage_logs'),
  };
}
