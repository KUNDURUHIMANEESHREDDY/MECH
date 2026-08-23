/// <reference types="vite/client" />

/** Extended import.meta.env for MECH Platform. */
interface ImportMetaEnv {
  readonly VITE_BACKEND_URL?: string;
  readonly VITE_API_KEY?: string;
  readonly VITE_DEV_BACKEND_URL?: string;
  readonly [key: string]: string | undefined;
}

/* ---- Typed bridge surfaces matching the REAL Electron preload (electron/preload.js) ---- */

/** Data + settings + build/log surface — exposed as window.appApi. */
interface AppSettings {
  theme: string;
  gpu: { enabled: boolean; acceleration: string };
  cache: { enabled: boolean; maxSizeMb: number; location: string };
  paths: { python: string; workspace: string; projects: string };
  models: { defaultModel: string; device: string; maxContextTokens: number; precision: string };
  python: { path: string; environment: string; venvPath: string };
  performance: { threads: number; memoryLimitGb: number; enableTorchCompile: boolean };
  plugins: { enabledPlugins: string[]; autoUpdate: boolean };
  debugger: { breakpointBehavior: string; tokenHighlighting: string; traceLevel: string };
  logging: { level: string; fileRetentionDays: number; consoleOutput: boolean };
}

interface AppProject {
  id: string;
  name: string;
  path: string;
  createdAt: string;
  updatedAt: string;
}

interface AppRecentFile {
  id: string;
  path: string;
  label: string;
  openedAt: string;
}

interface AppBuildLogEntry {
  ts: string;
  level: 'info' | 'warn' | 'error';
  message: string;
  meta: Record<string, unknown> | null;
}

interface AppBuildEvent {
  type: 'stdout' | 'stderr' | 'error' | 'close';
  ts: string;
  data: string | { code: number };
}

interface AppLogEntry {
  ts: string;
  level: string;
  event: string;
  meta: unknown;
}

type DeepPartial<T> = { [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P] };

interface AppApiBridge {
  getSettings(): Promise<AppSettings>;
  setSettings(patch: DeepPartial<AppSettings>): Promise<AppSettings>;
  resetSettings(): Promise<AppSettings>;

  listProjects(): Promise<AppProject[]>;
  addProject(project: { id: string; name?: string; path?: string }): Promise<AppProject>;
  removeProject(id: string): Promise<{ ok: boolean }>;

  listRecentFiles(): Promise<AppRecentFile[]>;
  addRecentFile(file: { path: string; label?: string }): Promise<AppRecentFile>;
  clearRecentFiles(): Promise<{ ok: boolean }>;

  listSessions(): Promise<unknown[]>;
  saveSession(session: unknown): Promise<unknown>;
  removeSession(id: string): Promise<{ ok: boolean }>;
  listExperiments(): Promise<unknown[]>;
  saveExperiment(exp: unknown): Promise<unknown>;
  removeExperiment(id: string): Promise<{ ok: boolean }>;

  startBuild(options?: { target?: 'renderer' | 'python' }): Promise<{ ok: boolean; pid?: number; target?: string; error?: string }>;
  getBuildLogs(): Promise<AppBuildLogEntry[]>;
  clearBuildLogs(): Promise<{ ok: boolean }>;
  getAppLogs(): Promise<AppLogEntry[]>;
  onBuildEvent(callback: (event: AppBuildEvent) => void): () => void;

  pythonPing(): Promise<{ ok: boolean; storage: string }>;
  pythonCall(method: string, payload?: unknown): Promise<unknown>;
  getRuntimeStatus(): Promise<unknown>;
  analyzeTokens(prompt: string): Promise<unknown>;

  openExternal(url: string): Promise<{ ok: boolean }>;
  showInFolder(p: string): Promise<{ ok: boolean }>;
}

/** Minimal legacy transport bridge — exposed as window.desktopApi. */
interface DesktopApiBridge {
  getApiKey(): Promise<string | null>;
  httpRequest(request: unknown): Promise<unknown>;
  ping(): Promise<{ ok: boolean; storage: string }>;
}

interface Window {
  appApi: AppApiBridge;
  desktopApi: DesktopApiBridge;
}