import { create } from 'zustand';
import { commandManager } from '../shared/managers/commandManager';
import { sessionManager } from '../shared/managers/sessionManager';

export interface AppState {
  // UI state
  activePage: string;
  sidebarCollapsed: boolean;
  activityCollapsed: boolean;
  commandPaletteOpen: boolean;

  // Model state
  activeModel: string | null;
  pythonStatus: boolean;
  gpuUtil: number | null;
  memoryUtil: number | null;

  // Console
  consoleLogs: Array<{ level: 'info' | 'warn' | 'error'; message: string; timestamp: number }>;

  // Workspace
  selection: {
    resource: any | null;
    layer: number | null;
    head: number | null;
    neuron: number | null;
    token: number | null;
  };

  // Pages registry
  pages: Record<string, { label: string; icon?: any; path?: string; component?: React.ComponentType }>;

  // Actions
  setActivePage: (page: string) => void;
  toggleSidebar: () => void;
  toggleActivity: () => void;
  setCommandPaletteOpen: (open: boolean) => void;
  setActiveModel: (model: string | null) => void;
  setPythonStatus: (status: boolean) => void;
  setGpuUtil: (util: number | null) => void;
  setMemoryUtil: (util: number | null) => void;
  addConsoleLog: (level: 'info' | 'warn' | 'error', message: string) => void;
  clearConsole: () => void;
  setSelectedLayer: (layer: number | null) => void;
  setSelectedHead: (head: number | null) => void;
  openPanel: (panelId: string) => void;
}

const PAGES = {
  workspace: { label: 'Workspace', path: '/workspace' },
  models: { label: 'Models', path: '/models' },
  explorer: { label: 'Model Explorer', path: '/explorer' },
  transformer: { label: 'Transformer Visualizer', path: '/transformer' },
  neuralexplorer: { label: 'Neural Explorer', path: '/neuralexplorer' },
  debugger: { label: 'Debugger', path: '/debugger' },
  experiments: { label: 'Experiments', path: '/experiments' },
  sessions: { label: 'Sessions', path: '/sessions' },
  reports: { label: 'Reports', path: '/reports' },
  settings: { label: 'Settings', path: '/settings' },
  logging: { label: 'Execution Logs', path: '/logging' },
  build: { label: 'Build Log', path: '/build' },
  benchmark: { label: 'Benchmark', path: '/benchmark' },
  benchmarksuite: { label: 'Benchmark Suite', path: '/benchmarksuite' },
  prompts: { label: 'Prompts', path: '/prompts' },
  notebooks: { label: 'Research Notebook', path: '/notebooks' },
  knowledgegraph: { label: 'Knowledge Graph', path: '/knowledgegraph' },
  reasoning: { label: 'Reasoning Trace', path: '/reasoning' },
  evidencefusion: { label: 'Evidence Fusion', path: '/evidencefusion' },
  analytics: { label: 'Analytics', path: '/analytics' },
  health: { label: 'Scientific Health', path: '/health' },
  campaigns: { label: 'Campaigns', path: '/campaigns' },
  plugins: { label: 'Plugin Registry', path: '/plugins' },
  reproduction: { label: 'Paper Reproduction', path: '/reproduction' },
  projects: { label: 'Research Projects', path: '/projects' },
  recent: { label: 'Recent Files', path: '/recent' },
};

export const useAppStore = create<AppState>((set, get) => ({
  activePage: 'workspace',
  sidebarCollapsed: false,
  activityCollapsed: false,
  commandPaletteOpen: false,
  activeModel: null,
  pythonStatus: false,
  gpuUtil: null,
  memoryUtil: null,
  consoleLogs: [],
  selection: {
    resource: null,
    layer: null,
    head: null,
    neuron: null,
    token: null,
  },
  pages: PAGES,

  setActivePage: (page) => {
    set({ activePage: page });
    commandManager.publish('page.changed', { page });
  },
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  toggleActivity: () => set((state) => ({ activityCollapsed: !state.activityCollapsed })),
  setCommandPaletteOpen: (open) => set({ commandPaletteOpen: open }),
  setActiveModel: (model) => set({ activeModel: model }),
  setPythonStatus: (status) => set({ pythonStatus: status }),
  setGpuUtil: (util) => set({ gpuUtil: util }),
  setMemoryUtil: (util) => set({ memoryUtil: util }),
  addConsoleLog: (level, message) => {
    const entry = { level, message, timestamp: Date.now() };
    set((state) => ({ consoleLogs: [...state.consoleLogs, entry] }));
  },
  clearConsole: () => set({ consoleLogs: [] }),
  setSelectedLayer: (layer) => set((state) => ({ selection: { ...state.selection, layer } })),
  setSelectedHead: (head) => set((state) => ({ selection: { ...state.selection, head } })),
  openPanel: (panelId) => {
    commandManager.publish('panel.opened', { panelId });
  },
}));
