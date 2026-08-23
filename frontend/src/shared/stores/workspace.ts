import { create } from 'zustand';
import { Resource, WorkspaceState, WindowState } from '../types';
import { commandManager } from '../managers/commandManager';
import { sessionManager } from '../managers/sessionManager';

interface WorkspaceStoreActions {
  setActiveWorkspace: (id: string) => void;
  setActiveModel: (model: string | null) => void;
  openResource: (resource: Resource, panelId?: string) => void;
  openPanel: (panelId: string) => void;
  closePanel: (panelId: string) => void;
  togglePanel: (panelId: string) => void;
  setVisiblePanels: (panels: Record<string, boolean>) => void;
  updateCamera: (camera: { x: number; y: number; zoom: number }) => void;
  addNote: (title: string, content: string) => void;
  addConsoleLog: (level: 'info' | 'warn' | 'error', message: string) => void;
  addTimelineEvent: (event: string, payload?: unknown) => void;
  detachPanelToWindow: (panelId: string, position?: { x: number; y: number }) => void;
  updateWindow: (windowId: string, patch: Partial<WindowState>) => void;
  closeWindow: (windowId: string) => void;
  saveSession: () => Promise<void>;
  restoreSession: (workspaceId?: string) => Promise<void>;
}

export type WorkspaceStore = WorkspaceState & WorkspaceStoreActions;

const DEFAULT_PANELS: Record<string, boolean> = {
  active_investigation: true,
  attention_heatmap: true,
  neuron_panel: true,
  logit_lens: true,
  circuit_explorer: true,
  dataset_viewer: true,
  sae_feature: true,
};

export const useWorkspaceStore = create<WorkspaceStore>((set, get) => ({
  workspaceId: 'default',
  activeWorkspace: 'GPT-2 Lab',
  workspaceNames: ['GPT-2 Lab', 'Gemma Research', 'Llama Circuit Discovery'],
  activeModel: 'gpt2',
  activeSessionId: 'sess_default_01',
  selection: {
    resource: null,
    layer: null,
    head: null,
    neuron: null,
    token: null,
  },
  windows: [],
  camera: { x: 0, y: 0, zoom: 1 },
  notes: [
    { id: 'note_1', title: 'Induction Heads in Layer 5', content: 'Observed strong prefix matching behavior on head 5.1.' },
  ],
  timeline: [
    { id: 't_1', ts: new Date().toLocaleTimeString(), event: 'Workspace Loaded', payload: { model: 'gpt2' } },
  ],
  console: [
    { id: 'c_1', ts: new Date().toLocaleTimeString(), level: 'info', message: 'MECH Visual Research Canvas initialised.' },
  ],
  visiblePanels: { ...DEFAULT_PANELS },
  layout: null,

  setActiveWorkspace: (id: string) => {
    set({ activeWorkspace: id });
    commandManager.publish('workspace.switched', { workspaceId: id });
    get().restoreSession(id);
  },

  setActiveModel: (model: string | null) => {
    set({ activeModel: model });
    if (model) {
      commandManager.publish('model.loaded', { model });
      get().addConsoleLog('info', `Loaded model: ${model}`);
    }
  },

  openResource: (resource: Resource, panelId?: string) => {
    const targetPanel = panelId || `${resource.kind}_viewer`;
    get().openPanel(targetPanel);
    commandManager.publish('resource.opened', { resource, panelId: targetPanel });
    get().addTimelineEvent(`Opened resource: ${resource.label} (${resource.kind})`);
  },

  openPanel: (panelId: string) => {
    set((state) => ({
      visiblePanels: { ...state.visiblePanels, [panelId]: true },
    }));
    commandManager.publish('panel.opened', { panelId });
  },

  closePanel: (panelId: string) => {
    set((state) => ({
      visiblePanels: { ...state.visiblePanels, [panelId]: false },
    }));
    commandManager.publish('panel.closed', { panelId });
  },

  togglePanel: (panelId: string) => {
    const current = get().visiblePanels[panelId];
    if (current) {
      get().closePanel(panelId);
    } else {
      get().openPanel(panelId);
    }
  },

  setVisiblePanels: (panels: Record<string, boolean>) => {
    set({ visiblePanels: panels });
  },

  updateCamera: (camera) => {
    set({ camera });
  },

  addNote: (title, content) => {
    const newNote = { id: `note_${Date.now()}`, title, content };
    set((state) => ({ notes: [...state.notes, newNote] }));
    commandManager.publish('note.created', { note: newNote });
  },

  addConsoleLog: (level, message) => {
    const newLog = { id: `log_${Date.now()}`, ts: new Date().toLocaleTimeString(), level, message };
    set((state) => ({ console: [newLog, ...state.console].slice(0, 200) }));
  },

  addTimelineEvent: (event, payload) => {
    const newEvt = { id: `evt_${Date.now()}`, ts: new Date().toLocaleTimeString(), event, payload };
    set((state) => ({ timeline: [...state.timeline, newEvt].slice(0, 200) }));
  },

  detachPanelToWindow: (panelId, position) => {
    const existing = get().windows.find((w) => w.panelId === panelId);
    if (existing) return;

    const newWindow: WindowState = {
      id: `win_${panelId}_${Date.now()}`,
      panelId,
      x: position?.x ?? 120 + get().windows.length * 30,
      y: position?.y ?? 100 + get().windows.length * 30,
      width: 500,
      height: 380,
      zIndex: get().windows.length + 10,
      detached: true,
      maximized: false,
    };

    set((state) => ({
      windows: [...state.windows, newWindow],
      visiblePanels: { ...state.visiblePanels, [panelId]: true },
    }));
  },

  updateWindow: (windowId, patch) => {
    set((state) => ({
      windows: state.windows.map((w) => (w.id === windowId ? { ...w, ...patch } : w)),
    }));
  },

  closeWindow: (windowId) => {
    set((state) => ({
      windows: state.windows.filter((w) => w.id !== windowId),
    }));
  },

  saveSession: async () => {
    const state = get();
    await sessionManager.saveSession(state);
  },

  restoreSession: async (workspaceId) => {
    const id = workspaceId || get().activeWorkspace;
    const session = await sessionManager.loadSession(id);
    if (session) {
      set({
        activeModel: session.activeModel || get().activeModel,
        windows: session.windows || [],
        camera: session.camera || { x: 0, y: 0, zoom: 1 },
        notes: session.notes.length ? session.notes : get().notes,
        timeline: session.timeline.length ? session.timeline : get().timeline,
        console: session.console.length ? session.console : get().console,
        visiblePanels: session.visiblePanels || get().visiblePanels,
      });
    }
  },
}));
