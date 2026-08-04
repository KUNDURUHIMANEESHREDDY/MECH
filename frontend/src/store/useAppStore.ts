import { useState, useEffect } from 'react';

export interface SelectionState {
  selectedLayer: number;
  selectedHead: number;
  selectedNeuron: number | null;
  hoveredToken: string | null;
  selectedTokenIdx: number | null;
}

export interface WorkspaceState {
  workspaceId: string;
  activeSessionId: string | null;
  activeModel: string;
  notes: Array<{ id: string; title: string; content: string }>;
  bookmarks: Array<{ id: string; title: string; target_type: string; target_id: string }>;
}

export interface AppState {
  commandPaletteOpen: boolean;
  activePage: string;
  workspace: WorkspaceState;
  selection: SelectionState;
  visiblePanels: Record<string, boolean>;
  capabilities: any | null;
}

const DEFAULT_STATE: AppState = {
  commandPaletteOpen: false,
  activePage: 'explorer',
  workspace: {
    workspaceId: 'default_workspace',
    activeSessionId: null,
    activeModel: 'gpt2',
    notes: [],
    bookmarks: [],
  },
  selection: {
    selectedLayer: 0,
    selectedHead: 0,
    selectedNeuron: null,
    hoveredToken: null,
    selectedTokenIdx: null,
  },
  visiblePanels: {
    token_viewer: true,
    attention_heatmap: true,
    activation_heatmap: true,
    neuron_panel: true,
    layer_inspector: true,
    prediction_inspector: true,
    token_inspector: true,
  },
  capabilities: null,
};

let globalState: AppState = DEFAULT_STATE;
const listeners = new Set<() => void>();

export const appStore = {
  getState: () => globalState,
  setState: (fn: ((prev: AppState) => Partial<AppState>) | Partial<AppState>) => {
    const patch = typeof fn === 'function' ? fn(globalState) : fn;
    globalState = { ...globalState, ...patch };
    listeners.forEach(l => l());
  },
  subscribe: (listener: () => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};

export function useAppStore(): [AppState, typeof appStore.setState] {
  const [state, setState] = useState<AppState>(globalState);

  useEffect(() => {
    return appStore.subscribe(() => setState(globalState));
  }, []);

  return [state, appStore.setState];
}
