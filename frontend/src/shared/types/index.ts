import type { FC } from 'react';

export type { FC };

/* ---------- Runtime DTOs (data contract) ---------- */

export interface AttentionMap {
  layer: number;
  head: number;
  tokens: string[];
  matrix: number[][];
}

export interface NeuronActivation {
  layer: number;
  index: number;
  activation: number;
  token_activations?: number[];
}

export interface TokenInfo {
  text: string;
  id: number;
}

export interface InferenceResponse {
  model_name: string;
  tokens: TokenInfo[];
  generated_text: string;
  attention_maps: AttentionMap[];
  neuron_activations: NeuronActivation[];
  gpu_util: number;
  memory_util: number;
}

export interface ModelInfo {
  model_name: string;
  status: string;
  num_layers: number;
  num_heads: number;
  hidden_dim: number;
}

/* ---------- Visualization state (derived from DTOs) ---------- */

export interface LayerData {
  index: number;
  heads: HeadData[];
}

export interface HeadData {
  index: number;
  attentionMatrix: number[][];
  neurons: NeuronData[];
}

export interface NeuronData {
  index: number;
  activation: number;
  tokenActivations?: number[];
}

export interface PanelState {
  selectedLayer: number;
  selectedHead: number;
  selectedNeuron: number | null;
  hoveredToken: number | null;
  error: string | null;
}

/* ── Resource Model ───────────────────────────────────────────────── */

export type ResourceKind =
  | 'model'
  | 'neuron'
  | 'head'
  | 'token'
  | 'layer'
  | 'circuit'
  | 'prompt'
  | 'dataset'
  | 'paper'
  | 'experiment'
  | 'sae'
  | 'feature'
  | 'session'
  | 'note'
  | 'bookmark'
  | 'workspace';

export interface Resource {
  id: string;
  kind: ResourceKind;
  label: string;
  icon?: string;
  metadata?: Record<string, unknown>;
  parentId?: string;
  children?: Resource[];
}

/* ── Panel Plugin Contract ───────────────────────────────────────── */

export interface PanelContext {
  selection: {
    resource: Resource | null;
    layer: number | null;
    head: number | null;
    neuron: number | null;
    token: number | null;
  };
  workspace: WorkspaceState;
}

export interface PanelPlugin {
  id: string;
  title: string;
  icon: string;
  category: string;
  resourceKinds: ResourceKind[];
  defaultDock: 'left' | 'center' | 'right' | 'bottom';
  Body: FC<PanelContext>;
  Header?: FC<PanelContext>;
  Inspector?: FC<PanelContext>;
  contextMenuItems?: Array<{
    label: string;
    action: string;
    icon?: string;
    disabled?: (ctx: PanelContext) => boolean;
  }>;
  commands?: Array<{
    id: string;
    label: string;
    icon?: string;
    handler: () => void;
  }>;
  persist?: (ctx: PanelContext) => unknown;
  restore?: (data: unknown, ctx: PanelContext) => void;
}

/* ── Selection & Window State ────────────────────────────────────── */

export interface SelectionState {
  resource: Resource | null;
  layer: number | null;
  head: number | null;
  neuron: number | null;
  token: number | null;
}

export interface WindowState {
  id: string;
  panelId: string;
  x: number;
  y: number;
  width: number;
  height: number;
  zIndex: number;
  detached: boolean;
  maximized: boolean;
}

/* ── Workspace State ─────────────────────────────────────────────── */

export interface WorkspaceState {
  workspaceId: string;
  activeModel: string | null;
  activeSessionId: string | null;
  selection: SelectionState;
  windows: WindowState[];
  camera: { x: number; y: number; zoom: number };
  notes: Array<{ id: string; title: string; content: string }>;
  timeline: Array<{ id: string; ts: string; event: string; payload?: unknown }>;
  console: Array<{ id: string; ts: string; level: string; message: string }>;
  visiblePanels: Record<string, boolean>;
  layout: unknown;
  activeWorkspace: string;
  workspaceNames: string[];
}

/* ── Command Bus Events ────────────────────────────────────────────── */

export type CommandEvent = {
  type: string;
  payload: unknown;
  timestamp?: number;
};

export type CommandHandler = (event: CommandEvent) => void;
