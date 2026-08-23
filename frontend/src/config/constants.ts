// Shared Constants for MECH Platform Frontend
// Mirrors backend constants from backend/core/config/model_config.py and capability definitions

// Model constants
export interface ModelInfo {
  name: string;
  displayName: string;
  layers: number;
  heads: number;
  hiddenSize: number;
  vocabSize: number;
  contextLength: number;
  source: 'huggingface' | 'local';
  modelId: string;
}

export const BUILTIN_MODELS: Record<string, ModelInfo> = {
  gpt2: {
    name: 'gpt2',
    displayName: 'GPT-2 Small (124M)',
    layers: 12,
    heads: 12,
    hiddenSize: 768,
    vocabSize: 50257,
    contextLength: 1024,
    source: 'huggingface',
    modelId: 'gpt2',
  },
  'gpt2-medium': {
    name: 'gpt2-medium',
    displayName: 'GPT-2 Medium (355M)',
    layers: 24,
    heads: 16,
    hiddenSize: 1024,
    vocabSize: 50257,
    contextLength: 1024,
    source: 'huggingface',
    modelId: 'gpt2-medium',
  },
  'gpt2-large': {
    name: 'gpt2-large',
    displayName: 'GPT-2 Large (774M)',
    layers: 36,
    heads: 20,
    hiddenSize: 1280,
    vocabSize: 50257,
    contextLength: 1024,
    source: 'huggingface',
    modelId: 'gpt2-large',
  },
  'gpt2-xl': {
    name: 'gpt2-xl',
    displayName: 'GPT-2 XL (1.5B)',
    layers: 48,
    heads: 25,
    hiddenSize: 1600,
    vocabSize: 50257,
    contextLength: 1024,
    source: 'huggingface',
    modelId: 'gpt2-xl',
  },
};

export const DEFAULT_MODEL = 'gpt2';

export const getModelInfo = (modelName: string): ModelInfo | undefined => {
  return BUILTIN_MODELS[modelName];
};

export const listModels = (): ModelInfo[] => {
  return Object.values(BUILTIN_MODELS);
};

// Task type constants
export type TaskType =
  | 'fact_retrieval'
  | 'induction'
  | 'ioi'
  | 'greater_than'
  | 'gender_bias'
  | 'sva'
  | 'custom';

export const TASK_TYPES: Record<TaskType, { label: string; description: string }> = {
  fact_retrieval: {
    label: 'Fact Retrieval',
    description: 'Retrieve factual knowledge (e.g., "The Eiffel Tower is in")',
  },
  induction: {
    label: 'Induction',
    description: 'Pattern completion and in-context learning',
  },
  ioi: {
    label: 'Indirect Object Identification',
    description: 'Syntactic reasoning task (IOI circuit)',
  },
  greater_than: {
    label: 'Greater Than',
    description: 'Numerical comparison reasoning',
  },
  gender_bias: {
    label: 'Gender Bias',
    description: 'Gender pronoun resolution bias detection',
  },
  sva: {
    label: 'Subject-Verb Agreement',
    description: 'Grammatical agreement reasoning',
  },
  custom: {
    label: 'Custom Task',
    description: 'User-defined task',
  },
};

// Component type constants
export type ComponentType =
  | 'mlp'
  | 'attention'
  | 'attention_head'
  | 'head'
  | 'residual'
  | 'sae_feature'
  | 'embedding'
  | 'unembedding'
  | 'layer_norm'
  | 'all';

export const COMPONENT_TYPES: Record<ComponentType, { label: string; description: string }> = {
  mlp: { label: 'MLP', description: 'Feed-forward network (c_fc/c_proj)' },
  attention: { label: 'Attention', description: 'Multi-head attention block' },
  attention_head: { label: 'Attention Head', description: 'Individual attention head' },
  head: { label: 'Head', description: 'Individual attention head (alias)' },
  residual: { label: 'Residual Stream', description: 'Residual stream pathway' },
  sae_feature: { label: 'SAE Feature', description: 'Sparse autoencoder feature direction' },
  embedding: { label: 'Embedding', description: 'Token/position embeddings' },
  unembedding: { label: 'Unembedding', description: 'Output projection to vocabulary' },
  layer_norm: { label: 'LayerNorm', description: 'Layer normalization' },
  all: { label: 'All Components', description: 'All component types' },
};

// Capability category constants (mirrored from backend)
export type CapabilityCategory =
  | 'localization'
  | 'causal'
  | 'dictionary_learning'
  | 'circuits'
  | 'verification'
  | 'comparative'
  | 'redundancy'
  | 'general';

export const CAPABILITY_CATEGORIES: Record<CapabilityCategory, { label: string; description: string }> = {
  localization: { label: 'Localization', description: 'Layer/component localization and attribution' },
  causal: { label: 'Causal', description: 'Causal intervention and mediation analysis' },
  dictionary_learning: { label: 'Dictionary Learning', description: 'SAE and dictionary learning methods' },
  circuits: { label: 'Circuits', description: 'Circuit discovery and analysis' },
  verification: { label: 'Verification', description: 'Scientific validation and falsification' },
  comparative: { label: 'Comparative', description: 'Cross-model comparison and alignment' },
  redundancy: { label: 'Redundancy', description: 'Backup heads and self-repair analysis' },
  general: { label: 'General', description: 'General platform capabilities' },
};

// Tool category constants (mirrored from backend)
export type ToolCategory =
  | 'localization'
  | 'causal'
  | 'dictionary_learning'
  | 'circuits'
  | 'verification'
  | 'comparative'
  | 'redundancy';

export const TOOL_CATEGORIES: Record<ToolCategory, { label: string; color: string }> = {
  localization: { label: 'Localization', color: '#0066cc' },
  causal: { label: 'Causal', color: '#d93025' },
  dictionary_learning: { label: 'Dictionary Learning', color: '#9334e6' },
  circuits: { label: 'Circuits', color: '#1e8e3e' },
  verification: { label: 'Verification', color: '#f9ab00' },
  comparative: { label: 'Comparative', color: '#d91d82' },
  redundancy: { label: 'Redundancy', color: '#00a8a8' },
};

// Built-in tool names (mirrored from backend plugin system)
export const BUILTIN_TOOLS = [
  'run_logit_lens',
  'run_activation_patching',
  'extract_sae_features',
  'discover_circuit',
  'run_path_patching',
  'run_hallucination_experiment',
  'run_semantic_falsification_probe',
  'evaluate_circuit_metrics',
  'run_live_tensor_intervention',
  'run_scientific_circuit_validation',
  'run_cross_model_universality_sweep',
  'discover_backup_redundant_circuits',
] as const;

export type BuiltinTool = typeof BUILTIN_TOOLS[number];

export const TOOL_CATEGORY_MAP: Record<BuiltinTool, ToolCategory> = {
  run_logit_lens: 'localization',
  run_activation_patching: 'causal',
  extract_sae_features: 'dictionary_learning',
  discover_circuit: 'circuits',
  run_path_patching: 'causal',
  run_hallucination_experiment: 'causal',
  run_semantic_falsification_probe: 'verification',
  evaluate_circuit_metrics: 'verification',
  run_live_tensor_intervention: 'causal',
  run_scientific_circuit_validation: 'verification',
  run_cross_model_universality_sweep: 'comparative',
  discover_backup_redundant_circuits: 'redundancy',
};

export const TOOL_DISPLAY_NAMES: Record<BuiltinTool, string> = {
  run_logit_lens: 'Logit Lens',
  run_activation_patching: 'Activation Patching',
  extract_sae_features: 'SAE Feature Discovery',
  discover_circuit: 'Circuit Discovery (ACDC)',
  run_path_patching: 'Path Patching',
  run_hallucination_experiment: 'Hallucination Pipeline',
  run_semantic_falsification_probe: 'Semantic Falsification',
  evaluate_circuit_metrics: 'Circuit Metrics (ACDC)',
  run_live_tensor_intervention: 'Live Intervention',
  run_scientific_circuit_validation: 'Scientific Validation',
  run_cross_model_universality_sweep: 'Cross-Model Universality',
  discover_backup_redundant_circuits: 'Backup Circuit Discovery',
};

export const TOOL_DESCRIPTIONS: Record<BuiltinTool, string> = {
  run_logit_lens: 'Decomposes intermediate residual stream states into token vocabulary projections per layer.',
  run_activation_patching: 'Causally intervenes by patching clean activations into corrupted runs to measure component importance.',
  extract_sae_features: 'Decomposes dense layer activations into monosemantic Sparse Autoencoder (SAE) feature directions.',
  discover_circuit: 'Applies Automated Circuit Discovery (ACDC) to extract a minimal subnetwork graph for a task.',
  run_path_patching: 'Intercepts and patches activations between sender component A and receiver component B to prove causal edge transmission.',
  run_hallucination_experiment: 'Executes the 4-pillar causal competition experiment between parametric MLP memory and induction attention heads.',
  run_semantic_falsification_probe: 'Tests candidate components with relational tuple steering or synthetic prefix-matching to falsify semantic role claims.',
  evaluate_circuit_metrics: 'Computes formal ACDC metrics: Faithfulness, Completeness, and Minimality on discovered subnetwork circuits.',
  run_live_tensor_intervention: 'Executes a live forward pass with PyTorch hooks intercepting and modifying activation tensors.',
  run_scientific_circuit_validation: 'Executes a 5-pillar scientific validation battery: held-out generalization, negative control specificity, multi-intervention triangulation, and statistical 95% confidence intervals.',
  run_cross_model_universality_sweep: 'Evaluates whether a discovered circuit represents a universal transformer primitive conserved across model families.',
  discover_backup_redundant_circuits: 'Discovers active compensatory backup heads and quantifies transformer self-repair resilience (Wang et al., 2022).',
};

// API-related constants
export const API_VERSION = 'v1';
export const API_PREFIX = '/api';
export const WS_PREFIX = '/ws';

// WebSocket message types
export type WSMessageType =
  | 'tool_progress'
  | 'tool_complete'
  | 'tool_error'
  | 'experiment_progress'
  | 'experiment_complete'
  | 'model_load_progress'
  | 'model_load_complete'
  | 'feature_flag_update'
  | 'heartbeat';

export interface WSMessage<T = unknown> {
  type: WSMessageType;
  payload: T;
  timestamp: string;
  requestId?: string;
}

// Default timeouts
export const DEFAULT_TIMEOUTS = {
  api: 30000, // 30 seconds
  tool_execution: 120000, // 2 minutes
  model_load: 60000, // 1 minute
  experiment: 300000, // 5 minutes
  websocket_reconnect: 5000, // 5 seconds
} as const;

// Pagination defaults
export const DEFAULT_PAGE_SIZE = 20;
export const MAX_PAGE_SIZE = 100;

// Storage keys
export const STORAGE_KEYS = {
  theme: 'mech-theme',
  apiKey: 'mech-api-key',
  userPreferences: 'mech-user-preferences',
  recentModels: 'mech-recent-models',
  recentTools: 'mech-recent-tools',
  experimentHistory: 'mech-experiment-history',
} as const;

// Feature flag keys (for localStorage sync)
export const FEATURE_FLAG_STORAGE_KEY = 'mech-feature-flags';

// Error codes
export const ERROR_CODES = {
  AUTH_REQUIRED: 'AUTH_REQUIRED',
  INVALID_API_KEY: 'INVALID_API_KEY',
  RATE_LIMITED: 'RATE_LIMITED',
  TOOL_NOT_FOUND: 'TOOL_NOT_FOUND',
  TOOL_EXECUTION_FAILED: 'TOOL_EXECUTION_FAILED',
  MODEL_NOT_LOADED: 'MODEL_NOT_LOADED',
  MODEL_LOAD_FAILED: 'MODEL_LOAD_FAILED',
  EXPERIMENT_NOT_FOUND: 'EXPERIMENT_NOT_FOUND',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  INTERNAL_ERROR: 'INTERNAL_ERROR',
  SERVICE_UNAVAILABLE: 'SERVICE_UNAVAILABLE',
} as const;

// Status codes
export const STATUS_CODES = {
  IDLE: 'idle',
  LOADING: 'loading',
  RUNNING: 'running',
  SUCCESS: 'success',
  ERROR: 'error',
  CANCELLED: 'cancelled',
} as const;

export type StatusCode = typeof STATUS_CODES[keyof typeof STATUS_CODES];

// Regex patterns
export const PATTERNS = {
  componentId: /^L\d+_(MLP|H\d+)$/,
  layerId: /^L\d+$/,
  headId: /^H\d+$/,
  saeFeatureId: /^SAE_\d+$/,
  modelName: /^[a-zA-Z0-9_-]+$/,
  experimentId: /^[a-zA-Z0-9_-]+$/,
} as const;

export default {
  BUILTIN_MODELS,
  DEFAULT_MODEL,
  getModelInfo,
  listModels,
  TASK_TYPES,
  COMPONENT_TYPES,
  CAPABILITY_CATEGORIES,
  TOOL_CATEGORIES,
  BUILTIN_TOOLS,
  TOOL_CATEGORY_MAP,
  TOOL_DISPLAY_NAMES,
  TOOL_DESCRIPTIONS,
  API_VERSION,
  API_PREFIX,
  WS_PREFIX,
  DEFAULT_TIMEOUTS,
  DEFAULT_PAGE_SIZE,
  MAX_PAGE_SIZE,
  STORAGE_KEYS,
  ERROR_CODES,
  STATUS_CODES,
  PATTERNS,
};