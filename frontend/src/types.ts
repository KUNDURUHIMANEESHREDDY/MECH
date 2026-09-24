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
  provenance?: string;
  provenance_note?: string;
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

/* ---------- Layer tensor DTOs (full per-token resid_post / mlp_post) ---------- */

export interface LayerTensorStats {
  resid_l2: number;
  mlp_mean: number;
  mlp_sparsity: number;
}

export interface LayerTensors {
  status: string;
  layer: number;
  prompt: string;
  tokens: string[];
  d_model: number;
  d_mlp: number;
  resid_post: number[][];
  mlp_post: number[][];
  stats: LayerTensorStats;
}

export interface LogitLensLayer {
  layer: number;
  top_token: string;
  top_k_tokens: Array<{ token: string; prob: number }>;
}

export interface LogitLensAll {
  status: string;
  method: string;
  prompt: string;
  layers: LogitLensLayer[];
}
