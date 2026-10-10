/* ---------- Runtime DTOs (data contract) ---------- */
/* Mirrors backend/contracts/models.py. Frontend must handle the
   unavailable shape: status "unavailable"/"error" with provenance
   "unavailable" and error/reason — never render it as a measurement. */

export type ProvenanceLabel = 'live' | 'seeded' | 'reference' | 'unavailable';

export type EvidenceLevel =
  | 'OBSERVATIONAL'
  | 'ATTRIBUTIONAL'
  | 'INTERVENTIONAL'
  | 'CAUSALLY_VALIDATED'
  | 'REPLICATED';

export type FieldProvenance = Record<string, ProvenanceLabel>;

export interface ProvenanceFields {
  provenance?: ProvenanceLabel;
  field_provenance?: FieldProvenance;
  attested?: boolean;
  reason?: string;
  error?: string;
  provenance_note?: string;
  model_loaded?: string;
  model_requested?: string;
  model_mismatch?: boolean;
  measured?: boolean;
  evidence_level?: EvidenceLevel;
  validation_eligible?: boolean;
  publication_eligible?: boolean;
}

export function isUnavailable(payload: unknown): boolean {
  if (!payload || typeof payload !== 'object') return false;
  const p = payload as Record<string, unknown>;
  return (
    p['provenance'] === 'unavailable' ||
    p['status'] === 'unavailable' ||
    p['status'] === 'error'
  );
}

export function isLive(payload: unknown): boolean {
  if (!payload || typeof payload !== 'object') return false;
  return (payload as Record<string, unknown>)['provenance'] === 'live';
}

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

export interface InferenceResponse extends ProvenanceFields {
  status?: string;
  model_name: string;
  tokens: TokenInfo[];
  generated_text: string;
  attention_maps: AttentionMap[];
  neuron_activations: NeuronActivation[];
  gpu_util: number | null;
  memory_util: number | null;
  provenance_note?: string;
}

export interface UnavailableResponse extends ProvenanceFields {
  status: 'unavailable' | 'error';
  provenance: 'unavailable';
}

export interface ModelInfo extends ProvenanceFields {
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

export interface LayerTensors extends ProvenanceFields {
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

export interface LogitLensAll extends ProvenanceFields {
  status: string;
  method: string;
  prompt: string;
  layers: LogitLensLayer[];
}
