/**
 * demoService.ts
 *
 * Deterministic synthetic fallback for the MECH research canvas. When the
 * Python sidecar is unreachable (backend offline / first launch), the model
 * services fall back to this module so every visual panel still has real
 * structured data to render (tokens, attention matrices, neuron activations).
 *
 * The generator is seeded from the prompt, so repeated runs are stable and
 * the visualizations show coherent, structured patterns (diagonal attention,
 * induction/copying heads, prefix heads, focal neurons).
 */
import {
  AttentionMap,
  InferenceResponse,
  ModelInfo,
  NeuronActivation,
  TokenInfo,
} from '../shared/types';

const FALLBACK_MODELS = ['gpt2', 'gpt2-medium', 'gpt2-large'];
const NUM_LAYERS = 12;
const NUM_HEADS = 12;

export function demoAvailableModels(): string[] {
  return [...FALLBACK_MODELS];
}

/** ModelInfo with status 'demo' and explicit synthetic provenance. */
export function demoModelInfo(modelName: string): ModelInfo {
  return {
    model_name: modelName,
    status: 'demo',
    provenance: 'DEMO_SYNTHETIC',
    num_layers: NUM_LAYERS,
    num_heads: NUM_HEADS,
    hidden_dim: 768,
  };
}


/* ---------- helpers ---------- */

function hashSeed(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const WORD_BANK = [
  ' the', ' and', ' was', ' of', ' to', ' it', ' that', ' in', ' with',
  ' cat', ' dog', ' tree', ' city', ' book', ' light', ' story', ' night',
  ' morning', ' slowly', ' quietly', ' then', ' after', ' before', ' again',
];

function tokenize(text: string): TokenInfo[] {
  const parts = text.match(/[\w']+|[^\s\w']/g) ?? [];
  return parts.map((p, i) => ({ text: p, id: i }));
}

/**
 * Causal-style attention matrix with interpretable structure:
 *  - strong diagonal (self-attention)
 *  - prev-token bias (induction-like heads)
 *  - repetition copying (head index % 3 === 1): attends to earlier identical tokens
 *  - prefix priming (head index % 3 === 2): attends to the first token
 */
function makeAttention(rng: () => number, tokens: string[]): number[][] {
  const n = tokens.length;
  const mat: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let w = 0.15 + rng() * 0.45;
      if (j === i) w += 0.9;                      // self attention
      if (j === i - 1) w *= 2.6;                  // prev-token bias
      mat[i][j] = w;
    }
    // peak noise for realistic small weights
    for (let j = 0; j < i; j++) mat[i][j] += rng() * 0.05;
  }
  // normalize rows
  for (let i = 0; i < n; i++) {
    const s = mat[i].reduce((a, b) => a + b, 0) || 1;
    for (let j = 0; j < n; j++) mat[i][j] /= s;
  }
  return mat;
}

function attentionMapsFor(rng: () => number, tokens: TokenInfo[]): AttentionMap[] {
  const maps: AttentionMap[] = [];
  const texts = tokens.map((t) => t.text);
  const n = tokens.length;
  for (let li = 0; li < NUM_LAYERS; li++) {
    for (let hi = 0; hi < NUM_HEADS; hi++) {
      const base = makeAttention(rng, texts);
      if (n > 0) {
        // Modulation per head pattern
        if (hi % 3 === 2 && n > 1) {
          // prefix priming: pull weight toward first token
          for (let i = 1; i < n; i++) base[i][0] *= 2.2;
        } else if (hi % 3 === 1) {
          // induction: copy attention to earlier repeated token positions
          for (let i = 2; i < n; i++) {
            for (let j = 0; j < i; j++) {
              if (texts[j] === texts[i]) base[i][j] *= 2.8;
            }
          }
        }
        // re-normalize
        for (let i = 0; i < n; i++) {
          const s = base[i].reduce((a, b) => a + b, 0) || 1;
          for (let j = 0; j < n; j++) base[i][j] /= s;
        }
      }
      maps.push({ layer: li, head: hi, tokens: texts, matrix: base });
    }
  }
  return maps;
}

/**
 * Deterministic inference response. maxNewTokens generated tokens are
 * appended, so downstream token viewers render a full sequence.
 */
export function demoInference(prompt: string, maxNewTokens = 10): InferenceResponse {
  const rng = mulberry32(hashSeed(prompt));
  const tokens = tokenize(prompt.trim() || 'The capital of France is');
  const generated: string[] = [];
  for (let k = 0; k < maxNewTokens; k++) {
    const word = WORD_BANK[Math.floor(rng() * WORD_BANK.length)];
    generated.push(word);
  }
  const fullTokens = [...tokens, ...generated.map((w, i) => ({ text: w.trim(), id: 1000 + i }))];

  const neuronActivations: NeuronActivation[] = [];
  for (let li = 0; li < NUM_LAYERS; li++) {
    const count = 12 + Math.floor(rng() * 12);
    for (let k = 0; k < count; k++) {
      const idx = Math.floor(rng() * 768);
      const act = 0.4 + rng() * 5.5;
      neuronActivations.push({
        layer: li,
        index: idx,
        activation: act,
        token_activations: fullTokens.map(() => {
          if (rng() < 0.18) return 2 + rng() * 4;
          return rng() * 0.9;
        }),
      });
    }
  }

  return {
    model_name: 'gpt2',
    status: 'demo',
    provenance: 'DEMO_SYNTHETIC',
    tokens: fullTokens,
    generated_text: fullTokens.map((t) => t.text).join(' '),
    attention_maps: attentionMapsFor(rng, fullTokens),
    neuron_activations: neuronActivations,
    gpu_util: Math.round((18 + rng() * 34) * 10) / 10,
    memory_util: Math.round((34 + rng() * 28) * 10) / 10,
  };
}

/** Demo interaction response for Model Interaction feature. */
export interface InteractDemoResponse {
  status: 'demo';
  provenance: 'DEMO_SYNTHETIC';
  backend: string;
  model: string;
  prompt: string;
  response: string;
  n_generated: number;
  latency_ms: number;
}

export function demoInteraction(req: { prompt: string; backend?: string; max_new_tokens?: number; temperature?: number }): InteractDemoResponse {
  const rng = mulberry32(hashSeed(req.prompt));
  const backend = req.backend ?? 'gpt2';
  const maxNew = Math.min(Math.max(1, req.max_new_tokens ?? 64), 512);
  const temp = Math.max(0, Math.min(2, req.temperature ?? 0.7));
  
  const suffixes = [
    ' continues naturally with generated text that flows from the prompt context.',
    ' and then the model produces a coherent continuation based on the input.',
    ' which leads to an extended response demonstrating whole-answer generation.',
    ' followed by additional tokens forming a complete thought.',
  ];
  
  const suffix = suffixes[Math.floor(rng() * suffixes.length)];
  const response = `${req.prompt.trim()}${suffix}`;
  
  return {
    status: 'demo',
    provenance: 'DEMO_SYNTHETIC',
    backend,
    model: req.backend === 'ollama' ? 'llama3' : req.backend === 'openai' ? 'gpt-4o' : 'gpt2',
    prompt: req.prompt,
    response,
    n_generated: Math.floor(rng() * maxNew) + 1,
    latency_ms: Math.floor(rng() * 200) + 10,
  };
}