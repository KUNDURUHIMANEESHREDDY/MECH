<template>
  <section class="network-tool" aria-labelledby="network-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">Explore / Weights</p>
        <h1 id="network-title">Network</h1>
        <p class="tool-intro">
          Every tensor of the loaded GPT-2, counted from live weights. Expand a block to read
          each attention head's measured Q/K/V/O norms and the MLP width. Nothing here is a
          diagram of a generic transformer.
        </p>
      </div>
      <div class="tool-header__actions">
        <span class="source-pill" :class="`source-pill--${sourceTone}`" role="status">
          <span class="source-dot" aria-hidden="true" />
          {{ sourceLabel }}
        </span>
        <button class="tool-button" type="button" :disabled="loading" @click="loadArchitecture">
          <span aria-hidden="true">↻</span>
          {{ loading ? 'Reading…' : 'Refresh weights' }}
        </button>
      </div>
    </header>

    <div v-if="errorMessage" class="tool-notice tool-notice--error" role="alert">
      <strong>Network read failed</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <section v-if="summary" class="metric-strip" aria-label="Network totals">
      <div class="metric-tile">
        <span class="metric-tile__value">{{ formatInt(summary.totalParams) }}</span>
        <span class="metric-tile__label">Parameters, counted live</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ summary.nLayers }}</span>
        <span class="metric-tile__label">Transformer blocks</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ summary.totalHeads }}</span>
        <span class="metric-tile__label">Attention heads</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ formatInt(summary.totalMlp) }}</span>
        <span class="metric-tile__label">MLP neurons</span>
      </div>
    </section>

    <section v-if="ledger.length" class="ledger" aria-label="Parameter ledger">
      <h2 class="section-title">Parameter ledger</h2>
      <ol class="ledger-list">
        <li v-for="row in ledger" :key="row.path" class="ledger-row">
          <code>{{ row.path }}</code>
          <span>{{ row.detail }}</span>
          <strong>{{ formatInt(row.params) }}</strong>
        </li>
      </ol>
    </section>

    <section v-if="layers.length" aria-label="Transformer blocks">
      <h2 class="section-title">Blocks</h2>
      <ol class="block-list">
        <li v-for="layer in layers" :key="layer.layer_index" class="block">
          <button
            type="button"
            class="block-head"
            :aria-expanded="expandedLayer === layer.layer_index"
            @click="toggleLayer(layer.layer_index)"
          >
            <span class="block-name">Block {{ layer.layer_index }}</span>
            <span class="block-meta">{{ layer.num_attention_heads }} heads · {{ formatInt(layer.num_mlp_neurons) }} MLP · {{ formatInt(layer.n_params) }} params</span>
            <span class="block-chevron" aria-hidden="true">{{ expandedLayer === layer.layer_index ? '▾' : '▸' }}</span>
          </button>
          <div v-if="expandedLayer === layer.layer_index" class="block-body">
            <div v-if="layerDetailError" class="tool-notice tool-notice--error" role="alert">
              <strong>Head detail unavailable</strong>
              <span>{{ layerDetailError }}</span>
            </div>
            <div v-else-if="layerDetailLoading" class="tool-state" role="status">Reading live head weights…</div>
            <ol v-else class="head-grid" :aria-label="`Heads of block ${layer.layer_index}`">
              <li v-for="head in layerDetailHeads" :key="head.head_index" class="head-chip">
                <strong>H{{ head.head_index }}</strong>
                <span>Q {{ formatWeight(head.q_weight_l2) }}</span>
                <span>K {{ formatWeight(head.k_weight_l2) }}</span>
                <span>V {{ formatWeight(head.v_weight_l2) }}</span>
                <span>O {{ formatWeight(head.o_weight_l2) }}</span>
              </li>
            </ol>
          </div>
        </li>
      </ol>
    </section>

    <div v-else-if="!loading" class="tool-state" role="status">
      <strong>No network returned.</strong>
      <span>Press Refresh weights to read the loaded model.</span>
    </div>

    <section aria-label="Live wiring">
      <h2 class="section-title">Wiring</h2>
      <p class="section-note">
        Run a prompt to light up real connections: token-to-token attention wires for one head,
        and input/output weight wires for one MLP neuron. Both come from live forward passes.
      </p>
      <form class="wire-form" data-testid="network-wire-form" @submit.prevent="runWiringPrompt">
        <label class="wire-field">
          <span>Prompt</span>
          <input v-model="wirePrompt" type="text" class="control" data-testid="network-wire-prompt" placeholder="The capital of France is" />
        </label>
        <button class="tool-button tool-button--primary" type="submit" data-testid="network-wire-run" :disabled="wiringLoading">
          {{ wiringLoading ? 'Running…' : 'Light up wires' }}
        </button>
        <button
          class="tool-button"
          type="button"
          :disabled="freshPromptState === 'loading'"
          @click="fetchFreshPrompt(true)"
        >
          {{ freshPromptState === 'loading' ? 'Creating…' : 'New prompt' }}
        </button>
      </form>
      <label class="auto-prompt-toggle">
        <input v-model="autoPrompt" type="checkbox" />
        <span>New model-created prompt every minute</span>
      </label>
      <p v-if="freshPromptNote" class="wire-note">{{ freshPromptNote }}</p>
      <div v-if="wiringError" class="tool-notice tool-notice--error" role="alert" data-testid="network-wire-error">
        <strong>Wiring run failed</strong>
        <span>{{ wiringError }}</span>
      </div>

      <div v-if="wireTokens.length" class="wire-panels">
        <div class="wire-panel">
          <div class="wire-panel__head">
            <h3>Attention wires</h3>
            <div class="wire-selectors">
              <label>Layer
                <select v-model.number="wireLayer" class="control control--inline">
                  <option v-for="n in 12" :key="n" :value="n - 1">L{{ n - 1 }}</option>
                </select>
              </label>
              <label>Head
                <select v-model.number="wireHead" class="control control--inline">
                  <option v-for="n in 12" :key="n" :value="n - 1">H{{ n - 1 }}</option>
                </select>
              </label>
            </div>
          </div>
          <p class="wire-caption" data-testid="network-wire-caption">{{ attentionEdgeCount }} wires above 0.08 · final-token focus: {{ finalFocusToken }}</p>
          <svg
            class="wire-svg"
            data-testid="network-wire-svg"
            :data-edge-count="attentionSvg.edges.length"
            :viewBox="`0 0 ${attentionSvg.width} ${attentionSvg.height}`"
            role="img"
            :aria-label="`Attention wiring for layer ${wireLayer} head ${wireHead}`"
          >
            <path
              v-for="edge in attentionSvg.edges"
              :key="`${edge.from}-${edge.to}`"
              :d="edge.d"
              class="wire wire--attention"
              :stroke-width="edge.width"
              :opacity="edge.opacity"
              data-testid="network-wire-edge"
            />
            <g v-for="node in attentionSvg.nodes" :key="node.index">
              <rect :x="node.x" :y="node.y" width="10" height="10" rx="5" class="wire-node" />
              <text :x="node.x + 5" :y="node.y + 24" text-anchor="middle" class="wire-label">{{ node.token }}</text>
            </g>
          </svg>
        </div>

        <div class="wire-panel">
          <div class="wire-panel__head">
            <h3>Neuron wires</h3>
            <div class="wire-selectors">
              <label>Layer
                <select v-model.number="wireNeuronLayer" class="control control--inline" @change="loadNeuronWires">
                  <option v-for="n in 12" :key="n" :value="n - 1">L{{ n - 1 }}</option>
                </select>
              </label>
              <label>Neuron
                <input v-model.number="wireNeuronIndex" type="number" min="0" max="3071" class="control control--inline control--number" @change="loadNeuronWires" />
              </label>
            </div>
          </div>
          <p class="wire-caption">{{ neuronEdgeCount }} strongest weight wires · green feeds, red drains</p>
          <div v-if="neuronWiresLoading" class="tool-state" role="status">Reading live neuron weights…</div>
          <div v-else-if="neuronWiresError" class="tool-notice tool-notice--error" role="alert">
            <strong>Neuron weights unavailable</strong>
            <span>{{ neuronWiresError }}</span>
          </div>
          <svg
            v-else-if="neuronSvg.edges.length"
            class="wire-svg"
            :viewBox="`0 0 ${neuronSvg.width} ${neuronSvg.height}`"
            role="img"
            :aria-label="`Weight wiring for neuron ${wireNeuronIndex} in layer ${wireNeuronLayer}`"
          >
            <path
              v-for="edge in neuronSvg.edges"
              :key="edge.key"
              :d="edge.d"
              :class="edge.positive ? 'wire wire--positive' : 'wire wire--negative'"
              :stroke-width="edge.width"
              :opacity="edge.opacity"
            />
            <g v-for="node in neuronSvg.nodes" :key="node.key">
              <rect :x="node.x" :y="node.y" :width="node.w" :height="node.h" rx="4" :class="node.class" />
              <text :x="node.x + node.w / 2" :y="node.y + node.h / 2 + 4" text-anchor="middle" class="wire-label">{{ node.label }}</text>
            </g>
          </svg>
        </div>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { api } from '../services/api';

type JsonRecord = Record<string, unknown>;

interface LayerSummary {
  layer_index: number;
  num_attention_heads: number;
  num_mlp_neurons: number;
  n_params: number;
}

interface LedgerRow {
  path: string;
  detail: string;
  params: number;
}

interface HeadNorms {
  head_index: number;
  q_weight_l2?: number;
  k_weight_l2?: number;
  v_weight_l2?: number;
  o_weight_l2?: number;
}

interface SvgEdge {
  key: string;
  d: string;
  width: number;
  opacity: number;
  from: number;
  to: number;
  positive?: boolean;
}

interface SvgNode {
  key: string;
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  index?: number;
  token?: string;
  class?: string;
}

interface AttentionMap {
  layer: number;
  head: number;
  tokens: string[];
  matrix: number[][];
}

interface WeightLink {
  dim: number;
  weight: number;
}

const ATTENTION_THRESHOLD = 0.08;
const MAX_ATTENTION_EDGES = 60;
const NEURON_LINKS = 12;

const loading = ref(false);
const errorMessage = ref('');
const layers = ref<LayerSummary[]>([]);
const ledger = ref<LedgerRow[]>([]);
const summary = ref<{ totalParams: number; nLayers: number; totalHeads: number; totalMlp: number } | null>(null);
const provenance = ref('unavailable');
const expandedLayer = ref<number | null>(null);
const layerDetailLoading = ref(false);
const layerDetailError = ref('');
const layerDetailHeads = ref<HeadNorms[]>([]);
const layerDetailCache = new Map<number, HeadNorms[]>();

const wirePrompt = ref('The capital of France is');
const wiringLoading = ref(false);
const wiringError = ref('');
const wireTokens = ref<string[]>([]);

const autoPrompt = ref(true);
const freshPromptNote = ref('');
const freshPromptState = ref<'idle' | 'loading'>('idle');
let freshPromptTimer: ReturnType<typeof setInterval> | null = null;
const attentionMaps = ref<AttentionMap[]>([]);
const wireLayer = ref(10);
const wireHead = ref(7);
const wireNeuronLayer = ref(5);
const wireNeuronIndex = ref(0);
const neuronWiresLoading = ref(false);
const neuronWiresError = ref('');
const neuronInLinks = ref<WeightLink[]>([]);
const neuronOutLinks = ref<WeightLink[]>([]);

const sourceTone = computed(() => {
  if (loading.value) return 'loading';
  if (provenance.value === 'live') return 'online';
  return 'offline';
});
const sourceLabel = computed(() => {
  if (loading.value) return 'Reading weights';
  if (provenance.value === 'live') return 'Live weights';
  return 'Weights unavailable';
});

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}
function formatInt(value: number | undefined): string {
  if (value === undefined) return '—';
  return Math.round(value).toLocaleString('en-US');
}
function formatWeight(value: number | undefined): string {
  if (value === undefined) return '—';
  return value.toFixed(2);
}

function shortToken(token: string): string {
  const cleaned = token.replace(/Ġ/g, ' ').trim();
  return cleaned.length > 10 ? `${cleaned.slice(0, 9)}…` : cleaned || '·';
}

const selectedAttentionMap = computed((): AttentionMap | null => {
  const found = attentionMaps.value.find(
    item => item.layer === wireLayer.value && item.head === wireHead.value,
  );
  return found ?? null;
});

const attentionEdgeCount = computed(() => attentionSvg.value.edges.length);

const finalFocusToken = computed(() => {
  const map = selectedAttentionMap.value;
  if (!map || !map.matrix.length) return '—';
  const row = map.matrix[map.matrix.length - 1] ?? [];
  let best = 0;
  for (let i = 1; i < row.length; i++) {
    if ((row[i] ?? 0) > (row[best] ?? 0)) best = i;
  }
  return shortToken(map.tokens[best] ?? '');
});

const attentionSvg = computed(() => {
  const map = selectedAttentionMap.value;
  const tokens = wireTokens.value;
  const width = 60 + Math.max(0, tokens.length - 1) * 70 + 30;
  const height = 130;
  const nodes: SvgNode[] = tokens.map((token, index) => ({
    key: `t-${index}`,
    x: 30 + index * 70 - 5,
    y: 88,
    w: 10,
    h: 10,
    label: shortToken(token),
    index,
    token,
  }));
  if (!map || !map.matrix.length) {
    return { width: Math.max(width, 200), height, nodes, edges: [] as SvgEdge[] };
  }
  const candidates: Array<{ from: number; to: number; weight: number }> = [];
  for (let to = 0; to < map.matrix.length && to < tokens.length; to++) {
    const row = map.matrix[to] ?? [];
    for (let from = 0; from < row.length && from < tokens.length; from++) {
      const weight = row[from] ?? 0;
      if (weight >= ATTENTION_THRESHOLD) {
        candidates.push({ from, to, weight });
      }
    }
  }
  candidates.sort((a, b) => b.weight - a.weight);
  const edges: SvgEdge[] = candidates.slice(0, MAX_ATTENTION_EDGES).map(edge => {
    const x1 = 30 + edge.from * 70;
    const x2 = 30 + edge.to * 70;
    const lift = edge.from === edge.to ? 26 : 14 + Math.abs(x2 - x1) * 0.35;
    return {
      key: `${edge.from}-${edge.to}`,
      from: edge.from,
      to: edge.to,
      d: `M ${x1} 88 Q ${(x1 + x2) / 2} ${88 - lift} ${x2} 88`,
      width: 0.8 + edge.weight * 3.2,
      opacity: 0.3 + edge.weight * 0.7,
    };
  });
  return { width: Math.max(width, 200), height, nodes, edges };
});

const neuronEdgeCount = computed(
  () => neuronInLinks.value.length + neuronOutLinks.value.length,
);

const neuronSvg = computed(() => {
  const width = 640;
  const rowHeight = 26;
  const inLinks = neuronInLinks.value;
  const outLinks = neuronOutLinks.value;
  const rows = Math.max(inLinks.length, outLinks.length, 1);
  const height = 60 + rows * rowHeight;
  const midY = 30 + ((rows - 1) * rowHeight) / 2;
  const nodes: SvgNode[] = [
    {
      key: 'neuron',
      x: 300,
      y: midY - 17,
      w: 40,
      h: 34,
      label: `N${wireNeuronIndex.value}`,
      class: 'wire-neuron',
    },
  ];
  const edges: SvgEdge[] = [];
  const peak = Math.max(
    0.0001,
    ...inLinks.map(link => Math.abs(link.weight)),
    ...outLinks.map(link => Math.abs(link.weight)),
  );
  inLinks.forEach((link, row) => {
    const y = 30 + row * rowHeight;
    nodes.push({
      key: `in-${link.dim}`,
      x: 10,
      y: y - 9,
      w: 150,
      h: 18,
      label: `d${link.dim} ${link.weight >= 0 ? '+' : ''}${link.weight.toFixed(2)}`,
      class: 'wire-dim',
    });
    edges.push({
      key: `in-${link.dim}`,
      from: link.dim,
      to: wireNeuronIndex.value,
      d: `M 160 ${y} C 210 ${y}, 250 ${midY}, 300 ${midY}`,
      width: 0.8 + (Math.abs(link.weight) / peak) * 3,
      opacity: 0.35 + (Math.abs(link.weight) / peak) * 0.6,
      positive: link.weight >= 0,
    });
  });
  outLinks.forEach((link, row) => {
    const y = 30 + row * rowHeight;
    nodes.push({
      key: `out-${link.dim}`,
      x: 480,
      y: y - 9,
      w: 150,
      h: 18,
      label: `d${link.dim} ${link.weight >= 0 ? '+' : ''}${link.weight.toFixed(2)}`,
      class: 'wire-dim',
    });
    edges.push({
      key: `out-${link.dim}`,
      from: wireNeuronIndex.value,
      to: link.dim,
      d: `M 340 ${midY} C 390 ${midY}, 430 ${y}, 480 ${y}`,
      width: 0.8 + (Math.abs(link.weight) / peak) * 3,
      opacity: 0.35 + (Math.abs(link.weight) / peak) * 0.6,
      positive: link.weight >= 0,
    });
  });
  return { width, height, nodes, edges };
});

function readLinks(value: unknown): WeightLink[] {
  if (!Array.isArray(value)) return [];
  const links: WeightLink[] = [];
  for (const item of value) {
    if (!isRecord(item)) continue;
    const dim = asNumber(item.dim);
    const weight = asNumber(item.weight);
    if (dim === undefined || weight === undefined) continue;
    links.push({ dim, weight });
  }
  links.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  return links.slice(0, NEURON_LINKS);
}

async function runWiringPrompt(): Promise<void> {
  const text = wirePrompt.value.trim();
  if (!text || wiringLoading.value) return;
  wiringLoading.value = true;
  wiringError.value = '';
  try {
    const res = await api.infer(text);
    if (!isRecord(res) || res.status === 'error') {
      throw new Error(typeof res.error === 'string' ? res.error : 'Inference failed.');
    }
    const tokens = Array.isArray(res.tokens)
      ? res.tokens.map(token => String(isRecord(token) ? token.text ?? '' : token ?? ''))
      : [];
    if (!tokens.length) throw new Error('The backend returned no token sequence.');
    wireTokens.value = tokens;
    const maps: AttentionMap[] = [];
    if (Array.isArray(res.attention_maps)) {
      for (const item of res.attention_maps) {
        if (!isRecord(item)) continue;
        const layer = asNumber(item.layer);
        const head = asNumber(item.head);
        if (layer === undefined || head === undefined || !Array.isArray(item.matrix)) continue;
        maps.push({
          layer,
          head,
          tokens,
          matrix: item.matrix as number[][],
        });
      }
    }
    attentionMaps.value = maps;
    await loadNeuronWires();
  } catch (error) {
    wiringError.value = error instanceof Error ? error.message : String(error);
    wireTokens.value = [];
    attentionMaps.value = [];
  } finally {
    wiringLoading.value = false;
  }
}

async function loadNeuronWires(): Promise<void> {
  const layer = Math.max(0, Math.min(11, Math.floor(wireNeuronLayer.value) || 0));
  const index = Math.max(0, Math.min(3071, Math.floor(wireNeuronIndex.value) || 0));
  wireNeuronLayer.value = layer;
  wireNeuronIndex.value = index;
  neuronWiresLoading.value = true;
  neuronWiresError.value = '';
  try {
    const res = await api.gpt2Neuron(layer, index, 'mlp', 32);
    if (!isRecord(res) || res.status === 'error') {
      throw new Error(typeof res.error === 'string' ? res.error : 'Neuron read failed.');
    }
    neuronInLinks.value = [
      ...readLinks(res.top_input_weights_positive).slice(0, 6),
      ...readLinks(res.top_input_weights_negative).slice(0, 6),
    ];
    neuronOutLinks.value = [
      ...readLinks(res.top_output_weights_positive).slice(0, 6),
      ...readLinks(res.top_output_weights_negative).slice(0, 6),
    ];
  } catch (error) {
    neuronWiresError.value = error instanceof Error ? error.message : String(error);
    neuronInLinks.value = [];
    neuronOutLinks.value = [];
  } finally {
    neuronWiresLoading.value = false;
  }
}

async function loadArchitecture(): Promise<void> {
  loading.value = true;
  errorMessage.value = '';
  try {
    const res = await api.gpt2Architecture();
    if (!isRecord(res) || res.status === 'error') {
      throw new Error(typeof res.error === 'string' ? res.error : 'Architecture read failed.');
    }
    provenance.value = typeof res.provenance === 'string' ? res.provenance : 'unavailable';
    const rawLayers = Array.isArray(res.layers) ? res.layers : [];
    const parsed: LayerSummary[] = [];
    for (const item of rawLayers) {
      if (!isRecord(item)) continue;
      const layerIndex = asNumber(item.layer_index);
      if (layerIndex === undefined) continue;
      parsed.push({
        layer_index: layerIndex,
        num_attention_heads: asNumber(item.num_attention_heads) ?? 0,
        num_mlp_neurons: asNumber(item.num_mlp_neurons) ?? 0,
        n_params: asNumber(item.n_params) ?? 0,
      });
    }
    parsed.sort((a, b) => a.layer_index - b.layer_index);
    layers.value = parsed;

    const modules = Array.isArray(res.modules) ? res.modules : [];
    const rows: LedgerRow[] = [];
    let total = 0;
    const pushRow = (path: string, detail: string, params: number) => {
      rows.push({ path, detail, params });
      total += params;
    };
    for (const module of modules) {
      if (!isRecord(module)) continue;
      const id = String(module.id ?? '');
      if (id === 'wte' || id === 'wpe' || id === 'ln_f') {
        const shape = Array.isArray(module.shape) ? module.shape.join(' × ') : '';
        pushRow(String(module.path ?? id), shape, asNumber(module.n_params) ?? 0);
      } else if (id === 'lm_head') {
        // lm_head is weight-tied with wte: same tensor, zero new parameters.
        const shape = Array.isArray(module.shape) ? module.shape.join(' × ') : '';
        pushRow(String(module.path ?? id), `${shape} · tied with wte, 0 new`, 0);
      }
    }
    for (const layer of parsed) {
      pushRow(`transformer.h.${layer.layer_index}`, `${layer.num_attention_heads} heads · ${formatInt(layer.num_mlp_neurons)} MLP`, layer.n_params);
    }
    ledger.value = rows;
    summary.value = {
      totalParams: total,
      nLayers: parsed.length,
      totalHeads: parsed.reduce((sum, layer) => sum + layer.num_attention_heads, 0),
      totalMlp: parsed.reduce((sum, layer) => sum + layer.num_mlp_neurons, 0),
    };
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
    layers.value = [];
    ledger.value = [];
    summary.value = null;
    provenance.value = 'unavailable';
  } finally {
    loading.value = false;
  }
}

async function toggleLayer(layerIndex: number): Promise<void> {
  if (expandedLayer.value === layerIndex) {
    expandedLayer.value = null;
    return;
  }
  expandedLayer.value = layerIndex;
  const cached = layerDetailCache.get(layerIndex);
  if (cached) {
    layerDetailHeads.value = cached;
    layerDetailError.value = '';
    return;
  }
  layerDetailLoading.value = true;
  layerDetailError.value = '';
  layerDetailHeads.value = [];
  try {
    const res = await api.gpt2Layer(layerIndex);
    if (!isRecord(res) || res.status === 'error' || !Array.isArray(res.attention_heads)) {
      throw new Error(typeof res.error === 'string' ? res.error : 'Head weights unreadable.');
    }
    const heads: HeadNorms[] = [];
    for (const item of res.attention_heads) {
      if (!isRecord(item)) continue;
      const headIndex = asNumber(item.head_index);
      if (headIndex === undefined) continue;
      heads.push({
        head_index: headIndex,
        q_weight_l2: asNumber(item.q_weight_l2),
        k_weight_l2: asNumber(item.k_weight_l2),
        v_weight_l2: asNumber(item.v_weight_l2),
        o_weight_l2: asNumber(item.o_weight_l2),
      });
    }
    heads.sort((a, b) => a.head_index - b.head_index);
    layerDetailCache.set(layerIndex, heads);
    layerDetailHeads.value = heads;
  } catch (error) {
    layerDetailError.value = error instanceof Error ? error.message : String(error);
  } finally {
    layerDetailLoading.value = false;
  }
}

onMounted(() => {
  void loadArchitecture();
  void fetchFreshPrompt(false);
  startFreshPromptTimer();
});

onBeforeUnmount(() => {
  stopFreshPromptTimer();
});

function freshPromptTimestamp(): string {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

async function fetchFreshPrompt(manual: boolean): Promise<void> {
  if (freshPromptState.value === 'loading') return;
  if (wiringLoading.value) {
    if (manual) freshPromptNote.value = 'Wait for the wiring run to finish, then try again.';
    return;
  }
  if (typeof navigator !== 'undefined' && !navigator.onLine) {
    freshPromptNote.value = 'Reconnect this browser before requesting a model-created prompt.';
    return;
  }
  freshPromptState.value = 'loading';
  try {
    const res = await api.gpt2FreshPrompt();
    const text = isRecord(res) && typeof res.prompt === 'string' ? res.prompt.trim() : '';
    const provenance = isRecord(res) && typeof res.provenance === 'string' ? res.provenance : 'unavailable';
    if (!text || provenance !== 'live') {
      freshPromptNote.value = 'The model did not return a usable prompt; the previous text was kept.';
      return;
    }
    wirePrompt.value = text;
    freshPromptNote.value = `New prompt created by GPT-2 (live) · ${freshPromptTimestamp()}`;
  } catch (error) {
    freshPromptNote.value = `Fresh prompt unavailable: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    freshPromptState.value = 'idle';
  }
}

function startFreshPromptTimer(): void {
  stopFreshPromptTimer();
  freshPromptTimer = setInterval(() => {
    if (!autoPrompt.value || document.hidden) return;
    void fetchFreshPrompt(false);
  }, 60_000);
}

function stopFreshPromptTimer(): void {
  if (freshPromptTimer !== null) {
    clearInterval(freshPromptTimer);
    freshPromptTimer = null;
  }
}
</script>

<style scoped>
.network-tool {
  display: flex;
  flex-direction: column;
  gap: 14px;
  width: 100%;
  max-width: 1120px;
  color: var(--text);
  font-size: 13px;
}

.network-tool h1,
.network-tool h2,
.network-tool p {
  margin: 0;
}

.tool-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.tool-eyebrow {
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.tool-header h1 {
  font-size: 24px;
  line-height: 1.15;
  font-weight: 720;
  letter-spacing: -0.02em;
}

.tool-intro {
  max-width: 720px;
  margin-top: 4px;
  color: var(--text-muted);
  line-height: 1.5;
}

.tool-header__actions {
  display: flex;
  flex: none;
  align-items: center;
  gap: 8px;
}

.source-pill {
  display: inline-flex;
  min-height: 26px;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  white-space: nowrap;
}

.source-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}

.source-pill--online {
  border-color: var(--success);
  color: var(--success);
}

.source-pill--loading {
  color: var(--warning);
}

.source-pill--offline {
  color: var(--danger);
}

.tool-button {
  display: inline-flex;
  min-height: 34px;
  align-items: center;
  gap: 6px;
  padding: 0 12px;
  border: 1px solid var(--border-light);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.tool-button:hover:not(:disabled) {
  background: var(--bg-hover);
}

.tool-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.tool-notice {
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--danger);
  border-radius: 7px;
  background: var(--surface-2);
  font-size: 12px;
}

.metric-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}

.metric-tile {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--surface);
}

.metric-tile__value {
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 17px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.metric-tile__label {
  color: var(--text-muted);
  font-size: 10px;
}

.section-title {
  margin-bottom: 8px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.ledger,
.block-list {
  margin: 0;
  padding: 0;
  list-style: none;
}

.ledger-list {
  margin: 0;
  padding: 0;
  list-style: none;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--surface);
  overflow: hidden;
}

.ledger-row {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) auto;
  gap: 10px;
  align-items: baseline;
  padding: 7px 12px;
  border-bottom: 1px solid var(--border);
  font-size: 11px;
}

.ledger-row:last-child {
  border-bottom: 0;
}

.ledger-row code {
  font-family: var(--font-mono);
  overflow-wrap: anywhere;
}

.ledger-row span {
  color: var(--text-muted);
}

.ledger-row strong {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

.block-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.block {
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--surface);
  overflow: hidden;
}

.block-head {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 10px;
  align-items: baseline;
  width: 100%;
  padding: 10px 12px;
  border: 0;
  background: transparent;
  color: var(--text);
  cursor: pointer;
  text-align: left;
  font-size: 12px;
}

.block-head:hover {
  background: var(--bg-hover);
}

.block-name {
  font-weight: 700;
}

.block-meta {
  color: var(--text-muted);
  font-family: var(--font-mono);
  font-size: 11px;
  overflow-wrap: anywhere;
}

.block-chevron {
  color: var(--text-muted);
}

.block-body {
  padding: 0 12px 12px;
  border-top: 1px solid var(--border);
}

.head-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin: 12px 0 0;
  padding: 0;
  list-style: none;
}

.head-chip {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  font-family: var(--font-mono);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}

.head-chip strong {
  font-size: 12px;
}

.head-chip span {
  color: var(--text-dim);
}

.tool-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 28px 16px;
  color: var(--text-muted);
  text-align: center;
}

.section-note {
  margin: -6px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.wire-form {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: 10px;
}

.wire-field {
  display: flex;
  min-width: 220px;
  flex: 1 1 280px;
  flex-direction: column;
  gap: 4px;
  font-size: 11px;
  font-weight: 650;
  color: var(--text-dim);
}

.wire-field .control {
  min-height: 36px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  font-size: 13px;
}

.tool-button--primary {
  border-color: var(--primary);
  background: var(--primary);
  color: #ffffff;
}

.tool-button--primary:hover:not(:disabled) {
  background: var(--primary-focus);
}

.wire-panels {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 12px;
}

.wire-panel {
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--surface);
  padding: 12px;
  min-width: 0;
}

.wire-panel__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 4px;
}

.wire-panel h3 {
  margin: 0;
  font-size: 13px;
  font-weight: 650;
}

.wire-selectors {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.wire-selectors label {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--text-muted);
}

.control--inline {
  min-height: 30px;
  padding: 0 6px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  font-size: 12px;
}

.control--number {
  width: 84px;
}

.wire-caption {
  margin: 0 0 6px;
  color: var(--text-muted);
  font-size: 11px;
}

.auto-prompt-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin: 0;
  color: var(--text-dim);
  font-size: 11px;
  cursor: pointer;
}

.auto-prompt-toggle input {
  width: 14px;
  height: 14px;
  accent-color: var(--primary);
}

.wire-note {
  margin: 0;
  color: var(--text-muted);
  font-family: var(--font-mono);
  font-size: 11px;
}

.wire-svg {
  width: 100%;
  height: auto;
  display: block;
}

.wire {
  fill: none;
  stroke: var(--primary);
}

.wire--positive {
  stroke: var(--success);
}

.wire--negative {
  stroke: var(--danger);
}

.wire-node {
  fill: var(--text-dim);
}

.wire-dim {
  fill: var(--surface-2);
  stroke: var(--border);
}

.wire-neuron {
  fill: var(--primary);
}

.wire-label {
  font-family: var(--font-mono);
  font-size: 11px;
  fill: var(--text-dim);
}

@media (max-width: 760px) {
  .tool-header {
    flex-direction: column;
    align-items: stretch;
  }

  .metric-strip,
  .head-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .ledger-row {
    grid-template-columns: minmax(0, 1fr) auto;
  }

  .ledger-row span {
    display: none;
  }
}
</style>
