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
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
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
});
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
