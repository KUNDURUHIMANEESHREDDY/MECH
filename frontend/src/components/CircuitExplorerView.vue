<template>
  <section class="circuit-tool explorer-page" aria-labelledby="circuit-explorer-title">
    <header class="tool-header">
      <div>
        <span class="tool-kicker">Interpretability / registry</span>
        <h2 id="circuit-explorer-title">Circuit Explorer</h2>
        <p>Inspect registered mechanisms and the component members returned by the backend.</p>
      </div>
      <div class="tool-header__actions">
        <span class="source-badge" :class="`source-badge--${sourceTone}`" data-testid="circuit-source">
          <span class="source-badge__dot" aria-hidden="true" />
          {{ sourceLabel }}
        </span>
        <button class="tool-button" type="button" :disabled="loading" @click="loadCircuits">
          {{ loading ? 'Refreshing…' : 'Refresh registry' }}
        </button>
      </div>
    </header>

    <div v-if="errorMessage" class="tool-notice" :class="offline ? 'tool-notice--warning' : 'tool-notice--error'" role="alert">
      <strong>{{ offline ? 'Circuit registry offline' : 'Circuit registry request failed' }}</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <div v-if="loading" class="tool-state" role="status">
      <span class="loading-bar" aria-hidden="true" />
      <strong>Loading circuit records…</strong>
      <span>Only records returned by <code>/api/circuits</code> are shown.</span>
    </div>

    <div v-else-if="!circuits.length" class="tool-state" role="status">
      <strong>No circuit records returned.</strong>
      <span>The registry is reachable but did not provide a usable circuit list.</span>
    </div>

    <div v-else class="tool-layout">
      <aside class="registry-panel" aria-label="Circuit registry">
        <div class="registry-panel__heading">
          <div>
            <span class="panel-kicker">Returned records</span>
            <h3>Circuit registry</h3>
          </div>
          <span class="count-badge">{{ circuits.length }}</span>
        </div>
        <label class="search-field">
          <span>Search circuits</span>
          <input v-model="query" type="search" placeholder="Name, paper, or ID" autocomplete="off" />
        </label>
        <label class="search-field">
          <span>Selected circuit</span>
          <select v-model="selectedId" class="input-text circuit-select">
            <option v-for="circuit in circuits" :key="`select-${circuit.circuit_id}`" :value="circuit.circuit_id">
              {{ circuit.name || circuit.circuit_id }}
            </option>
          </select>
        </label>
        <div class="registry-list" role="listbox" aria-label="Circuit records">
          <button
            v-for="circuit in filteredCircuits"
            :key="circuit.circuit_id"
            class="registry-item"
            :class="{ 'registry-item--selected': selectedId === circuit.circuit_id }"
            type="button"
            role="option"
            :aria-selected="selectedId === circuit.circuit_id"
            @click="selectedId = circuit.circuit_id"
          >
            <span class="registry-item__name">{{ circuit.name || circuit.circuit_id }}</span>
            <span class="registry-item__meta">{{ circuit.paper || 'Paper unavailable' }}</span>
            <span class="registry-item__id">{{ circuit.circuit_id }}</span>
          </button>
          <p v-if="!filteredCircuits.length" class="registry-empty">No records match this search.</p>
        </div>
      </aside>

      <article class="detail-panel" aria-live="polite">
        <div v-if="detailLoading" class="detail-state" role="status">Loading returned circuit members…</div>
        <div v-else-if="detailError" class="detail-state detail-state--error" role="alert">
          <strong>Member details unavailable.</strong>
          <span>{{ detailError }}</span>
        </div>
        <template v-else-if="detail">
          <header class="detail-header">
            <div>
              <span class="panel-kicker">Selected record</span>
              <h3>{{ detail.name || detail.circuit_id }}</h3>
              <code>{{ detail.circuit_id }}</code>
            </div>
            <span class="source-badge source-badge--compact" :class="`source-badge--${detailTone}`">
              {{ detailTone === 'live' ? 'Live payload' : 'Reference record' }}
            </span>
          </header>

          <p class="detail-description">{{ detail.description || 'No description returned by the backend.' }}</p>

          <dl class="metric-grid" aria-label="Returned circuit metrics">
            <div>
              <dt>Faithfulness</dt>
              <dd>{{ formatMetric(detail.faithfulness) }}</dd>
            </div>
            <div>
              <dt>Completeness</dt>
              <dd>{{ formatMetric(detail.completeness) }}</dd>
            </div>
            <div>
              <dt>Minimality</dt>
              <dd>{{ formatMetric(detail.minimality) }}</dd>
            </div>
            <div>
              <dt>Nodes / edges</dt>
              <dd>{{ detail.node_count ?? '—' }} / {{ detail.edge_count ?? '—' }}</dd>
            </div>
          </dl>

          <section class="member-section" aria-labelledby="attention-members-title">
            <div class="section-heading">
              <div>
                <span class="panel-kicker">Returned members</span>
                <h4 id="attention-members-title">Attention heads</h4>
              </div>
              <span class="count-badge">{{ heads.length }}</span>
            </div>
            <div class="member-list">
              <span v-if="!heads.length" class="empty-inline">No attention heads returned.</span>
              <span v-for="head in heads" :key="head.node_id" class="member-chip" :title="head.description || head.label">
                <strong>{{ head.label }}</strong>
                <small v-if="head.importance_score !== undefined">{{ formatNumber(head.importance_score) }}</small>
              </span>
            </div>
          </section>

          <section class="member-section" aria-labelledby="other-members-title">
            <div class="section-heading">
              <div>
                <span class="panel-kicker">Returned members</span>
                <h4 id="other-members-title">Other components</h4>
              </div>
              <span class="count-badge">{{ others.length }}</span>
            </div>
            <div class="member-list">
              <span v-if="!others.length" class="empty-inline">No other components returned.</span>
              <span v-for="node in others" :key="node.node_id" class="member-chip" :title="node.description || node.label">
                <strong>{{ node.label }}</strong>
                <small>{{ node.node_type || 'type unavailable' }}</small>
              </span>
            </div>
          </section>

          <footer class="detail-footer">
            <span>Source: <code>GET /api/circuits/:circuit_id</code></span>
            <span>Registry values are not regenerated by this view.</span>
          </footer>
        </template>
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { apiUrl } from '../services/api';

type Provenance = 'live' | 'reference' | 'unavailable';
type SourceTone = 'online' | 'offline' | 'loading' | 'reference';
type JsonRecord = Record<string, unknown>;

interface CircuitSummary {
  circuit_id: string;
  name?: string;
  paper?: string;
  arxiv_id?: string;
  description?: string;
  faithfulness?: number;
  completeness?: number;
  minimality?: number;
  node_count?: number;
  edge_count?: number;
}

interface CircuitNode {
  node_id: string;
  node_type: string;
  label: string;
  description?: string;
  importance_score?: number;
}

interface CircuitDetail extends CircuitSummary {
  nodes?: CircuitNode[];
  provenance?: string;
}

const circuits = ref<CircuitSummary[]>([]);
const selectedId = ref('');
const detail = ref<CircuitDetail | null>(null);
const query = ref('');
const loading = ref(true);
const detailLoading = ref(false);
const offline = ref(false);
const errorMessage = ref('');
const detailError = ref('');
const detailVersion = ref(0);

const filteredCircuits = computed(() => {
  const needle = query.value.trim().toLowerCase();
  if (!needle) return circuits.value;
  return circuits.value.filter(circuit => [circuit.name, circuit.paper, circuit.circuit_id]
    .filter(Boolean)
    .some(value => String(value).toLowerCase().includes(needle)));
});

const sourceTone = computed<SourceTone>(() => {
  if (loading.value) return 'loading';
  if (offline.value) return 'offline';
  return 'reference';
});
const sourceLabel = computed(() => {
  if (loading.value) return 'Loading registry';
  if (offline.value) return 'Offline / unavailable';
  return 'Reference registry';
});
const detailTone = computed<Provenance>(() => detail.value?.provenance === 'live' ? 'live' : 'reference');
const heads = computed(() => (detail.value?.nodes ?? []).filter(node => node.node_type === 'attention_head'));
const others = computed(() => (detail.value?.nodes ?? []).filter(node => node.node_type !== 'attention_head'));

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function text(value: unknown): string {
  return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? String(value) : '';
}

function errorOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function looksOffline(message: string): boolean {
  return /offline|failed to fetch|network|connection|timeout|load/i.test(message);
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path), { headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error(`GET ${path} returned ${response.status}`);
  return response.json() as Promise<T>;
}

function normalizeSummary(value: unknown): CircuitSummary | null {
  if (!isRecord(value)) return null;
  const id = text(value.circuit_id) || text(value.id);
  if (!id) return null;
  return {
    circuit_id: id,
    name: text(value.name) || undefined,
    paper: text(value.paper) || undefined,
    arxiv_id: text(value.arxiv_id) || undefined,
    description: text(value.description) || undefined,
    faithfulness: typeof value.faithfulness === 'number' ? value.faithfulness : undefined,
    completeness: typeof value.completeness === 'number' ? value.completeness : undefined,
    minimality: typeof value.minimality === 'number' ? value.minimality : undefined,
    node_count: typeof value.node_count === 'number' ? value.node_count : undefined,
    edge_count: typeof value.edge_count === 'number' ? value.edge_count : undefined,
  };
}

function normalizeDetail(value: unknown, fallback: CircuitSummary): CircuitDetail | null {
  if (!isRecord(value)) return null;
  const nodes = Array.isArray(value.nodes) ? value.nodes.flatMap(item => {
    if (!isRecord(item)) return [];
    const id = text(item.node_id) || text(item.id);
    const label = text(item.label) || id;
    if (!id || !label) return [];
    return [{
      node_id: id,
      node_type: text(item.node_type) || text(item.type) || 'type unavailable',
      label,
      description: text(item.description) || undefined,
      importance_score: typeof item.importance_score === 'number' ? item.importance_score : undefined,
    }];
  }) : [];
  return {
    ...fallback,
    description: text(value.description) || fallback.description,
    faithfulness: typeof value.faithfulness === 'number' ? value.faithfulness : fallback.faithfulness,
    completeness: typeof value.completeness === 'number' ? value.completeness : fallback.completeness,
    minimality: typeof value.minimality === 'number' ? value.minimality : fallback.minimality,
    node_count: typeof value.node_count === 'number' ? value.node_count : nodes.length,
    edge_count: typeof value.edge_count === 'number' ? value.edge_count : undefined,
    provenance: text(value.provenance) || undefined,
    nodes,
  };
}

async function loadCircuits(): Promise<void> {
  loading.value = true;
  offline.value = false;
  errorMessage.value = '';
  try {
    const response = await getJson<unknown>('/api/circuits');
    const values = Array.isArray(response) ? response : isRecord(response) && Array.isArray(response.circuits) ? response.circuits : [];
    circuits.value = values.map(normalizeSummary).filter((item): item is CircuitSummary => Boolean(item));
    if (!circuits.value.some(circuit => circuit.circuit_id === selectedId.value)) {
      selectedId.value = circuits.value[0]?.circuit_id ?? '';
    }
  } catch (error) {
    circuits.value = [];
    detail.value = null;
    errorMessage.value = errorOf(error);
    offline.value = looksOffline(errorMessage.value);
  } finally {
    loading.value = false;
  }
}

async function loadDetail(): Promise<void> {
  const id = selectedId.value;
  const summary = circuits.value.find(circuit => circuit.circuit_id === id);
  if (!id || !summary) {
    detail.value = null;
    return;
  }
  const version = ++detailVersion.value;
  detailLoading.value = true;
  detailError.value = '';
  try {
    const response = await getJson<unknown>(`/api/circuits/${encodeURIComponent(id)}`);
    if (version !== detailVersion.value) return;
    detail.value = normalizeDetail(response, summary);
    if (!detail.value) detailError.value = 'The backend returned an unreadable circuit record.';
  } catch (error) {
    if (version !== detailVersion.value) return;
    detail.value = null;
    detailError.value = errorOf(error);
  } finally {
    if (version === detailVersion.value) detailLoading.value = false;
  }
}

function formatMetric(value: number | undefined): string {
  return value === undefined ? 'Unavailable' : `${Math.round(value * 100)}%`;
}

function formatNumber(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : 'Unavailable';
}

watch(selectedId, () => {
  void loadDetail();
});

onMounted(async () => {
  await loadCircuits();
  await loadDetail();
});
</script>

<style scoped>
.circuit-tool {
  min-height: 100%;
  padding: 22px;
  background: var(--surface);
  color: var(--text);
}

.tool-header,
.registry-panel__heading,
.detail-header,
.section-heading,
.detail-footer {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.tool-header { margin-bottom: 18px; }
.tool-header h2,
.registry-panel h3,
.detail-panel h3,
.section-heading h4 {
  margin: 3px 0 0;
  color: var(--text);
  letter-spacing: -0.025em;
}
.tool-header h2 { font-size: 26px; }
.tool-header p { max-width: 680px; margin: 7px 0 0; color: var(--text-muted); font-size: 13px; line-height: 1.5; }
.tool-header__actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.tool-kicker,
.panel-kicker { color: var(--text-muted); font: 700 10px/1.2 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; }

.source-badge,
.count-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 24px;
  padding: 0 9px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-muted);
  font: 700 10px/1 var(--font-mono);
  white-space: nowrap;
}
.source-badge__dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.source-badge--online { border-color: #b9dfd3; background: var(--success-soft); color: var(--success); }
.source-badge--offline { border-color: #efc2c7; background: var(--danger-soft); color: var(--danger); }
.source-badge--loading { border-color: #c9d8f6; background: var(--accent-soft); color: var(--primary); }
.source-badge--reference { border-color: #ecd79c; background: var(--warning-soft); color: var(--warning); }
.source-badge--compact { min-height: 22px; }
.count-badge { min-width: 28px; justify-content: center; }

.tool-button {
  min-height: 34px;
  padding: 0 11px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  cursor: pointer;
  font-size: 12px;
  font-weight: 700;
}
.tool-button:hover { background: var(--surface-2); }
.tool-button:disabled { cursor: wait; opacity: .6; }

.tool-notice,
.tool-state {
  display: grid;
  gap: 5px;
  margin: 12px 0;
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--surface-2);
  color: var(--text-muted);
  font-size: 12px;
}
.tool-notice strong,
.tool-state strong { color: var(--text); }
.tool-notice--warning { border-color: #ecd79c; background: var(--warning-soft); color: var(--warning); }
.tool-notice--error,
.detail-state--error { border-color: #efc2c7; background: var(--danger-soft); color: var(--danger); }
.loading-bar { width: 180px; height: 3px; border-radius: 3px; background: var(--primary); animation: pulse 1.1s ease-in-out infinite alternate; }
@keyframes pulse { from { opacity: .35; transform: scaleX(.55); transform-origin: left; } to { opacity: 1; transform: scaleX(1); transform-origin: left; } }

.tool-layout { display: grid; grid-template-columns: minmax(220px, 280px) minmax(0, 1fr); gap: 14px; align-items: start; }
.registry-panel,
.detail-panel { border: 1px solid var(--border); border-radius: 8px; background: var(--surface); }
.registry-panel { padding: 14px; }
.registry-panel__heading { align-items: center; margin-bottom: 12px; }
.registry-panel h3 { font-size: 16px; }
.search-field { display: grid; gap: 5px; margin-bottom: 10px; color: var(--text-muted); font-size: 11px; font-weight: 700; }
.search-field input,
.circuit-select { min-height: 34px; padding: 0 9px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--text); }
.registry-list { display: grid; gap: 6px; }
.registry-item { display: grid; gap: 3px; width: 100%; padding: 9px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--text); text-align: left; cursor: pointer; }
.registry-item:hover { border-color: var(--primary); background: var(--accent-soft); }
.registry-item--selected { border-color: var(--primary); box-shadow: 0 0 0 2px var(--accent-soft); }
.registry-item__name { font-size: 12px; font-weight: 750; }
.registry-item__meta,
.registry-item__id { color: var(--text-muted); font: 10px/1.3 var(--font-mono); }
.registry-empty,
.empty-inline { margin: 6px 0; color: var(--text-muted); font-size: 12px; }

.detail-panel { min-height: 360px; padding: 18px; }
.detail-header { align-items: flex-start; }
.detail-header h3 { font-size: 20px; }
.detail-header code,
.detail-footer code { color: var(--text-muted); font: 10px/1.3 var(--font-mono); }
.detail-description { margin: 16px 0; color: var(--text-dim); font-size: 13px; line-height: 1.55; }
.metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 0 0 20px; }
.metric-grid > div { padding: 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); }
.metric-grid dt { color: var(--text-muted); font: 10px/1.2 var(--font-mono); text-transform: uppercase; }
.metric-grid dd { margin: 5px 0 0; color: var(--text); font-size: 16px; font-weight: 750; }
.member-section { margin-top: 18px; }
.section-heading { align-items: center; margin-bottom: 8px; }
.section-heading h4 { font-size: 14px; }
.member-list { display: flex; flex-wrap: wrap; gap: 6px; }
.member-chip { display: inline-flex; align-items: center; gap: 6px; padding: 6px 8px; border: 1px solid var(--border); border-radius: 5px; background: var(--surface-2); color: var(--text); font-size: 11px; }
.member-chip strong { font-weight: 700; }
.member-chip small { color: var(--text-muted); font: 10px var(--font-mono); }
.detail-state { display: grid; min-height: 280px; place-items: center; gap: 5px; color: var(--text-muted); text-align: center; font-size: 12px; }
.detail-state strong { color: var(--text); }
.detail-footer { margin-top: 22px; padding-top: 12px; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 10px; }

@media (max-width: 760px) {
  .circuit-tool { padding: 14px; }
  .tool-header { flex-direction: column; }
  .tool-header__actions { justify-content: flex-start; }
  .tool-layout { grid-template-columns: 1fr; }
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .detail-footer { flex-direction: column; }
}
</style>
