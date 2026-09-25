<template>
  <section class="fusion-tool" aria-labelledby="evidence-fusion-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">Research / Provenance</p>
        <h1 id="evidence-fusion-title">Evidence fusion</h1>
        <p class="tool-intro">
          Inspect the evidence graph emitted with a Society publication. There is no standalone evidence-fusion
          endpoint in the current runtime, so this view never creates a fused score or relationship on the client.
        </p>
      </div>
      <div class="tool-header__actions">
        <span class="source-pill" :class="`source-pill--${sourceTone}`" role="status">
          <span class="source-dot" aria-hidden="true" />
          {{ sourceLabel }}
        </span>
        <button class="tool-button" type="button" :disabled="loading" @click="loadEvidence">
          <span aria-hidden="true">↻</span>
          Refresh evidence
        </button>
      </div>
    </header>

    <div class="source-strip" role="status" aria-live="polite">
      <span class="source-strip__label">Evidence source</span>
      <code>result.publication.evidence_graph</code>
      <span class="source-strip__detail">Read from a selected Society run; no synthetic graph is shown.</span>
    </div>

    <div v-if="errorMessage" class="tool-notice" :class="`tool-notice--${offline ? 'warning' : 'error'}`" role="alert">
      <strong>{{ offline ? 'Evidence source offline' : 'Evidence source unavailable' }}</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <section class="metric-strip" aria-label="Evidence graph observations">
      <div class="metric-tile">
        <span class="metric-tile__value">{{ graphNodes.length }}</span>
        <span class="metric-tile__label">Graph nodes returned</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ graphEdges.length }}</span>
        <span class="metric-tile__label">Relationships returned</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ runs.length }}</span>
        <span class="metric-tile__label">Runs available</span>
      </div>
      <div class="metric-tile metric-tile--note">
        <span class="metric-tile__label">Fusion boundary</span>
        <span class="metric-tile__caption">No client-side ranking, confidence, or relationship inference is performed.</span>
      </div>
    </section>

    <section class="tool-panel selector-panel" aria-labelledby="evidence-selector-title">
      <div>
        <p class="section-kicker">Select a publication record</p>
        <h2 id="evidence-selector-title">Evidence graph run</h2>
      </div>
      <div class="selector-control">
        <label for="evidence-run-select">Research Society run</label>
        <select id="evidence-run-select" v-model="selectedRunId" :disabled="!runs.length || loading" @change="loadSelectedRun">
          <option value="" disabled>{{ runs.length ? 'Select a run' : 'No runs returned' }}</option>
          <option v-for="run in runs" :key="run.id" :value="run.id">{{ run.goal || run.id }} · {{ run.status || 'status unavailable' }}</option>
        </select>
      </div>
    </section>

    <div class="evidence-layout">
      <section class="tool-panel graph-panel" aria-labelledby="graph-title" :aria-busy="loading">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Returned graph structure</p>
            <h2 id="graph-title">Evidence nodes</h2>
          </div>
          <span class="count-label">{{ filteredNodes.length }} shown</span>
        </header>
        <div v-if="loading" class="state-block state-block--loading" role="status" aria-live="polite">
          <span class="spinner" aria-hidden="true" />
          <span>Reading the selected run and evidence payload…</span>
        </div>
        <div v-else-if="offline" class="state-block" role="alert">
          <span class="state-mark" aria-hidden="true">!</span>
          <strong>Evidence unavailable offline</strong>
          <span>Reconnect the runtime to request the selected run record.</span>
        </div>
        <div v-else-if="!selectedRun" class="state-block">
          <span class="state-mark" aria-hidden="true">↖</span>
          <strong>Select a run</strong>
          <span>Choose a run to inspect its returned evidence graph.</span>
        </div>
        <div v-else-if="!graphNodes.length" class="state-block">
          <span class="state-mark" aria-hidden="true">∅</span>
          <strong>No evidence graph returned</strong>
          <span>The selected run has no readable <code>evidence_graph</code> payload. This is not a zero-evidence score.</span>
        </div>
        <template v-else>
          <div class="graph-toolbar">
            <label class="search-label" for="evidence-node-search">Filter nodes</label>
            <input id="evidence-node-search" v-model="nodeQuery" class="tool-input" type="search" placeholder="Label, type, or ID" />
          </div>
          <div v-if="filteredNodes.length" class="node-grid">
            <button
              v-for="node in filteredNodes"
              :key="node.id"
              class="node-card"
              :class="{ 'node-card--selected': selectedNodeId === node.id }"
              type="button"
              :aria-pressed="selectedNodeId === node.id"
              @click="selectedNodeId = node.id"
            >
              <span class="node-card__type">{{ node.type || 'Untyped' }}</span>
              <strong>{{ node.label || node.id }}</strong>
              <code>{{ node.id }}</code>
            </button>
          </div>
          <div v-else class="inline-state">No nodes match the current filter.</div>
        </template>
      </section>

      <section class="tool-panel inspector-panel" aria-labelledby="evidence-inspector-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Relationship inspection</p>
            <h2 id="evidence-inspector-title">{{ selectedNode ? selectedNode.label : 'Node inspector' }}</h2>
          </div>
        </header>
        <div v-if="!selectedNode" class="state-block state-block--compact">
          <span class="state-mark" aria-hidden="true">↖</span>
          <strong>Select a node</strong>
          <span>Inspect returned metadata and relationships without adding inferred values.</span>
        </div>
        <div v-else class="inspector-content">
          <dl class="node-meta">
            <div><dt>Node ID</dt><dd><code>{{ selectedNode.id }}</code></dd></div>
            <div><dt>Type</dt><dd>{{ selectedNode.type || 'Unavailable' }}</dd></div>
            <div><dt>Created</dt><dd>{{ formatDate(selectedNode.createdAt) }}</dd></div>
          </dl>
          <div v-if="selectedNode.properties" class="metadata-block">
            <h3>Returned properties</h3>
            <pre>{{ pretty(selectedNode.properties) }}</pre>
          </div>
          <div class="relationship-block">
            <h3>Relationships touching node</h3>
            <ul v-if="relatedEdges.length" class="relationship-list">
              <li v-for="edge in relatedEdges" :key="edge.id">
                <span class="relationship-direction">{{ edge.sourceId === selectedNode.id ? 'OUT' : 'IN' }}</span>
                <span>{{ edge.sourceId }} <span aria-hidden="true">→</span> {{ edge.targetId }}</span>
                <strong>{{ edge.relationship || 'relationship unavailable' }}</strong>
              </li>
            </ul>
            <p v-else class="inline-state">No returned relationship touches this node.</p>
          </div>
        </div>
      </section>
    </div>

    <section class="tool-panel relationships-panel" aria-labelledby="relationships-title">
      <header class="panel-header">
        <div>
          <p class="section-kicker">Raw relationship payload</p>
          <h2 id="relationships-title">Relationships</h2>
        </div>
        <span class="count-label">{{ graphEdges.length }} returned</span>
      </header>
      <div v-if="graphEdges.length" class="relationship-table-wrap">
        <table class="relationship-table">
          <caption class="desktop-sr-only">Evidence graph relationships returned by the backend</caption>
          <thead><tr><th scope="col">Source</th><th scope="col">Relationship</th><th scope="col">Target</th><th scope="col">Weight</th></tr></thead>
          <tbody>
            <tr v-for="edge in graphEdges" :key="edge.id">
              <td><code>{{ edge.sourceId }}</code></td>
              <td>{{ edge.relationship || 'Unavailable' }}</td>
              <td><code>{{ edge.targetId }}</code></td>
              <td>{{ edge.weight ?? 'Not returned' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-else class="inline-state">No relationships are available for this run.</div>
    </section>

    <footer class="tool-footer">
      <span>Unavailable by design</span>
      <span>A missing graph is shown as unavailable, never replaced with an illustrative graph.</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { apiUrl } from '../services/api';

type JsonRecord = Record<string, unknown>;
type SourceTone = 'online' | 'offline' | 'loading' | 'partial';

interface RunSummary { id: string; goal: string; status: string; created: string; }
interface GraphNode { id: string; label: string; type: string; createdAt: string; properties?: JsonRecord; }
interface GraphEdge { id: string; sourceId: string; targetId: string; relationship: string; weight?: string; }
interface RunDetail extends RunSummary { graph: { nodes: GraphNode[]; edges: GraphEdge[] } | null; }

const API_BASE = apiUrl('/api');
const runs = ref<RunSummary[]>([]);
const selectedRunId = ref('');
const selectedRun = ref<RunDetail | null>(null);
const loading = ref(true);
const offline = ref(false);
const errorMessage = ref('');
const nodeQuery = ref('');
const selectedNodeId = ref('');
let version = 0;

function isRecord(value: unknown): value is JsonRecord { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function text(value: unknown, fallback = ''): string { return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? String(value) : fallback; }
function messageOf(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function isOffline(message: string): boolean { return /offline|failed to fetch|network|load|connection|timeout/i.test(message); }
async function request<T>(path: string): Promise<T> {
  if (typeof navigator !== 'undefined' && navigator.onLine === false) throw new Error('Browser is offline');
  const response = await fetch(`${API_BASE}${path}`, { headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error(`GET ${path} returned ${response.status}`);
  return response.json() as Promise<T>;
}
function normalizeRun(value: unknown): RunSummary | null {
  if (!isRecord(value)) return null;
  const id = text(value.run_id) || text(value.id);
  if (!id) return null;
  return { id, goal: text(value.goal), status: text(value.status), created: text(value.created) || text(value.published_at) };
}
function normalizeNode(value: unknown): GraphNode | null {
  if (!isRecord(value)) return null;
  const id = text(value.id) || text(value.node_id);
  if (!id) return null;
  const properties = isRecord(value.properties) ? value.properties : isRecord(value.metadata) ? value.metadata : undefined;
  return { id, label: text(value.label) || id, type: text(value.node_type) || text(value.type), createdAt: text(value.created_at) || text(value.createdAt), ...(properties ? { properties } : {}) };
}
function normalizeEdge(value: unknown, index: number): GraphEdge | null {
  if (!isRecord(value)) return null;
  const sourceId = text(value.source_id) || text(value.source);
  const targetId = text(value.target_id) || text(value.target);
  if (!sourceId || !targetId) return null;
  return { id: text(value.edge_id) || `${sourceId}-${targetId}-${index}`, sourceId, targetId, relationship: text(value.relationship) || text(value.edge_type), weight: value.weight === undefined ? undefined : text(value.weight) };
}
function normalizeDetail(value: unknown, fallback: RunSummary): RunDetail {
  const record = isRecord(value) ? value : {};
  const result = isRecord(record.result) ? record.result : record;
  const publication = isRecord(result.publication) ? result.publication : isRecord(record.publication) ? record.publication : null;
  const rawGraph = publication?.evidence_graph ?? result.evidence_graph ?? record.evidence_graph;
  const graphRecord = isRecord(rawGraph) ? rawGraph : null;
  const rawResult = isRecord(graphRecord?.result) ? graphRecord.result : graphRecord;
  const nodes = Array.isArray(rawResult?.nodes) ? rawResult.nodes.map((value) => normalizeNode(value)).filter((node): node is GraphNode => Boolean(node)) : [];
  const edges = Array.isArray(rawResult?.edges) ? rawResult.edges.map(normalizeEdge).filter((edge): edge is GraphEdge => Boolean(edge)) : [];
  return { ...fallback, status: text(record.status, text(result.status, fallback.status)), graph: graphRecord ? { nodes, edges } : null };
}
function formatDate(value: string): string { if (!value) return 'Unavailable'; const date = new Date(value); return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }); }
function pretty(value: JsonRecord): string { try { return JSON.stringify(value, null, 2); } catch { return 'Properties could not be serialized.'; } }

const graphNodes = computed(() => selectedRun.value?.graph?.nodes ?? []);
const graphEdges = computed(() => selectedRun.value?.graph?.edges ?? []);
const filteredNodes = computed(() => {
  const query = nodeQuery.value.trim().toLowerCase();
  if (!query) return graphNodes.value;
  return graphNodes.value.filter((node) => `${node.label} ${node.type} ${node.id}`.toLowerCase().includes(query));
});
const selectedNode = computed(() => graphNodes.value.find((node) => node.id === selectedNodeId.value) ?? null);
const relatedEdges = computed(() => graphEdges.value.filter((edge) => edge.sourceId === selectedNodeId.value || edge.targetId === selectedNodeId.value));
const sourceTone = computed<SourceTone>(() => loading.value ? 'loading' : offline.value ? 'offline' : errorMessage.value ? 'partial' : 'online');
const sourceLabel = computed(() => sourceTone.value === 'loading' ? 'Loading evidence' : sourceTone.value === 'offline' ? 'Offline / unavailable' : sourceTone.value === 'partial' ? 'Partially available' : graphNodes.value.length ? 'Backend evidence available' : 'Run source reachable');

async function loadSelectedRun(): Promise<void> {
  const id = selectedRunId.value;
  if (!id) { selectedRun.value = null; return; }
  const requestVersion = ++version;
  errorMessage.value = '';
  const fallback = runs.value.find((run) => run.id === id) ?? { id, goal: '', status: '', created: '' };
  try {
    const raw = await request<unknown>(`/society/runs/${encodeURIComponent(id)}`);
    if (requestVersion !== version) return;
    selectedRun.value = normalizeDetail(raw, fallback);
    selectedNodeId.value = selectedRun.value.graph?.nodes[0]?.id ?? '';
  } catch (error) {
    if (requestVersion !== version) return;
    selectedRun.value = { ...fallback, graph: null };
    errorMessage.value = `Could not load evidence for ${id}: ${messageOf(error)}`;
    offline.value = isOffline(errorMessage.value);
  }
}

async function loadEvidence(): Promise<void> {
  loading.value = true;
  offline.value = false;
  errorMessage.value = '';
  try {
    const raw = await request<unknown>('/society/runs');
    const values = Array.isArray(raw) ? raw : isRecord(raw) && Array.isArray(raw.runs) ? raw.runs : [];
    runs.value = values.map(normalizeRun).filter((run): run is RunSummary => Boolean(run));
    if (!runs.value.some((run) => run.id === selectedRunId.value)) selectedRunId.value = runs.value[0]?.id ?? '';
    if (selectedRunId.value) await loadSelectedRun();
  } catch (error) {
    runs.value = [];
    selectedRunId.value = '';
    selectedRun.value = null;
    errorMessage.value = messageOf(error);
    offline.value = isOffline(errorMessage.value);
  } finally {
    loading.value = false;
  }
}

onMounted(() => { void loadEvidence(); });
</script>

<style scoped>
.fusion-tool { display: flex; width: 100%; min-height: 100%; flex-direction: column; gap: 14px; padding: 22px; color: var(--text); font-size: 13px; }
.tool-header, .tool-header__actions, .panel-header, .tool-footer, .selector-panel, .graph-toolbar, .node-card, .event-item__heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.tool-header { align-items: flex-start; }
.tool-header__copy { min-width: 0; }
.tool-eyebrow, .section-kicker, .source-strip__label, .metric-tile__label, .count-label, .tool-footer > span:first-child { color: var(--text-muted); font: 700 10px/1.2 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; }
.tool-eyebrow, .section-kicker { margin: 0 0 5px; }
.tool-header h1 { margin: 0; color: var(--text); font-size: clamp(22px, 3vw, 30px); font-weight: 720; letter-spacing: -.035em; line-height: 1.1; }
.tool-intro { max-width: 760px; margin: 8px 0 0; color: var(--text-muted); line-height: 1.55; }
.tool-header__actions { flex: 0 0 auto; flex-wrap: wrap; justify-content: flex-end; }
.source-pill { display: inline-flex; min-height: 25px; align-items: center; gap: 6px; border: 1px solid var(--border); border-radius: 999px; background: var(--surface-2); padding: 0 9px; white-space: nowrap; }
.source-pill--online { border-color: #b9dfd3; background: var(--success-soft); color: var(--success); }
.source-pill--partial { border-color: #ecd79c; background: var(--warning-soft); color: var(--warning); }
.source-pill--offline { border-color: #efc2c7; background: var(--danger-soft); color: var(--danger); }
.source-pill--loading { border-color: #cbd8ef; background: var(--accent-soft); color: var(--primary-focus); }
.source-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.source-pill--loading .source-dot { animation: pulse 1.2s ease-in-out infinite; }
.tool-button { display: inline-flex; min-height: 34px; align-items: center; justify-content: center; gap: 7px; border: 1px solid var(--border-light); border-radius: 6px; background: var(--surface); color: var(--text); cursor: pointer; font: 650 12px/1.2 var(--font); padding: 0 12px; white-space: nowrap; }
.tool-button:hover:not(:disabled) { border-color: var(--text); background: var(--surface-2); }
.tool-button:disabled { cursor: not-allowed; opacity: .45; }
.source-strip { display: flex; min-height: 34px; align-items: center; flex-wrap: wrap; gap: 8px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 7px 10px; color: var(--text-muted); font-size: 11px; }
.source-strip code, .tool-footer code, .node-card code, .node-meta code, .relationship-table code, .metadata-block pre { font-family: var(--font-mono); }
.source-strip__detail { margin-left: auto; }
.tool-notice { display: flex; flex-direction: column; gap: 3px; border: 1px solid var(--border); border-left: 3px solid var(--danger); border-radius: 6px; background: var(--danger-soft); padding: 10px 12px; color: var(--danger); line-height: 1.45; }
.tool-notice--warning { border-left-color: var(--warning); background: var(--warning-soft); color: var(--warning); }
.metric-strip { display: grid; grid-template-columns: repeat(3, minmax(100px, 1fr)) minmax(220px, 1.5fr); overflow: hidden; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); }
.metric-tile { min-height: 66px; border-right: 1px solid var(--border); padding: 12px 14px; }
.metric-tile:last-child { border-right: 0; }
.metric-tile__value { display: block; margin-bottom: 3px; color: var(--text); font: 700 21px/1 var(--font-mono); }
.metric-tile__label { display: block; color: var(--text-muted); font-size: 9px; }
.metric-tile__caption { display: block; margin-top: 4px; color: var(--text-muted); font-size: 10px; }
.metric-tile--note { display: flex; flex-direction: column; justify-content: center; background: var(--surface-2); }
.tool-panel { min-width: 0; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); padding: 14px; }
.selector-panel { display: grid; grid-template-columns: 1fr minmax(260px, 420px); align-items: end; }
.selector-panel h2, .panel-header h2 { margin: 0; color: var(--text); font-size: 14px; font-weight: 680; }
.selector-control { display: flex; flex-direction: column; gap: 5px; }
.selector-control label, .search-label { color: var(--text-dim); font-size: 10px; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; }
.selector-control select, .tool-input { width: 100%; min-height: 36px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--text); padding: 0 9px; font: inherit; font-size: 11px; }
.selector-control select:focus, .tool-input:focus { border-color: var(--primary); box-shadow: 0 0 0 2px var(--accent-soft); outline: none; }
.evidence-layout { display: grid; grid-template-columns: minmax(400px, 1.2fr) minmax(300px, .8fr); gap: 14px; min-height: 450px; }
.panel-header { min-height: 34px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.count-label { font-size: 9px; white-space: nowrap; }
.state-block { display: flex; min-height: 220px; flex-direction: column; align-items: center; justify-content: center; gap: 7px; padding: 22px; color: var(--text-muted); text-align: center; line-height: 1.5; }
.state-block strong { color: var(--text); font-size: 13px; }
.state-block code { font-family: var(--font-mono); font-size: 10px; }
.state-block--compact { min-height: 180px; }
.state-block--loading { min-height: 180px; }
.state-mark { display: grid; width: 32px; height: 32px; place-items: center; border: 1px solid var(--border-light); border-radius: 50%; color: var(--text-muted); font: 700 15px/1 var(--font-mono); }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border-light); border-top-color: var(--primary); border-radius: 50%; animation: spin .75s linear infinite; }
.graph-toolbar { align-items: center; margin-bottom: 10px; }
.graph-toolbar .tool-input { max-width: 260px; }
.node-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; max-height: 400px; overflow-y: auto; }
.node-card { min-width: 0; flex-direction: column; align-items: flex-start; justify-content: flex-start; gap: 4px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); color: var(--text); cursor: pointer; padding: 9px; text-align: left; }
.node-card:hover { border-color: var(--border-light); background: var(--surface); }
.node-card--selected { border-color: var(--primary); background: var(--accent-soft); box-shadow: 0 0 0 1px var(--primary); }
.node-card__type { color: var(--text-muted); font: 9px/1.2 var(--font-mono); text-transform: uppercase; }
.node-card strong { width: 100%; overflow: hidden; font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
.node-card code { width: 100%; overflow: hidden; color: var(--text-muted); font-size: 9px; text-overflow: ellipsis; white-space: nowrap; }
.inspector-content { display: flex; flex-direction: column; gap: 13px; }
.node-meta { display: grid; gap: 7px; margin: 0; }
.node-meta > div, .metadata-block { border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 8px 10px; }
.node-meta dt, .metadata-block h3, .relationship-block h3 { color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); letter-spacing: .06em; text-transform: uppercase; }
.node-meta dd { margin: 4px 0 0; overflow-wrap: anywhere; color: var(--text); font-size: 11px; }
.metadata-block h3, .relationship-block h3 { margin: 0 0 7px; }
.metadata-block pre { max-height: 190px; margin: 0; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; color: var(--text-dim); font-size: 9px; line-height: 1.5; }
.relationship-list { display: flex; flex-direction: column; gap: 6px; margin: 0; padding: 0; list-style: none; }
.relationship-list li { display: grid; grid-template-columns: 28px minmax(0, 1fr); gap: 5px 7px; border-bottom: 1px solid var(--border); padding: 6px 0; font-size: 10px; line-height: 1.4; }
.relationship-list li:last-child { border-bottom: 0; }
.relationship-direction { color: var(--text-muted); font: 9px/1.3 var(--font-mono); }
.relationship-list strong { grid-column: 2; color: var(--text-dim); font-size: 9px; font-weight: 600; }
.inline-state { color: var(--text-muted); font-size: 11px; line-height: 1.5; padding: 10px 0; }
.relationships-panel { display: flex; flex-direction: column; gap: 2px; }
.relationship-table-wrap { overflow-x: auto; }
.relationship-table { width: 100%; border-collapse: collapse; font-size: 10px; }
.relationship-table th { border-bottom: 1px solid var(--border); color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); letter-spacing: .06em; padding: 7px 8px; text-align: left; text-transform: uppercase; }
.relationship-table td { border-bottom: 1px solid var(--border); padding: 8px; vertical-align: top; }
.relationship-table tr:last-child td { border-bottom: 0; }
.relationship-table td:nth-child(2) { color: var(--text-dim); }
.relationship-table td:last-child { color: var(--text-muted); font-family: var(--font-mono); }
.desktop-sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; margin: -1px; padding: 0; border: 0; clip: rect(0, 0, 0, 0); white-space: nowrap; }
.tool-footer { justify-content: flex-start; flex-wrap: wrap; color: var(--text-muted); font-size: 10px; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 50% { opacity: .35; } }
@media (max-width: 1000px) { .evidence-layout { grid-template-columns: 1fr; } }
@media (max-width: 700px) { .fusion-tool { padding: 14px; } .tool-header { flex-direction: column; } .tool-header__actions { width: 100%; justify-content: flex-start; } .source-strip__detail { width: 100%; margin-left: 0; } .metric-strip { grid-template-columns: repeat(3, 1fr); } .metric-tile--note { grid-column: 1 / -1; border-top: 1px solid var(--border); } .selector-panel { grid-template-columns: 1fr; align-items: stretch; } .graph-toolbar { align-items: stretch; flex-direction: column; } .graph-toolbar .tool-input { max-width: none; } }
@media (max-width: 460px) { .node-grid { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { .spinner, .source-pill--loading .source-dot { animation: none; } }
</style>
