<template>
  <section class="knowledge-tool" aria-labelledby="knowledge-graph-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">Explore / Connected evidence</p>
        <h1 id="knowledge-graph-title">Knowledge graph</h1>
        <p class="tool-intro">
          Browse graph payloads returned by Society publications. The runtime status endpoint reports the graph
          service, while the graph itself is read from run evidence; no reference graph is drawn as if it were live.
        </p>
      </div>
      <div class="tool-header__actions">
        <span class="source-pill" :class="`source-pill--${sourceTone}`" role="status">
          <span class="source-dot" aria-hidden="true" />
          {{ sourceLabel }}
        </span>
        <button class="tool-button" type="button" :disabled="loading" @click="loadGraph">
          <span aria-hidden="true">↻</span>
          Refresh graph
        </button>
      </div>
    </header>

    <div class="source-strip" role="status" aria-live="polite">
      <span class="source-strip__label">Graph sources</span>
      <code>GET /api/knowledge-graph</code>
      <code>result.publication.evidence_graph</code>
      <span class="source-strip__detail">The first endpoint reports service status; the second supplies graph data.</span>
    </div>

    <div v-if="errorMessage" class="tool-notice" :class="`tool-notice--${offline ? 'warning' : 'error'}`" role="alert">
      <strong>{{ offline ? 'Graph sources offline' : 'Graph data partially unavailable' }}</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <div class="graph-summary-grid">
      <article class="service-card" :class="`service-card--${serviceTone}`">
        <div class="service-card__header"><span class="service-card__icon" aria-hidden="true">◇</span><span>{{ serviceState }}</span></div>
        <h2>Graph service</h2>
        <p>{{ serviceDetail }}</p>
        <code>GET /api/knowledge-graph</code>
        <div v-if="stores.length" class="store-list"><span v-for="store in stores" :key="store" class="store-tag">{{ store }}</span></div>
      </article>
      <article class="service-card service-card--neutral">
        <div class="service-card__header"><span class="service-card__icon" aria-hidden="true">↗</span><span>Run-derived</span></div>
        <h2>Evidence payloads</h2>
        <p>Only graphs attached to persisted Society publications are available to browse.</p>
        <code>publication.evidence_graph</code>
        <div class="store-list"><span class="store-tag">{{ runs.length }} runs inspected</span></div>
      </article>
      <article class="service-card service-card--warning">
        <div class="service-card__header"><span class="service-card__icon" aria-hidden="true">!</span><span>Explicit boundary</span></div>
        <h2>No inferred links</h2>
        <p>Missing nodes, edges, or types remain unavailable instead of being filled with illustrative content.</p>
        <code>Provenance preserved</code>
      </article>
    </div>

    <section class="tool-panel selector-panel" aria-labelledby="graph-run-selector-title">
      <div>
        <p class="section-kicker">Select an evidence source</p>
        <h2 id="graph-run-selector-title">Society run</h2>
      </div>
      <div class="selector-control">
        <label for="graph-run-select">Research Society run</label>
        <select id="graph-run-select" v-model="selectedRunId" :disabled="!runs.length || loading" @change="loadSelectedRun">
          <option value="" disabled>{{ runs.length ? 'Select a run' : 'No runs returned' }}</option>
          <option v-for="run in runs" :key="run.id" :value="run.id">{{ run.goal || run.id }} · {{ run.status || 'status unavailable' }}</option>
        </select>
      </div>
    </section>

    <div class="graph-workspace">
      <section class="tool-panel canvas-panel" aria-labelledby="graph-canvas-title" :aria-busy="loading">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Returned node set</p>
            <h2 id="graph-canvas-title">Graph canvas</h2>
          </div>
          <span class="count-label">{{ filteredNodes.length }} / {{ nodes.length }} nodes</span>
        </header>
        <div v-if="loading" class="state-block state-block--loading" role="status" aria-live="polite">
          <span class="spinner" aria-hidden="true" />
          <span>Loading graph evidence…</span>
        </div>
        <div v-else-if="offline" class="state-block" role="alert">
          <span class="state-mark" aria-hidden="true">!</span>
          <strong>Graph data unavailable offline</strong>
          <span>Reconnect the runtime to request the selected run.</span>
        </div>
        <div v-else-if="!selectedRun" class="state-block">
          <span class="state-mark" aria-hidden="true">↖</span>
          <strong>Select a run</strong>
          <span>Choose a run whose publication includes an evidence graph.</span>
        </div>
        <div v-else-if="!nodes.length" class="state-block">
          <span class="state-mark" aria-hidden="true">∅</span>
          <strong>Graph payload empty</strong>
          <span>The selected run returned no evidence graph nodes. This is an empty source, not a blank graph to fill.</span>
        </div>
        <template v-else>
          <div class="canvas-toolbar">
            <label class="search-label" for="graph-search">Search nodes</label>
            <input id="graph-search" v-model="search" class="tool-input" type="search" placeholder="Label, type, or ID" />
            <label class="search-label" for="graph-type-filter">Type</label>
            <select id="graph-type-filter" v-model="typeFilter" class="tool-select">
              <option value="all">All returned types</option>
              <option v-for="type in nodeTypes" :key="type" :value="type">{{ type }}</option>
            </select>
          </div>
          <div class="node-board" aria-label="Evidence graph nodes">
            <button v-for="node in filteredNodes" :key="node.id" class="graph-node" :class="{ 'graph-node--selected': node.id === selectedNodeId }" type="button" :aria-pressed="node.id === selectedNodeId" @click="selectedNodeId = node.id">
              <span class="graph-node__type">{{ node.type || 'Untyped' }}</span>
              <strong>{{ node.label || node.id }}</strong>
              <code>{{ node.id }}</code>
            </button>
          </div>
          <p v-if="!filteredNodes.length" class="inline-state">No nodes match the current filters.</p>
        </template>
      </section>

      <section class="tool-panel inspector-panel" aria-labelledby="graph-inspector-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Node and edge context</p>
            <h2 id="graph-inspector-title">{{ selectedNode ? selectedNode.label : 'Inspector' }}</h2>
          </div>
        </header>
        <div v-if="!selectedNode" class="state-block state-block--compact">
          <span class="state-mark" aria-hidden="true">↖</span>
          <strong>Select a node</strong>
          <span>Inspect its returned metadata and connected edges.</span>
        </div>
        <div v-else class="inspector-body">
          <dl class="node-meta">
            <div><dt>ID</dt><dd><code>{{ selectedNode.id }}</code></dd></div>
            <div><dt>Type</dt><dd>{{ selectedNode.type || 'Unavailable' }}</dd></div>
            <div><dt>Created</dt><dd>{{ formatDate(selectedNode.createdAt) }}</dd></div>
          </dl>
          <div v-if="selectedNode.properties" class="property-block">
            <h3>Returned properties</h3>
            <pre>{{ pretty(selectedNode.properties) }}</pre>
          </div>
          <div class="edge-block">
            <h3>Connected edges</h3>
            <ul v-if="connectedEdges.length" class="edge-list">
              <li v-for="edge in connectedEdges" :key="edge.id"><span>{{ edge.sourceId }} → {{ edge.targetId }}</span><strong>{{ edge.relationship || 'Unavailable' }}</strong></li>
            </ul>
            <p v-else class="inline-state">No returned edges connect this node.</p>
          </div>
        </div>
      </section>
    </div>

    <section class="tool-panel edge-panel" aria-labelledby="graph-edges-title">
      <header class="panel-header">
        <div><p class="section-kicker">Returned relationship set</p><h2 id="graph-edges-title">Graph edges</h2></div>
        <span class="count-label">{{ edges.length }} returned</span>
      </header>
      <div v-if="edges.length" class="table-wrap">
        <table class="edge-table">
          <caption class="desktop-sr-only">Knowledge graph edges returned by the selected run</caption>
          <thead><tr><th scope="col">Source</th><th scope="col">Relation</th><th scope="col">Target</th><th scope="col">Weight</th></tr></thead>
          <tbody><tr v-for="edge in edges" :key="edge.id"><td><code>{{ edge.sourceId }}</code></td><td>{{ edge.relationship || 'Unavailable' }}</td><td><code>{{ edge.targetId }}</code></td><td>{{ edge.weight ?? 'Not returned' }}</td></tr></tbody>
        </table>
      </div>
      <div v-else class="inline-state">No graph edges are available for this run.</div>
    </section>

    <footer class="tool-footer">
      <span>Contract note</span>
      <span><code>GET /api/knowledge-graph</code> is a service-status contract, not a node/edge payload.</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { apiUrl } from '../services/api';

type JsonRecord = Record<string, unknown>;
type ServiceTone = 'online' | 'offline' | 'empty';
type GraphNode = { id: string; label: string; type: string; createdAt: string; properties?: JsonRecord };
type GraphEdge = { id: string; sourceId: string; targetId: string; relationship: string; weight?: string };
type RunSummary = { id: string; goal: string; status: string; created: string; };
type RunDetail = RunSummary & { graph: { nodes: GraphNode[]; edges: GraphEdge[] } | null };

const API_BASE = apiUrl('/api');
const serviceState = ref('Unknown');
const serviceDetail = ref('The graph service has not been queried.');
const stores = ref<string[]>([]);
const runs = ref<RunSummary[]>([]);
const selectedRunId = ref('');
const selectedRun = ref<RunDetail | null>(null);
const loading = ref(true);
const offline = ref(false);
const errorMessage = ref('');
const search = ref('');
const typeFilter = ref('all');
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

const serviceTone = computed<ServiceTone>(() => offline.value ? 'offline' : serviceState.value === 'Active' ? 'online' : 'empty');
const sourceTone = computed(() => loading.value ? 'loading' : offline.value ? 'offline' : errorMessage.value ? 'partial' : 'online');
const sourceLabel = computed(() => loading.value ? 'Loading graph sources' : offline.value ? 'Offline / unavailable' : errorMessage.value ? 'Partially available' : 'Graph sources reachable');
const nodes = computed(() => selectedRun.value?.graph?.nodes ?? []);
const edges = computed(() => selectedRun.value?.graph?.edges ?? []);
const nodeTypes = computed(() => [...new Set(nodes.value.map((node) => node.type || 'Untyped'))].sort());
const filteredNodes = computed(() => {
  const query = search.value.trim().toLowerCase();
  return nodes.value.filter((node) => {
    const matchesQuery = !query || `${node.label} ${node.type} ${node.id}`.toLowerCase().includes(query);
    const matchesType = typeFilter.value === 'all' || (node.type || 'Untyped') === typeFilter.value;
    return matchesQuery && matchesType;
  });
});
const selectedNode = computed(() => nodes.value.find((node) => node.id === selectedNodeId.value) ?? null);
const connectedEdges = computed(() => edges.value.filter((edge) => edge.sourceId === selectedNodeId.value || edge.targetId === selectedNodeId.value));

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
    errorMessage.value = `Could not load graph for ${id}: ${messageOf(error)}`;
    offline.value = isOffline(errorMessage.value);
  }
}

async function loadGraph(): Promise<void> {
  loading.value = true;
  offline.value = false;
  errorMessage.value = '';
  const [serviceResult, runResult] = await Promise.allSettled([request<unknown>('/knowledge-graph'), request<unknown>('/society/runs')]);
  if (serviceResult.status === 'fulfilled') {
    const record = isRecord(serviceResult.value) ? serviceResult.value : {};
    serviceState.value = text(record.status, 'Response received');
    serviceDetail.value = serviceState.value.toLowerCase() === 'active' ? 'The backend reports the graph service as active.' : 'The backend returned a service response without an active status.';
    stores.value = Array.isArray(record.stores) ? record.stores.map((store) => text(store)).filter(Boolean) : [];
  } else {
    serviceState.value = 'Unavailable';
    serviceDetail.value = messageOf(serviceResult.reason);
    errorMessage.value = `Graph service status unavailable: ${messageOf(serviceResult.reason)}`;
    offline.value = isOffline(errorMessage.value);
  }
  if (runResult.status === 'fulfilled') {
    const values = Array.isArray(runResult.value) ? runResult.value : isRecord(runResult.value) && Array.isArray(runResult.value.runs) ? runResult.value.runs : [];
    runs.value = values.map(normalizeRun).filter((run): run is RunSummary => Boolean(run));
  } else {
    runs.value = [];
    errorMessage.value = [errorMessage.value, `Run graph source unavailable: ${messageOf(runResult.reason)}`].filter(Boolean).join(' · ');
    offline.value = offline.value || isOffline(messageOf(runResult.reason));
  }
  if (!runs.value.some((run) => run.id === selectedRunId.value)) selectedRunId.value = runs.value[0]?.id ?? '';
  if (selectedRunId.value) await loadSelectedRun();
  loading.value = false;
}

onMounted(() => { void loadGraph(); });
</script>

<style scoped>
.knowledge-tool { display: flex; width: 100%; min-height: 100%; flex-direction: column; gap: 14px; padding: 22px; color: var(--text); font-size: 13px; }
.tool-header, .tool-header__actions, .panel-header, .tool-footer, .selector-panel, .service-card__header, .canvas-toolbar, .edge-list li { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.tool-header { align-items: flex-start; }
.tool-header__copy { min-width: 0; }
.tool-eyebrow, .section-kicker, .source-strip__label, .count-label, .tool-footer > span:first-child, .service-card__header > span:last-child { color: var(--text-muted); font: 700 10px/1.2 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; }
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
.source-strip code, .tool-footer code, .service-card code, .graph-node code, .node-meta code, .edge-table code, .property-block pre { font-family: var(--font-mono); }
.source-strip__detail { margin-left: auto; }
.tool-notice { display: flex; flex-direction: column; gap: 3px; border: 1px solid var(--border); border-left: 3px solid var(--danger); border-radius: 6px; background: var(--danger-soft); padding: 10px 12px; color: var(--danger); line-height: 1.45; }
.tool-notice--warning { border-left-color: var(--warning); background: var(--warning-soft); color: var(--warning); }
.graph-summary-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.service-card { display: flex; min-height: 150px; flex-direction: column; gap: 7px; border: 1px solid var(--border); border-top: 3px solid var(--border-light); border-radius: 8px; background: var(--surface); padding: 12px; box-shadow: var(--shadow-sm); }
.service-card--online { border-top-color: var(--success); }
.service-card--offline { border-top-color: var(--danger); }
.service-card--warning { border-top-color: var(--warning); }
.service-card__header { color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); text-transform: uppercase; }
.service-card__icon { color: var(--text-dim); font: 16px/1 var(--font-mono); }
.service-card h2 { margin: 0; color: var(--text); font-size: 13px; font-weight: 680; }
.service-card p { flex: 1; margin: 0; color: var(--text-muted); font-size: 10px; line-height: 1.5; }
.service-card code { color: var(--text-muted); font-size: 9px; }
.store-list { display: flex; flex-wrap: wrap; gap: 5px; }
.store-tag { border: 1px solid var(--border); border-radius: 999px; background: var(--surface-2); color: var(--text-muted); padding: 3px 7px; font: 9px/1.2 var(--font-mono); }
.tool-panel { min-width: 0; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); padding: 14px; }
.selector-panel { display: grid; grid-template-columns: 1fr minmax(260px, 420px); align-items: end; }
.selector-panel h2, .panel-header h2 { margin: 0; color: var(--text); font-size: 14px; font-weight: 680; }
.selector-control { display: flex; flex-direction: column; gap: 5px; }
.selector-control label, .search-label { color: var(--text-dim); font-size: 10px; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; }
.selector-control select, .tool-input, .tool-select { width: 100%; min-height: 36px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--text); padding: 0 9px; font: inherit; font-size: 11px; }
.selector-control select:focus, .tool-input:focus, .tool-select:focus { border-color: var(--primary); box-shadow: 0 0 0 2px var(--accent-soft); outline: none; }
.graph-workspace { display: grid; grid-template-columns: minmax(400px, 1.2fr) minmax(300px, .8fr); gap: 14px; min-height: 440px; }
.panel-header { min-height: 34px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.count-label { font-size: 9px; white-space: nowrap; }
.state-block { display: flex; min-height: 220px; flex-direction: column; align-items: center; justify-content: center; gap: 7px; padding: 22px; color: var(--text-muted); text-align: center; line-height: 1.5; }
.state-block--compact { min-height: 180px; }
.state-block--loading { min-height: 180px; }
.state-block strong { color: var(--text); font-size: 13px; }
.state-mark { display: grid; width: 32px; height: 32px; place-items: center; border: 1px solid var(--border-light); border-radius: 50%; color: var(--text-muted); font: 700 15px/1 var(--font-mono); }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border-light); border-top-color: var(--primary); border-radius: 50%; animation: spin .75s linear infinite; }
.canvas-toolbar { align-items: center; margin-bottom: 10px; }
.canvas-toolbar .tool-input { max-width: 240px; }
.canvas-toolbar .tool-select { max-width: 170px; }
.node-board { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; max-height: 370px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 10px; }
.graph-node { display: flex; min-width: 0; flex-direction: column; align-items: flex-start; gap: 4px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--text); cursor: pointer; padding: 9px; text-align: left; }
.graph-node:hover { border-color: var(--border-light); }
.graph-node--selected { border-color: var(--primary); box-shadow: 0 0 0 1px var(--primary); }
.graph-node__type { color: var(--text-muted); font: 9px/1.2 var(--font-mono); text-transform: uppercase; }
.graph-node strong { width: 100%; overflow: hidden; font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
.graph-node code { width: 100%; overflow: hidden; color: var(--text-muted); font-size: 9px; text-overflow: ellipsis; white-space: nowrap; }
.inspector-body { display: flex; flex-direction: column; gap: 13px; }
.node-meta { display: grid; gap: 7px; margin: 0; }
.node-meta > div, .property-block { border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 8px 10px; }
.node-meta dt, .property-block h3, .edge-block h3 { color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); letter-spacing: .06em; text-transform: uppercase; }
.node-meta dd { margin: 4px 0 0; overflow-wrap: anywhere; color: var(--text); font-size: 11px; }
.property-block h3, .edge-block h3 { margin: 0 0 7px; }
.property-block pre { max-height: 180px; margin: 0; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; color: var(--text-dim); font-size: 9px; line-height: 1.5; }
.edge-list { display: flex; flex-direction: column; gap: 6px; margin: 0; padding: 0; list-style: none; }
.edge-list li { display: flex; flex-direction: column; align-items: flex-start; gap: 2px; border-bottom: 1px solid var(--border); padding: 6px 0; color: var(--text-dim); font: 9px/1.4 var(--font-mono); }
.edge-list li:last-child { border-bottom: 0; }
.edge-list strong { color: var(--text-muted); font: 9px/1.3 var(--font); }
.inline-state { color: var(--text-muted); font-size: 11px; line-height: 1.5; padding: 10px 0; }
.edge-panel { display: flex; flex-direction: column; gap: 2px; }
.table-wrap { overflow-x: auto; }
.edge-table { width: 100%; border-collapse: collapse; font-size: 10px; }
.edge-table th { border-bottom: 1px solid var(--border); color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); letter-spacing: .06em; padding: 7px 8px; text-align: left; text-transform: uppercase; }
.edge-table td { border-bottom: 1px solid var(--border); padding: 8px; vertical-align: top; }
.edge-table tr:last-child td { border-bottom: 0; }
.edge-table td:nth-child(2) { color: var(--text-dim); }
.edge-table td:last-child { color: var(--text-muted); font-family: var(--font-mono); }
.desktop-sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; margin: -1px; padding: 0; border: 0; clip: rect(0, 0, 0, 0); white-space: nowrap; }
.tool-footer { justify-content: flex-start; flex-wrap: wrap; color: var(--text-muted); font-size: 10px; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 50% { opacity: .35; } }
@media (max-width: 1000px) { .graph-workspace { grid-template-columns: 1fr; } }
@media (max-width: 700px) { .knowledge-tool { padding: 14px; } .tool-header { flex-direction: column; } .tool-header__actions { width: 100%; justify-content: flex-start; } .source-strip__detail { width: 100%; margin-left: 0; } .graph-summary-grid { grid-template-columns: 1fr; } .selector-panel { grid-template-columns: 1fr; align-items: stretch; } .canvas-toolbar { align-items: stretch; flex-direction: column; } .canvas-toolbar .tool-input, .canvas-toolbar .tool-select { max-width: none; } .node-board { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 440px) { .node-board { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { .spinner, .source-pill--loading .source-dot { animation: none; } }
</style>
