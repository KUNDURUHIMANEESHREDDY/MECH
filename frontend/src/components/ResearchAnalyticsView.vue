<template>
  <section class="analytics-tool" aria-labelledby="analytics-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">Research / Operational signals</p>
        <h1 id="analytics-title">Analytics</h1>
        <p class="tool-intro">
          Summarize what the desktop can actually observe: persisted Society runs, experiment records, and session
          records. This is descriptive inventory, not a scientific performance score.
        </p>
      </div>
      <div class="tool-header__actions">
        <span class="source-pill" :class="`source-pill--${sourceTone}`" role="status">
          <span class="source-dot" aria-hidden="true" />
          {{ sourceLabel }}
        </span>
        <button class="tool-button" type="button" :disabled="loading" @click="loadAnalytics">
          <span aria-hidden="true">↻</span>
          Refresh analytics
        </button>
      </div>
    </header>

    <div class="source-strip" role="status" aria-live="polite">
      <span class="source-strip__label">Observed sources</span>
      <code>/api/society/runs</code>
      <code>/api/experiments</code>
      <code>/api/sessions</code>
      <span class="source-strip__detail">No aggregate analytics endpoint is exposed.</span>
    </div>

    <div v-if="errorMessage" class="tool-notice" :class="`tool-notice--${offline ? 'warning' : 'error'}`" role="alert">
      <strong>{{ offline ? 'Analytics sources offline' : 'Some analytics sources are unavailable' }}</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <section class="metric-strip" aria-label="Observed record counts">
      <div class="metric-tile">
        <span class="metric-tile__value">{{ runs.length }}</span>
        <span class="metric-tile__label">Society runs</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ experiments.length }}</span>
        <span class="metric-tile__label">Experiments</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ sessions.length }}</span>
        <span class="metric-tile__label">Sessions</span>
      </div>
      <div class="metric-tile metric-tile--note">
        <span class="metric-tile__label">Metric policy</span>
        <span class="metric-tile__caption">No hypothesis confidence, effect size, or model quality values are synthesized here.</span>
      </div>
    </section>

    <div class="analytics-grid">
      <section class="tool-panel status-panel" aria-labelledby="status-distribution-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Run ledger</p>
            <h2 id="status-distribution-title">Observed run status</h2>
          </div>
          <span class="count-label">{{ runs.length }} records</span>
        </header>
        <div v-if="loading" class="state-block state-block--compact" role="status" aria-live="polite">
          <span class="spinner" aria-hidden="true" />
          <span>Reading local research records…</span>
        </div>
        <div v-else-if="offline" class="state-block state-block--compact" role="alert">
          <span class="state-mark" aria-hidden="true">!</span>
          <strong>Sources offline</strong>
          <span>Counts are unavailable until the runtime responds.</span>
        </div>
        <div v-else-if="!runs.length" class="state-block state-block--compact">
          <span class="state-mark" aria-hidden="true">∅</span>
          <strong>No run records</strong>
          <span>The run ledger returned no records to summarize.</span>
        </div>
        <div v-else class="status-list">
          <div v-for="item in statusCounts" :key="item.label" class="status-row">
            <span class="status-row__label">{{ item.label }}</span>
            <span class="status-row__bar" aria-hidden="true"><span :style="{ width: `${item.percent}%` }" /></span>
            <strong>{{ item.count }}</strong>
          </div>
          <p class="chart-note">Counts are grouped from returned <code>status</code> values; no scientific meaning is assigned.</p>
        </div>
      </section>

      <section class="tool-panel source-panel" aria-labelledby="source-coverage-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Connection coverage</p>
            <h2 id="source-coverage-title">Source status</h2>
          </div>
        </header>
        <ul class="source-list">
          <li v-for="source in sourceStates" :key="source.key">
            <span class="source-list__icon" :class="`source-list__icon--${source.tone}`" aria-hidden="true">{{ source.tone === 'online' ? '✓' : source.tone === 'offline' ? '!' : '—' }}</span>
            <span class="source-list__body"><strong>{{ source.label }}</strong><code>{{ source.path }}</code></span>
            <span class="source-list__value">{{ source.detail }}</span>
          </li>
        </ul>
        <p class="source-note">A missing endpoint or empty response is reported as such; it is never replaced with a placeholder metric.</p>
      </section>
    </div>

    <section class="tool-panel run-table-panel" aria-labelledby="run-table-title">
      <header class="panel-header">
        <div>
          <p class="section-kicker">Chronology</p>
          <h2 id="run-table-title">Recent Society runs</h2>
        </div>
        <label class="filter-label" for="analytics-status-filter">Filter status
          <select id="analytics-status-filter" v-model="statusFilter">
            <option value="all">All returned statuses</option>
            <option v-for="status in availableStatuses" :key="status" :value="status">{{ status }}</option>
          </select>
        </label>
      </header>
      <div v-if="!loading && filteredRuns.length" class="table-wrap">
        <table class="data-table">
          <caption class="desktop-sr-only">Persisted Society run records</caption>
          <thead><tr><th scope="col">Goal</th><th scope="col">Status</th><th scope="col">Steps</th><th scope="col">Created</th><th scope="col">Run ID</th></tr></thead>
          <tbody>
            <tr v-for="run in filteredRuns" :key="run.id">
              <td>{{ run.goal || 'Goal unavailable' }}</td>
              <td><span class="table-status" :class="`table-status--${run.tone}`">{{ run.status || 'Unavailable' }}</span></td>
              <td>{{ run.stepsCompleted || 'Not returned' }}</td>
              <td>{{ formatDate(run.created) }}</td>
              <td><code>{{ run.id }}</code></td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-else class="inline-state">{{ loading ? 'Loading run records…' : 'No run records match this filter.' }}</div>
    </section>

    <div class="record-grid-layout">
      <section class="tool-panel record-panel" aria-labelledby="experiment-records-title">
        <header class="panel-header">
          <div><p class="section-kicker">Local storage</p><h2 id="experiment-records-title">Experiment records</h2></div>
          <span class="count-label">{{ experiments.length }}</span>
        </header>
        <ul v-if="experiments.length" class="record-list">
          <li v-for="item in experiments.slice(0, 8)" :key="item.id">
            <span class="record-list__body"><strong>{{ item.name || item.id }}</strong><code>{{ item.id }}</code></span>
            <span class="record-list__value">{{ item.status || 'Status not returned' }}</span>
          </li>
        </ul>
        <p v-else class="inline-state">No experiment records returned.</p>
      </section>
      <section class="tool-panel record-panel" aria-labelledby="session-records-title">
        <header class="panel-header">
          <div><p class="section-kicker">Local storage</p><h2 id="session-records-title">Session records</h2></div>
          <span class="count-label">{{ sessions.length }}</span>
        </header>
        <ul v-if="sessions.length" class="record-list">
          <li v-for="item in sessions.slice(0, 8)" :key="item.id">
            <span class="record-list__body"><strong>{{ item.name || item.id }}</strong><code>{{ item.id }}</code></span>
            <span class="record-list__value">{{ item.status || 'Status not returned' }}</span>
          </li>
        </ul>
        <p v-else class="inline-state">No session records returned.</p>
      </section>
    </div>

    <footer class="tool-footer">
      <span>Scope</span>
      <span>Analytics describes available records and source availability; it does not infer research quality.</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { apiUrl } from '../services/api';

type JsonRecord = Record<string, unknown>;
type SourceTone = 'online' | 'offline' | 'loading' | 'partial';
type ItemTone = 'success' | 'running' | 'error' | 'neutral';
type RunSummary = { id: string; goal: string; status: string; stepsCompleted: string; created: string; tone: ItemTone };
type StoredRecord = { id: string; name: string; status: string };
type SourceState = { key: string; label: string; path: string; tone: 'online' | 'offline' | 'empty'; detail: string };

const API_BASE = apiUrl('/api');
const runs = ref<RunSummary[]>([]);
const experiments = ref<StoredRecord[]>([]);
const sessions = ref<StoredRecord[]>([]);
const loading = ref(true);
const offline = ref(false);
const errorMessage = ref('');
const statusFilter = ref('all');
const sourceResults = ref<Record<string, 'online' | 'offline' | 'empty'>>({});

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
function statusTone(status: string): ItemTone {
  const normalized = status.toLowerCase();
  if (['completed', 'complete', 'ok', 'success', 'passed'].includes(normalized)) return 'success';
  if (['running', 'started', 'starting', 'queued'].includes(normalized)) return 'running';
  if (['error', 'failed', 'failure', 'stopped', 'cancelled'].includes(normalized)) return 'error';
  return 'neutral';
}
function normalizeRun(value: unknown): RunSummary | null {
  if (!isRecord(value)) return null;
  const id = text(value.run_id) || text(value.id);
  if (!id) return null;
  const status = text(value.status);
  return { id, goal: text(value.goal), status, stepsCompleted: text(value.steps_completed), created: text(value.created) || text(value.published_at), tone: statusTone(status) };
}
function normalizeStored(value: unknown, prefix: string): StoredRecord | null {
  if (!isRecord(value)) return null;
  const id = text(value.id) || text(value.item_id);
  if (!id) return null;
  return { id, name: text(value.name) || text(value.title), status: text(value.status) };
}
function formatDate(value: string): string { if (!value) return 'Unavailable'; const date = new Date(value); return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }); }

const sourceTone = computed<SourceTone>(() => loading.value ? 'loading' : offline.value ? 'offline' : errorMessage.value ? 'partial' : 'online');
const sourceLabel = computed(() => sourceTone.value === 'loading' ? 'Loading records' : sourceTone.value === 'offline' ? 'Offline / unavailable' : sourceTone.value === 'partial' ? 'Partially available' : 'Record sources reachable');
const availableStatuses = computed(() => [...new Set(runs.value.map((run) => run.status || 'Unavailable'))].sort());
const filteredRuns = computed(() => statusFilter.value === 'all' ? runs.value : runs.value.filter((run) => (run.status || 'Unavailable') === statusFilter.value));
const statusCounts = computed(() => {
  const counts = new Map<string, number>();
  for (const run of runs.value) counts.set(run.status || 'Unavailable', (counts.get(run.status || 'Unavailable') ?? 0) + 1);
  const total = runs.value.length || 1;
  return [...counts.entries()].sort((left, right) => right[1] - left[1]).map(([label, count]) => ({ label, count, percent: Math.round((count / total) * 100) }));
});
const sourceStates = computed<SourceState[]>(() => [
  { key: 'runs', label: 'Society runs', path: '/api/society/runs', tone: sourceResults.value.runs ?? 'empty', detail: sourceResults.value.runs === 'online' ? `${runs.value.length} returned` : sourceResults.value.runs === 'offline' ? 'offline' : 'empty' },
  { key: 'experiments', label: 'Experiments', path: '/api/experiments', tone: sourceResults.value.experiments ?? 'empty', detail: sourceResults.value.experiments === 'online' ? `${experiments.value.length} returned` : sourceResults.value.experiments === 'offline' ? 'offline' : 'empty' },
  { key: 'sessions', label: 'Sessions', path: '/api/sessions', tone: sourceResults.value.sessions ?? 'empty', detail: sourceResults.value.sessions === 'online' ? `${sessions.value.length} returned` : sourceResults.value.sessions === 'offline' ? 'offline' : 'empty' },
]);

async function loadAnalytics(): Promise<void> {
  loading.value = true;
  offline.value = false;
  errorMessage.value = '';
  const [runResult, experimentResult, sessionResult] = await Promise.allSettled([
    request<unknown>('/society/runs'),
    request<unknown>('/experiments'),
    request<unknown>('/sessions'),
  ]);
  const failures: string[] = [];
  if (runResult.status === 'fulfilled') {
    const values = Array.isArray(runResult.value) ? runResult.value : isRecord(runResult.value) && Array.isArray(runResult.value.runs) ? runResult.value.runs : [];
    runs.value = values.map(normalizeRun).filter((run): run is RunSummary => Boolean(run));
    sourceResults.value.runs = 'online';
  } else {
    runs.value = [];
    sourceResults.value.runs = isOffline(messageOf(runResult.reason)) ? 'offline' : 'empty';
    failures.push(`runs: ${messageOf(runResult.reason)}`);
  }
  if (experimentResult.status === 'fulfilled') {
    const values = Array.isArray(experimentResult.value) ? experimentResult.value : isRecord(experimentResult.value) && Array.isArray(experimentResult.value.experiments) ? experimentResult.value.experiments : [];
    experiments.value = values.map((item) => normalizeStored(item, 'experiment')).filter((item): item is StoredRecord => Boolean(item));
    sourceResults.value.experiments = 'online';
  } else {
    experiments.value = [];
    sourceResults.value.experiments = isOffline(messageOf(experimentResult.reason)) ? 'offline' : 'empty';
    failures.push(`experiments: ${messageOf(experimentResult.reason)}`);
  }
  if (sessionResult.status === 'fulfilled') {
    const values = Array.isArray(sessionResult.value) ? sessionResult.value : isRecord(sessionResult.value) && Array.isArray(sessionResult.value.sessions) ? sessionResult.value.sessions : [];
    sessions.value = values.map((item) => normalizeStored(item, 'session')).filter((item): item is StoredRecord => Boolean(item));
    sourceResults.value.sessions = 'online';
  } else {
    sessions.value = [];
    sourceResults.value.sessions = isOffline(messageOf(sessionResult.reason)) ? 'offline' : 'empty';
    failures.push(`sessions: ${messageOf(sessionResult.reason)}`);
  }
  if (failures.length) {
    offline.value = failures.some((item) => isOffline(item));
    errorMessage.value = `Unavailable sources — ${failures.join('; ')}`;
  }
  if (!runs.value.some((run) => run.status === statusFilter.value) && statusFilter.value !== 'all') statusFilter.value = 'all';
  loading.value = false;
}

onMounted(() => { void loadAnalytics(); });
</script>

<style scoped>
.analytics-tool { display: flex; width: 100%; min-height: 100%; flex-direction: column; gap: 14px; padding: 22px; color: var(--text); font-size: 13px; }
.tool-header, .tool-header__actions, .panel-header, .tool-footer, .filter-label { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
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
.source-strip code, .tool-footer code, .chart-note code, .table-wrap code, .record-list code, .source-list code { font-family: var(--font-mono); }
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
.analytics-grid { display: grid; grid-template-columns: minmax(300px, .9fr) minmax(400px, 1.3fr); gap: 14px; }
.tool-panel { min-width: 0; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); padding: 14px; }
.panel-header { min-height: 34px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.panel-header h2 { margin: 0; color: var(--text); font-size: 14px; font-weight: 680; }
.count-label { color: var(--text-muted); font-size: 9px; white-space: nowrap; }
.state-block { display: flex; min-height: 180px; flex-direction: column; align-items: center; justify-content: center; gap: 7px; padding: 22px; color: var(--text-muted); text-align: center; line-height: 1.5; }
.state-block--compact { min-height: 150px; }
.state-block strong { color: var(--text); font-size: 13px; }
.state-mark { display: grid; width: 32px; height: 32px; place-items: center; border: 1px solid var(--border-light); border-radius: 50%; color: var(--text-muted); font: 700 15px/1 var(--font-mono); }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border-light); border-top-color: var(--primary); border-radius: 50%; animation: spin .75s linear infinite; }
.status-list { display: flex; flex-direction: column; gap: 10px; }
.status-row { display: grid; grid-template-columns: 100px minmax(0, 1fr) 32px; align-items: center; gap: 8px; }
.status-row__label { overflow: hidden; color: var(--text-dim); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
.status-row__bar { height: 7px; overflow: hidden; border-radius: 999px; background: var(--surface-2); }
.status-row__bar span { display: block; height: 100%; border-radius: inherit; background: var(--primary); }
.status-row strong { color: var(--text); font: 11px/1 var(--font-mono); text-align: right; }
.chart-note, .source-note { margin: 2px 0 0; color: var(--text-muted); font-size: 10px; line-height: 1.5; }
.source-list { display: flex; flex-direction: column; gap: 4px; margin: 0; padding: 0; list-style: none; }
.source-list li { display: grid; grid-template-columns: 22px minmax(0, 1fr) auto; align-items: center; gap: 8px; border-bottom: 1px solid var(--border); padding: 9px 0; }
.source-list li:last-child { border-bottom: 0; }
.source-list__icon { display: grid; width: 20px; height: 20px; place-items: center; border: 1px solid var(--border); border-radius: 50%; font: 700 10px/1 var(--font-mono); }
.source-list__icon--online { border-color: #b9dfd3; background: var(--success-soft); color: var(--success); }
.source-list__icon--offline { border-color: #efc2c7; background: var(--danger-soft); color: var(--danger); }
.source-list__icon--empty { color: var(--text-muted); }
.source-list__body { display: flex; min-width: 0; flex-direction: column; gap: 2px; }
.source-list__body strong { font-size: 11px; }
.source-list__body code { color: var(--text-muted); font-size: 9px; }
.source-list__value { color: var(--text-muted); font: 10px/1.3 var(--font-mono); white-space: nowrap; }
.run-table-panel { display: flex; flex-direction: column; gap: 2px; }
.filter-label { display: flex; align-items: center; gap: 6px; color: var(--text-muted); font-size: 10px; white-space: nowrap; }
.filter-label select { min-height: 30px; border: 1px solid var(--border); border-radius: 5px; background: var(--surface); color: var(--text); padding: 0 7px; font: inherit; font-size: 10px; }
.table-wrap { overflow-x: auto; }
.data-table { width: 100%; border-collapse: collapse; font-size: 10px; }
.data-table th { border-bottom: 1px solid var(--border); color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); letter-spacing: .06em; padding: 7px 8px; text-align: left; text-transform: uppercase; }
.data-table td { border-bottom: 1px solid var(--border); padding: 8px; vertical-align: top; }
.data-table tr:last-child td { border-bottom: 0; }
.data-table td:first-child { max-width: 280px; color: var(--text-dim); }
.table-status { display: inline-flex; border-radius: 999px; background: var(--surface-2); padding: 3px 7px; font: 9px/1.2 var(--font-mono); }
.table-status--success { color: var(--success); }
.table-status--running { color: var(--primary-focus); }
.table-status--error { color: var(--danger); }
.inline-state { color: var(--text-muted); font-size: 11px; line-height: 1.5; padding: 10px 0; }
.record-grid-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.record-list { display: flex; flex-direction: column; margin: 0; padding: 0; list-style: none; }
.record-list li { display: flex; align-items: center; justify-content: space-between; gap: 8px; border-bottom: 1px solid var(--border); padding: 8px 0; }
.record-list li:last-child { border-bottom: 0; }
.record-list__body { display: flex; min-width: 0; flex-direction: column; gap: 2px; }
.record-list__body strong { overflow: hidden; font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
.record-list__body code { color: var(--text-muted); font-size: 9px; }
.record-list__value { color: var(--text-muted); font-size: 9px; white-space: nowrap; }
.tool-footer { justify-content: flex-start; flex-wrap: wrap; color: var(--text-muted); font-size: 10px; }
.desktop-sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; margin: -1px; padding: 0; border: 0; clip: rect(0, 0, 0, 0); white-space: nowrap; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 50% { opacity: .35; } }
@media (max-width: 850px) { .analytics-grid, .record-grid-layout { grid-template-columns: 1fr; } }
@media (max-width: 620px) { .analytics-tool { padding: 14px; } .tool-header { flex-direction: column; } .tool-header__actions { width: 100%; justify-content: flex-start; } .source-strip__detail { width: 100%; margin-left: 0; } .metric-strip { grid-template-columns: repeat(3, 1fr); } .metric-tile--note { grid-column: 1 / -1; border-top: 1px solid var(--border); } .panel-header { align-items: flex-start; flex-direction: column; } .filter-label { width: 100%; justify-content: space-between; } .filter-label select { flex: 1; } }
@media (prefers-reduced-motion: reduce) { .spinner, .source-pill--loading .source-dot { animation: none; } }
</style>
