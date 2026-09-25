<template>
  <section class="health-tool" aria-labelledby="scientific-health-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">Research / Runtime checks</p>
        <h1 id="scientific-health-title">Scientific health</h1>
        <p class="tool-intro">
          Check whether the research runtime and its evidence sources are reachable. A dedicated scientific health
          score is not exposed by the current API, so no health percentage is invented here.
        </p>
      </div>
      <div class="tool-header__actions">
        <span class="source-pill" :class="`source-pill--${sourceTone}`" role="status">
          <span class="source-dot" aria-hidden="true" />
          {{ sourceLabel }}
        </span>
        <button class="tool-button" type="button" :disabled="loading" @click="loadHealth">
          <span aria-hidden="true">↻</span>
          Run checks
        </button>
      </div>
    </header>

    <div class="source-strip" role="status" aria-live="polite">
      <span class="source-strip__label">Checks</span>
      <code>GET /api/status</code>
      <code>GET /api/society/runs</code>
      <code>GET /api/experiments</code>
      <span class="source-strip__detail">Connectivity and returned-record checks only.</span>
    </div>

    <div v-if="errorMessage" class="tool-notice" :class="`tool-notice--${offline ? 'warning' : 'error'}`" role="alert">
      <strong>{{ offline ? 'Runtime checks incomplete' : 'One or more checks failed' }}</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <section class="check-grid" aria-label="Runtime and data-source checks">
      <article v-for="check in checks" :key="check.key" class="check-card" :class="`check-card--${check.tone}`">
        <div class="check-card__top">
          <span class="check-card__icon" aria-hidden="true">{{ check.tone === 'online' ? '✓' : check.tone === 'offline' ? '!' : '—' }}</span>
          <span class="check-card__state">{{ check.state }}</span>
        </div>
        <h2>{{ check.label }}</h2>
        <p>{{ check.detail }}</p>
        <code>{{ check.path }}</code>
      </article>
    </section>

    <section class="unavailable-banner" aria-labelledby="health-score-unavailable-title">
      <div class="unavailable-banner__mark" aria-hidden="true">—</div>
      <div>
        <p class="section-kicker">Scientific health source</p>
        <h2 id="health-score-unavailable-title">Health scoring unavailable</h2>
        <p>
          The current runtime exposes status and record endpoints, but no <code>/api/health</code> or scientific
          validation summary contract. This panel therefore reports operational health only.
        </p>
      </div>
    </section>

    <div class="health-grid">
      <section class="tool-panel run-health-panel" aria-labelledby="run-health-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Persisted evidence</p>
            <h2 id="run-health-title">Run health signals</h2>
          </div>
          <span class="count-label">{{ runs.length }} records</span>
        </header>
        <div v-if="loading" class="state-block state-block--compact" role="status" aria-live="polite">
          <span class="spinner" aria-hidden="true" />
          <span>Checking Society run records…</span>
        </div>
        <div v-else-if="!runs.length" class="state-block state-block--compact">
          <span class="state-mark" aria-hidden="true">∅</span>
          <strong>No run signals</strong>
          <span>No persisted Society runs were returned for inspection.</span>
        </div>
        <ul v-else class="signal-list">
          <li v-for="run in runs.slice(0, 12)" :key="run.id">
            <span class="signal-list__marker" :class="`signal-list__marker--${run.tone}`" aria-hidden="true" />
            <span class="signal-list__body"><strong>{{ run.goal || 'Goal unavailable' }}</strong><code>{{ run.id }}</code></span>
            <span class="signal-list__status">{{ run.status || 'Unavailable' }}</span>
          </li>
        </ul>
        <p class="panel-note">Statuses are displayed as returned. They are not converted into a health grade.</p>
      </section>

      <section class="tool-panel record-health-panel" aria-labelledby="record-health-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Source inventory</p>
            <h2 id="record-health-title">Record availability</h2>
          </div>
        </header>
        <dl class="record-health">
          <div><dt>Society run ledger</dt><dd>{{ sourceSummary.runs }}</dd></div>
          <div><dt>Experiment storage</dt><dd>{{ sourceSummary.experiments }}</dd></div>
          <div><dt>Session storage</dt><dd>{{ sourceSummary.sessions }}</dd></div>
          <div><dt>Scientific health score</dt><dd>Unavailable — no endpoint</dd></div>
        </dl>
        <p class="panel-note">An empty local store is a valid observation; it is not an error unless the source itself is unreachable.</p>
      </section>
    </div>

    <section class="tool-panel guidance-panel" aria-labelledby="guidance-title">
      <header class="panel-header">
        <div>
          <p class="section-kicker">Operator guidance</p>
          <h2 id="guidance-title">What this tool can tell you</h2>
        </div>
      </header>
      <ul class="guidance-list">
        <li><span class="guidance-list__icon" aria-hidden="true">1</span><span><strong>Runtime reachable</strong> — the API returned a status response.</span></li>
        <li><span class="guidance-list__icon" aria-hidden="true">2</span><span><strong>Evidence present</strong> — persisted runs or records are available to inspect.</span></li>
        <li><span class="guidance-list__icon" aria-hidden="true">3</span><span><strong>Scientific verdict</strong> — unavailable until a dedicated validation summary contract exists.</span></li>
      </ul>
    </section>

    <footer class="tool-footer">
      <span>Next step</span>
      <span>Use a Society run or a validation-capable backend surface for scientific claims; this view remains operational.</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { apiUrl } from '../services/api';

type JsonRecord = Record<string, unknown>;
type CheckTone = 'online' | 'offline' | 'empty';
type RunSummary = { id: string; goal: string; status: string; tone: 'success' | 'running' | 'error' | 'neutral' };
type SourceSummary = { runs: string; experiments: string; sessions: string };
type Check = { key: string; label: string; path: string; state: string; detail: string; tone: CheckTone };

const API_BASE = apiUrl('/api');
const statusValue = ref('');
const statusAvailable = ref(false);
const runs = ref<RunSummary[]>([]);
const experimentCount = ref(0);
const sessionCount = ref(0);
const sourceResults = ref<Record<string, CheckTone>>({});
const loading = ref(true);
const offline = ref(false);
const errorMessage = ref('');

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
function statusTone(status: string): RunSummary['tone'] {
  const normalized = status.toLowerCase();
  if (['completed', 'complete', 'ok', 'success', 'passed'].includes(normalized)) return 'success';
  if (['running', 'started', 'starting', 'queued'].includes(normalized)) return 'running';
  if (['error', 'failed', 'failure', 'stopped', 'blocked', 'unavailable'].includes(normalized)) return 'error';
  return 'neutral';
}
function normalizeRun(value: unknown): RunSummary | null {
  if (!isRecord(value)) return null;
  const id = text(value.run_id) || text(value.id);
  if (!id) return null;
  const status = text(value.status);
  return { id, goal: text(value.goal), status, tone: statusTone(status) };
}
function sourceDetail(tone: CheckTone, count: number): string { return tone === 'online' ? `${count} returned` : tone === 'offline' ? 'Unavailable' : 'Empty response'; }

const sourceTone = computed(() => loading.value ? 'loading' : offline.value ? 'offline' : statusAvailable.value ? 'online' : 'partial');
const sourceLabel = computed(() => loading.value ? 'Running checks' : offline.value ? 'Offline / unavailable' : statusAvailable.value ? 'Runtime reachable' : 'Partially available');
const sourceSummary = computed<SourceSummary>(() => ({
  runs: sourceDetail(sourceResults.value.runs ?? 'empty', runs.value.length),
  experiments: sourceDetail(sourceResults.value.experiments ?? 'empty', experimentCount.value),
  sessions: sourceDetail(sourceResults.value.sessions ?? 'empty', sessionCount.value),
}));
const checks = computed<Check[]>(() => [
  { key: 'runtime', label: 'Backend runtime', path: 'GET /api/status', state: statusAvailable.value ? 'Online' : offline.value ? 'Offline' : 'Unavailable', detail: statusAvailable.value ? `Returned status: ${statusValue.value || 'not specified'}` : 'The runtime did not return a status response.', tone: statusAvailable.value ? 'online' : offline.value ? 'offline' : 'empty' },
  { key: 'runs', label: 'Society evidence', path: 'GET /api/society/runs', state: sourceResults.value.runs === 'online' ? 'Available' : sourceResults.value.runs === 'offline' ? 'Offline' : 'Empty', detail: sourceResults.value.runs === 'online' ? `${runs.value.length} persisted run records returned.` : 'No usable run records returned.', tone: sourceResults.value.runs ?? 'empty' },
  { key: 'experiments', label: 'Experiment records', path: 'GET /api/experiments', state: sourceResults.value.experiments === 'online' ? 'Available' : sourceResults.value.experiments === 'offline' ? 'Offline' : 'Empty', detail: sourceDetail(sourceResults.value.experiments ?? 'empty', experimentCount.value), tone: sourceResults.value.experiments ?? 'empty' },
  { key: 'sessions', label: 'Session records', path: 'GET /api/sessions', state: sourceResults.value.sessions === 'online' ? 'Available' : sourceResults.value.sessions === 'offline' ? 'Offline' : 'Empty', detail: sourceDetail(sourceResults.value.sessions ?? 'empty', sessionCount.value), tone: sourceResults.value.sessions ?? 'empty' },
]);

async function loadHealth(): Promise<void> {
  loading.value = true;
  offline.value = false;
  errorMessage.value = '';
  const [statusResult, runResult, experimentResult, sessionResult] = await Promise.allSettled([
    request<unknown>('/status'), request<unknown>('/society/runs'), request<unknown>('/experiments'), request<unknown>('/sessions'),
  ]);
  if (statusResult.status === 'fulfilled') {
    statusAvailable.value = true;
    statusValue.value = isRecord(statusResult.value) ? text(statusResult.value.status, 'ok') : 'response received';
  } else {
    statusAvailable.value = false;
    statusValue.value = '';
    errorMessage.value = `Runtime status unavailable: ${messageOf(statusResult.reason)}`;
    offline.value = isOffline(errorMessage.value);
  }
  if (runResult.status === 'fulfilled') {
    const values = Array.isArray(runResult.value) ? runResult.value : isRecord(runResult.value) && Array.isArray(runResult.value.runs) ? runResult.value.runs : [];
    runs.value = values.map(normalizeRun).filter((run): run is RunSummary => Boolean(run));
    sourceResults.value.runs = 'online';
  } else {
    runs.value = [];
    sourceResults.value.runs = isOffline(messageOf(runResult.reason)) ? 'offline' : 'empty';
    errorMessage.value = [errorMessage.value, `Run evidence unavailable: ${messageOf(runResult.reason)}`].filter(Boolean).join(' · ');
  }
  if (experimentResult.status === 'fulfilled') {
    const values = Array.isArray(experimentResult.value) ? experimentResult.value : isRecord(experimentResult.value) && Array.isArray(experimentResult.value.experiments) ? experimentResult.value.experiments : [];
    experimentCount.value = values.length;
    sourceResults.value.experiments = values.length ? 'online' : 'empty';
  } else {
    experimentCount.value = 0;
    sourceResults.value.experiments = isOffline(messageOf(experimentResult.reason)) ? 'offline' : 'empty';
  }
  if (sessionResult.status === 'fulfilled') {
    const values = Array.isArray(sessionResult.value) ? sessionResult.value : isRecord(sessionResult.value) && Array.isArray(sessionResult.value.sessions) ? sessionResult.value.sessions : [];
    sessionCount.value = values.length;
    sourceResults.value.sessions = values.length ? 'online' : 'empty';
  } else {
    sessionCount.value = 0;
    sourceResults.value.sessions = isOffline(messageOf(sessionResult.reason)) ? 'offline' : 'empty';
  }
  loading.value = false;
}

onMounted(() => { void loadHealth(); });
</script>

<style scoped>
.health-tool { display: flex; width: 100%; min-height: 100%; flex-direction: column; gap: 14px; padding: 22px; color: var(--text); font-size: 13px; }
.tool-header, .tool-header__actions, .panel-header, .tool-footer { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.tool-header { align-items: flex-start; }
.tool-header__copy { min-width: 0; }
.tool-eyebrow, .section-kicker, .source-strip__label, .count-label, .tool-footer > span:first-child, .check-card__state { color: var(--text-muted); font: 700 10px/1.2 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; }
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
.source-strip code, .tool-footer code, .check-card code, .record-health code, .unavailable-banner code, .signal-list code { font-family: var(--font-mono); }
.source-strip__detail { margin-left: auto; }
.tool-notice { display: flex; flex-direction: column; gap: 3px; border: 1px solid var(--border); border-left: 3px solid var(--danger); border-radius: 6px; background: var(--danger-soft); padding: 10px 12px; color: var(--danger); line-height: 1.45; }
.tool-notice--warning { border-left-color: var(--warning); background: var(--warning-soft); color: var(--warning); }
.check-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.check-card { display: flex; min-height: 150px; flex-direction: column; gap: 7px; border: 1px solid var(--border); border-top: 3px solid var(--border-light); border-radius: 8px; background: var(--surface); padding: 12px; box-shadow: var(--shadow-sm); }
.check-card--online { border-top-color: var(--success); }
.check-card--offline { border-top-color: var(--danger); }
.check-card--empty { border-top-color: var(--warning); }
.check-card__top { display: flex; align-items: center; justify-content: space-between; }
.check-card__icon { display: grid; width: 22px; height: 22px; place-items: center; border: 1px solid currentColor; border-radius: 50%; color: var(--text-muted); font: 700 10px/1 var(--font-mono); }
.check-card--online .check-card__icon { color: var(--success); }
.check-card--offline .check-card__icon { color: var(--danger); }
.check-card--empty .check-card__icon { color: var(--warning); }
.check-card h2 { margin: 0; color: var(--text); font-size: 13px; font-weight: 680; }
.check-card p { flex: 1; margin: 0; color: var(--text-muted); font-size: 10px; line-height: 1.5; }
.check-card code { color: var(--text-muted); font-size: 9px; }
.unavailable-banner { display: flex; align-items: flex-start; gap: 12px; border: 1px dashed var(--border-light); border-radius: 8px; background: var(--surface-2); padding: 14px; }
.unavailable-banner__mark { display: grid; width: 30px; height: 30px; flex: 0 0 auto; place-items: center; border: 1px solid var(--border-light); border-radius: 50%; color: var(--text-muted); font: 700 14px/1 var(--font-mono); }
.unavailable-banner h2 { margin: 0; color: var(--text); font-size: 14px; font-weight: 680; }
.unavailable-banner p:last-child { max-width: 760px; margin: 6px 0 0; color: var(--text-muted); font-size: 11px; line-height: 1.5; }
.health-grid { display: grid; grid-template-columns: minmax(400px, 1.2fr) minmax(300px, .8fr); gap: 14px; }
.tool-panel { min-width: 0; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); padding: 14px; }
.panel-header { min-height: 34px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.panel-header h2 { margin: 0; color: var(--text); font-size: 14px; font-weight: 680; }
.count-label { font-size: 9px; white-space: nowrap; }
.state-block { display: flex; min-height: 160px; flex-direction: column; align-items: center; justify-content: center; gap: 7px; padding: 22px; color: var(--text-muted); text-align: center; line-height: 1.5; }
.state-block--compact { min-height: 150px; }
.state-block strong { color: var(--text); font-size: 13px; }
.state-mark { display: grid; width: 32px; height: 32px; place-items: center; border: 1px solid var(--border-light); border-radius: 50%; color: var(--text-muted); font: 700 15px/1 var(--font-mono); }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border-light); border-top-color: var(--primary); border-radius: 50%; animation: spin .75s linear infinite; }
.signal-list { display: flex; flex-direction: column; max-height: 300px; margin: 0; overflow-y: auto; padding: 0; list-style: none; }
.signal-list li { display: grid; grid-template-columns: 8px minmax(0, 1fr) auto; align-items: center; gap: 8px; border-bottom: 1px solid var(--border); padding: 8px 0; }
.signal-list li:last-child { border-bottom: 0; }
.signal-list__marker { width: 7px; height: 7px; border-radius: 50%; background: var(--text-muted); }
.signal-list__marker--success { background: var(--success); }
.signal-list__marker--running { background: var(--primary); }
.signal-list__marker--error { background: var(--danger); }
.signal-list__body { display: flex; min-width: 0; flex-direction: column; gap: 2px; }
.signal-list__body strong { overflow: hidden; font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
.signal-list__body code { color: var(--text-muted); font-size: 9px; }
.signal-list__status { color: var(--text-muted); font-size: 9px; white-space: nowrap; }
.panel-note { margin: 10px 0 0; color: var(--text-muted); font-size: 10px; line-height: 1.5; }
.record-health { display: flex; flex-direction: column; gap: 0; margin: 0; }
.record-health > div { display: flex; align-items: center; justify-content: space-between; gap: 8px; border-bottom: 1px solid var(--border); padding: 10px 0; }
.record-health > div:last-child { border-bottom: 0; }
.record-health dt { color: var(--text-muted); font-size: 11px; }
.record-health dd { margin: 0; color: var(--text); font: 10px/1.3 var(--font-mono); text-align: right; }
.guidance-list { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 0; padding: 0; list-style: none; }
.guidance-list li { display: flex; align-items: flex-start; gap: 8px; color: var(--text-muted); font-size: 10px; line-height: 1.5; }
.guidance-list strong { color: var(--text); }
.guidance-list__icon { display: grid; width: 22px; height: 22px; flex: 0 0 auto; place-items: center; border: 1px solid var(--border); border-radius: 50%; color: var(--text-dim); font: 700 10px/1 var(--font-mono); }
.tool-footer { justify-content: flex-start; flex-wrap: wrap; color: var(--text-muted); font-size: 10px; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 50% { opacity: .35; } }
@media (max-width: 1000px) { .check-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .health-grid { grid-template-columns: 1fr; } }
@media (max-width: 620px) { .health-tool { padding: 14px; } .tool-header { flex-direction: column; } .tool-header__actions { width: 100%; justify-content: flex-start; } .source-strip__detail { width: 100%; margin-left: 0; } .check-grid, .guidance-list { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { .spinner, .source-pill--loading .source-dot { animation: none; } }
</style>
