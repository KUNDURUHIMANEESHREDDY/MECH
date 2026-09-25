<template>
  <section class="trace-tool" aria-labelledby="reasoning-trace-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">Research / Execution evidence</p>
        <h1 id="reasoning-trace-title">Reasoning trace</h1>
        <p class="tool-intro">
          Read the execution trace emitted by a Research Society run. This is an event and step viewer, not a model
          chain-of-thought simulator; only backend-returned nodes, statuses, and reasons are shown.
        </p>
      </div>
      <div class="tool-header__actions">
        <span class="source-pill" :class="`source-pill--${sourceTone}`" role="status">
          <span class="source-dot" aria-hidden="true" />
          {{ sourceLabel }}
        </span>
        <button class="tool-button" type="button" :disabled="loadingRuns" @click="loadRuns">
          <span aria-hidden="true">↻</span>
          Refresh trace
        </button>
      </div>
    </header>

    <div class="source-strip" role="status" aria-live="polite">
      <span class="source-strip__label">Trace source</span>
      <code>GET /api/society/runs/:runId</code>
      <span class="source-strip__detail">The current runtime does not expose a separate reasoning endpoint.</span>
    </div>

    <div v-if="errorMessage" class="tool-notice" role="alert">
      <strong>{{ offline ? 'Trace source offline' : 'Could not read trace source' }}</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <section class="metric-strip" aria-label="Trace observations">
      <div class="metric-tile">
        <span class="metric-tile__value">{{ runs.length }}</span>
        <span class="metric-tile__label">Runs available</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ selectedTrace.length }}</span>
        <span class="metric-tile__label">Trace steps returned</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ selectedEvents.length }}</span>
        <span class="metric-tile__label">Events returned</span>
      </div>
      <div class="metric-tile metric-tile--note">
        <span class="metric-tile__label">Interpretation boundary</span>
        <span class="metric-tile__caption">A step is a backend record. Missing reasoning is reported as unavailable.</span>
      </div>
    </section>

    <section class="tool-panel selector-panel" aria-labelledby="trace-selector-title">
      <div class="selector-copy">
        <p class="section-kicker">Choose a backend record</p>
        <h2 id="trace-selector-title">Trace run</h2>
      </div>
      <div class="selector-control">
        <label for="trace-run-select">Research Society run</label>
        <select id="trace-run-select" v-model="selectedRunId" :disabled="!runs.length || loadingRuns" @change="loadSelectedRun">
          <option value="" disabled>{{ runs.length ? 'Select a run' : 'No runs returned' }}</option>
          <option v-for="run in runs" :key="run.id" :value="run.id">{{ run.goal || run.id }} · {{ run.status || 'status unavailable' }}</option>
        </select>
      </div>
    </section>

    <div class="trace-layout">
      <section class="tool-panel timeline-panel" aria-labelledby="timeline-title" :aria-busy="loadingDetail">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Returned pipeline record</p>
            <h2 id="timeline-title">Execution timeline</h2>
          </div>
          <span v-if="selectedRun" class="status-chip" :class="`status-chip--${statusTone(selectedRun.status)}`">{{ selectedRun.status || 'status unavailable' }}</span>
        </header>

        <div v-if="loadingDetail" class="state-block state-block--loading" role="status" aria-live="polite">
          <span class="spinner" aria-hidden="true" />
          <span>Loading trace steps…</span>
        </div>
        <div v-else-if="offline" class="state-block" role="alert">
          <span class="state-mark" aria-hidden="true">!</span>
          <strong>Trace unavailable offline</strong>
          <span>Reconnect the MECH runtime to read run steps.</span>
        </div>
        <div v-else-if="!selectedRun" class="state-block">
          <span class="state-mark" aria-hidden="true">↖</span>
          <strong>Select a run</strong>
          <span>A run must be selected before its trace can be requested.</span>
        </div>
        <div v-else-if="selectedTrace.length === 0" class="state-block">
          <span class="state-mark" aria-hidden="true">∅</span>
          <strong>No trace steps returned</strong>
          <span>The selected backend record contains no readable pipeline trace.</span>
        </div>
        <ol v-else class="timeline-list">
          <li v-for="(step, index) in selectedTrace" :key="`${step.node}-${index}`" class="timeline-item">
            <div class="timeline-item__rail" aria-hidden="true">
              <span class="timeline-item__dot" :class="`timeline-item__dot--${statusTone(step.status)}`" />
              <span v-if="index < selectedTrace.length - 1" class="timeline-item__line" />
            </div>
            <div class="timeline-item__content">
              <div class="timeline-item__heading">
                <span class="timeline-item__index">{{ String(index + 1).padStart(2, '0') }}</span>
                <h3>{{ step.node || 'Unnamed node' }}</h3>
                <span class="status-text" :class="`status-text--${statusTone(step.status)}`">{{ step.status || 'status unavailable' }}</span>
              </div>
              <p class="timeline-item__agent">Agent: {{ step.agent || 'unavailable' }}</p>
              <p v-if="step.reason || step.error" class="timeline-item__message">{{ step.error || step.reason }}</p>
              <details v-if="step.operation" class="raw-details">
                <summary>Returned operation</summary>
                <code>{{ step.operation }}</code>
              </details>
            </div>
          </li>
        </ol>
      </section>

      <section class="tool-panel event-panel" aria-labelledby="trace-events-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Backend event stream</p>
            <h2 id="trace-events-title">Events</h2>
          </div>
          <span class="count-label">{{ selectedEvents.length }} total</span>
        </header>
        <ol v-if="selectedEvents.length" class="event-list" role="log" aria-live="polite" aria-label="Backend trace events">
          <li v-for="(event, index) in selectedEvents" :key="`${event.timestamp || 'event'}-${index}`" class="event-item">
            <div class="event-item__heading">
              <strong>{{ event.event_type || 'Unknown event' }}</strong>
              <time v-if="event.timestamp" :datetime="event.timestamp">{{ formatDate(event.timestamp) }}</time>
            </div>
            <p>{{ eventLabel(event) }}</p>
            <details class="raw-details">
              <summary>Payload</summary>
              <pre>{{ prettyPayload(event.payload) }}</pre>
            </details>
          </li>
        </ol>
        <div v-else class="state-block state-block--compact">
          <span class="state-mark" aria-hidden="true">∅</span>
          <strong>No events returned</strong>
          <span>The selected record has no event payload to display.</span>
        </div>
      </section>
    </div>

    <section v-if="selectedRun" class="tool-panel provenance-panel" aria-labelledby="trace-provenance-title">
      <header class="panel-header">
        <div>
          <p class="section-kicker">Record context</p>
          <h2 id="trace-provenance-title">Trace provenance</h2>
        </div>
      </header>
      <dl class="record-grid">
        <div><dt>Run ID</dt><dd><code>{{ selectedRun.id }}</code></dd></div>
        <div><dt>Goal</dt><dd>{{ selectedRun.goal || 'Unavailable' }}</dd></div>
        <div><dt>Model</dt><dd>{{ selectedRun.model || 'Not returned' }}</dd></div>
        <div><dt>Created</dt><dd>{{ formatDate(selectedRun.created) }}</dd></div>
      </dl>
    </section>

    <footer class="tool-footer">
      <span>Contract note</span>
      <span>Trace fields are shown exactly as returned; the client does not infer hidden model reasoning.</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { apiUrl } from '../services/api';

type JsonRecord = Record<string, unknown>;
type SourceTone = 'online' | 'offline' | 'loading' | 'partial';
type StatusTone = 'success' | 'running' | 'error' | 'neutral';

interface RunSummary { id: string; goal: string; status: string; model: string; created: string; }
interface TraceStep { node: string; agent: string; status: string; reason: string; error: string; operation: string; }
interface RunDetail extends RunSummary { events: TraceEvent[]; trace: TraceStep[]; }
interface TraceEvent { event_type: string; payload: JsonRecord; timestamp?: string; }

const API_BASE = apiUrl('/api');
const runs = ref<RunSummary[]>([]);
const selectedRunId = ref('');
const selectedRun = ref<RunDetail | null>(null);
const loadingRuns = ref(true);
const loadingDetail = ref(false);
const offline = ref(false);
const errorMessage = ref('');
let selectedVersion = 0;

function isRecord(value: unknown): value is JsonRecord { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function text(value: unknown, fallback = ''): string { return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? String(value) : fallback; }
function messageOf(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function offlineMessage(message: string): boolean { return /offline|failed to fetch|network|load|connection|timeout/i.test(message); }
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
  return { id, goal: text(value.goal), status: text(value.status), model: text(value.model_name) || text(value.model), created: text(value.created) || text(value.created_at) || text(value.published_at) };
}
function normalizeTrace(value: unknown): TraceStep[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!isRecord(item)) return [];
    const node = text(item.node);
    if (!node) return [];
    return [{ node, agent: text(item.agent), status: text(item.status), reason: text(item.reason), error: text(item.error), operation: text(item.op) }];
  });
}
function normalizeEvents(value: unknown): TraceEvent[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!isRecord(item)) return [];
    return [{ event_type: text(item.event_type, 'Unknown event'), payload: isRecord(item.payload) ? item.payload : {}, ...(typeof item.timestamp === 'string' ? { timestamp: item.timestamp } : {}) }];
  });
}
function normalizeDetail(value: unknown, fallback: RunSummary): RunDetail {
  const record = isRecord(value) ? value : {};
  const result = isRecord(record.result) ? record.result : record;
  return { ...fallback, goal: text(result.goal, fallback.goal), status: text(record.status, text(result.status, fallback.status)), model: text(record.model_name, text(result.model_name, fallback.model)), created: text(record.created, fallback.created), events: normalizeEvents(record.events ?? result.events), trace: normalizeTrace(result.trace ?? record.trace) };
}
function statusTone(status: string): StatusTone {
  const normalized = status.toLowerCase();
  if (['completed', 'complete', 'ok', 'success', 'passed', 'loaded'].includes(normalized)) return 'success';
  if (['running', 'started', 'starting', 'queued'].includes(normalized)) return 'running';
  if (['error', 'failed', 'failure', 'unavailable', 'stopped'].includes(normalized)) return 'error';
  return 'neutral';
}
function formatDate(value: string): string {
  if (!value) return 'Unavailable';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
}
function eventLabel(event: TraceEvent): string {
  const payload = event.payload;
  switch (event.event_type) {
    case 'ResearchStarted': return `Goal: ${text(payload.goal, 'unavailable')}`;
    case 'ExperimentQueued': return Array.isArray(payload.plan) ? `Plan: ${payload.plan.map((item) => text(item)).filter(Boolean).join(' → ') || 'unavailable'}` : 'Experiment queued';
    case 'DiscoveryCreated': return `Discovery ${text(payload.discovery_id, '(ID unavailable)')}`;
    case 'CircuitValidated': return `Validation event · ${text(payload.fidelity_pct, 'fidelity not returned')}`;
    case 'PublicationGenerated': return `Publication ${text(payload.experiment_id, '(ID unavailable)')}`;
    case 'ResearchFinished': return `Finished · ${text(payload.steps_completed, 'step count unavailable')}`;
    default: return 'Backend event payload';
  }
}
function prettyPayload(payload: JsonRecord): string {
  try { return JSON.stringify(payload, null, 2); } catch { return 'Payload could not be serialized.'; }
}

const selectedTrace = computed(() => selectedRun.value?.trace ?? []);
const selectedEvents = computed(() => selectedRun.value?.events ?? []);
const sourceTone = computed<SourceTone>(() => loadingRuns.value ? 'loading' : offline.value ? 'offline' : errorMessage.value ? 'partial' : 'online');
const sourceLabel = computed(() => sourceTone.value === 'loading' ? 'Loading runs' : sourceTone.value === 'offline' ? 'Offline / unavailable' : sourceTone.value === 'partial' ? 'Partially available' : 'Backend trace source live');
const statusText = computed(() => selectedRun.value?.status || '');

async function loadSelectedRun(): Promise<void> {
  const id = selectedRunId.value;
  if (!id) { selectedRun.value = null; return; }
  const version = ++selectedVersion;
  loadingDetail.value = true;
  errorMessage.value = '';
  const fallback = runs.value.find((run) => run.id === id) ?? { id, goal: '', status: '', model: '', created: '' };
  try {
    const raw = await request<unknown>(`/society/runs/${encodeURIComponent(id)}`);
    if (version !== selectedVersion) return;
    selectedRun.value = normalizeDetail(raw, fallback);
  } catch (error) {
    if (version !== selectedVersion) return;
    selectedRun.value = { ...fallback, events: [], trace: [] };
    errorMessage.value = `Could not load trace for ${id}: ${messageOf(error)}`;
    offline.value = offlineMessage(messageOf(error));
  } finally {
    if (version === selectedVersion) loadingDetail.value = false;
  }
}

async function loadRuns(): Promise<void> {
  loadingRuns.value = true;
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
    offline.value = offlineMessage(errorMessage.value);
  } finally {
    loadingRuns.value = false;
  }
}

onMounted(() => { void loadRuns(); });
</script>

<style scoped>
.trace-tool { display: flex; width: 100%; min-height: 100%; flex-direction: column; gap: 14px; padding: 22px; color: var(--text); font-size: 13px; }
.tool-header, .tool-header__actions, .panel-header, .tool-footer, .selector-panel { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.tool-header { align-items: flex-start; }
.tool-header__copy { min-width: 0; }
.tool-eyebrow, .section-kicker, .source-strip__label, .metric-tile__label, .count-label, .tool-footer > span:first-child, .status-text { color: var(--text-muted); font: 700 10px/1.2 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; }
.tool-eyebrow, .section-kicker { margin: 0 0 5px; }
.tool-header h1 { margin: 0; color: var(--text); font-size: clamp(22px, 3vw, 30px); font-weight: 720; letter-spacing: -.035em; line-height: 1.1; }
.tool-intro { max-width: 760px; margin: 8px 0 0; color: var(--text-muted); line-height: 1.55; }
.tool-header__actions { flex: 0 0 auto; flex-wrap: wrap; justify-content: flex-end; }
.source-pill, .status-chip { display: inline-flex; min-height: 25px; align-items: center; gap: 6px; border: 1px solid var(--border); border-radius: 999px; background: var(--surface-2); padding: 0 9px; white-space: nowrap; }
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
.source-strip code, .tool-footer code, .record-grid code, .raw-details code, .raw-details pre { font-family: var(--font-mono); }
.source-strip__detail { margin-left: auto; }
.tool-notice { display: flex; flex-direction: column; gap: 3px; border: 1px solid #efc2c7; border-left: 3px solid var(--danger); border-radius: 6px; background: var(--danger-soft); padding: 10px 12px; color: var(--danger); line-height: 1.45; }
.metric-strip { display: grid; grid-template-columns: repeat(3, minmax(100px, 1fr)) minmax(210px, 1.5fr); overflow: hidden; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); }
.metric-tile { min-height: 66px; border-right: 1px solid var(--border); padding: 12px 14px; }
.metric-tile:last-child { border-right: 0; }
.metric-tile__value { display: block; margin-bottom: 3px; color: var(--text); font: 700 21px/1 var(--font-mono); }
.metric-tile__label { display: block; color: var(--text-muted); font-size: 9px; }
.metric-tile__caption { display: block; margin-top: 4px; color: var(--text-muted); font-size: 10px; }
.metric-tile--note { display: flex; flex-direction: column; justify-content: center; background: var(--surface-2); }
.tool-panel { min-width: 0; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); padding: 14px; }
.selector-panel { display: grid; grid-template-columns: 1fr minmax(260px, 420px); align-items: end; }
.selector-copy h2, .panel-header h2 { margin: 0; color: var(--text); font-size: 14px; font-weight: 680; }
.selector-control { display: flex; flex-direction: column; gap: 5px; }
.selector-control label { color: var(--text-dim); font-size: 10px; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; }
.selector-control select { width: 100%; min-height: 36px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--text); padding: 0 9px; font: inherit; font-size: 11px; }
.selector-control select:focus { border-color: var(--primary); box-shadow: 0 0 0 2px var(--accent-soft); outline: none; }
.trace-layout { display: grid; grid-template-columns: minmax(400px, 1.2fr) minmax(300px, .8fr); gap: 14px; min-height: 480px; }
.panel-header { min-height: 34px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.count-label { font-size: 9px; white-space: nowrap; }
.status-chip--success { border-color: #b9dfd3; background: var(--success-soft); color: var(--success); }
.status-chip--running { border-color: #cbd8ef; background: var(--accent-soft); color: var(--primary-focus); }
.status-chip--error { border-color: #efc2c7; background: var(--danger-soft); color: var(--danger); }
.status-chip--neutral { color: var(--text-muted); }
.state-block { display: flex; min-height: 220px; flex-direction: column; align-items: center; justify-content: center; gap: 7px; padding: 22px; color: var(--text-muted); text-align: center; line-height: 1.5; }
.state-block strong { color: var(--text); font-size: 13px; }
.state-block--compact { min-height: 180px; }
.state-block--loading { min-height: 180px; }
.state-mark { display: grid; width: 32px; height: 32px; place-items: center; border: 1px solid var(--border-light); border-radius: 50%; color: var(--text-muted); font: 700 15px/1 var(--font-mono); }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border-light); border-top-color: var(--primary); border-radius: 50%; animation: spin .75s linear infinite; }
.timeline-list { display: flex; flex-direction: column; margin: 0; padding: 0; list-style: none; }
.timeline-item { display: grid; grid-template-columns: 25px minmax(0, 1fr); gap: 9px; min-height: 72px; }
.timeline-item__rail { position: relative; display: flex; flex-direction: column; align-items: center; }
.timeline-item__dot { z-index: 1; width: 10px; height: 10px; margin-top: 3px; border: 2px solid var(--surface); border-radius: 50%; background: var(--text-muted); box-shadow: 0 0 0 1px var(--border-light); }
.timeline-item__dot--success { background: var(--success); }
.timeline-item__dot--running { background: var(--primary); animation: pulse 1.2s ease-in-out infinite; }
.timeline-item__dot--error { background: var(--danger); }
.timeline-item__line { width: 1px; flex: 1; margin: 4px 0; background: var(--border); }
.timeline-item__content { min-width: 0; padding: 0 0 14px; }
.timeline-item__heading { display: flex; min-width: 0; align-items: center; flex-wrap: wrap; gap: 7px; }
.timeline-item__index { color: var(--text-muted); font: 10px/1.3 var(--font-mono); }
.timeline-item h3 { margin: 0; color: var(--text); font-size: 12px; font-weight: 680; }
.status-text { margin-left: auto; font-size: 9px; }
.status-text--success { color: var(--success); }
.status-text--running { color: var(--primary-focus); }
.status-text--error { color: var(--danger); }
.timeline-item__agent, .timeline-item__message { margin: 4px 0 0; color: var(--text-muted); font-size: 10px; line-height: 1.45; }
.timeline-item__message { color: var(--warning); overflow-wrap: anywhere; }
.raw-details { margin-top: 7px; color: var(--text-muted); font-size: 10px; }
.raw-details summary { cursor: pointer; color: var(--text-dim); }
.raw-details code, .raw-details pre { display: block; margin-top: 5px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; }
.raw-details pre { max-height: 180px; border: 1px solid var(--border); border-radius: 5px; background: var(--surface-2); padding: 7px; color: var(--text-dim); font-size: 9px; line-height: 1.5; }
.event-list { display: flex; max-height: 500px; flex-direction: column; gap: 0; margin: 0; overflow-y: auto; padding: 0; list-style: none; }
.event-item { border-bottom: 1px solid var(--border); padding: 9px 0; }
.event-item:last-child { border-bottom: 0; }
.event-item__heading { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
.event-item__heading strong { color: var(--text); font: 650 10px/1.3 var(--font-mono); }
.event-item__heading time { color: var(--text-muted); font: 9px/1.3 var(--font-mono); white-space: nowrap; }
.event-item p { margin: 4px 0 0; color: var(--text-dim); font-size: 10px; line-height: 1.45; }
.provenance-panel { display: flex; flex-direction: column; gap: 2px; }
.record-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 0; }
.record-grid > div { min-width: 0; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 8px 10px; }
.record-grid dt { color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); letter-spacing: .06em; text-transform: uppercase; }
.record-grid dd { margin: 4px 0 0; overflow-wrap: anywhere; color: var(--text); font-size: 11px; line-height: 1.4; }
.tool-footer { justify-content: flex-start; flex-wrap: wrap; color: var(--text-muted); font-size: 10px; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 50% { opacity: .35; } }
@media (max-width: 1000px) { .trace-layout { grid-template-columns: 1fr; } .event-panel { min-height: 300px; } }
@media (max-width: 700px) { .trace-tool { padding: 14px; } .tool-header { flex-direction: column; } .tool-header__actions { width: 100%; justify-content: flex-start; } .source-strip__detail { width: 100%; margin-left: 0; } .metric-strip { grid-template-columns: repeat(3, 1fr); } .metric-tile--note { grid-column: 1 / -1; border-top: 1px solid var(--border); } .selector-panel { grid-template-columns: 1fr; align-items: stretch; } .record-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 460px) { .record-grid { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { .spinner, .source-pill--loading .source-dot, .timeline-item__dot--running { animation: none; } }
</style>
