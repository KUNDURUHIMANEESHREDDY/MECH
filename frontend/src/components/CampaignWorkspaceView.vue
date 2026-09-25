<template>
  <section class="campaign-tool" aria-labelledby="campaign-workspace-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">Workspace / Campaigns</p>
        <h1 id="campaign-workspace-title">Campaign workspace</h1>
        <p class="tool-intro">
          Coordinate research goals through the live Research Society run ledger. A campaign is a backend run;
          this view never treats an unreturned result as a completed measurement.
        </p>
      </div>
      <div class="tool-header__actions">
        <span class="source-pill" :class="`source-pill--${sourceTone}`" role="status">
          <span class="source-dot" aria-hidden="true" />
          {{ sourceLabel }}
        </span>
        <button class="tool-button" type="button" :disabled="loadingRuns" @click="refreshRuns">
          <span aria-hidden="true">↻</span>
          Refresh runs
        </button>
      </div>
    </header>

    <div class="source-strip" role="status" aria-live="polite">
      <span class="source-strip__label">Data source</span>
      <code>GET /api/society/runs</code>
      <span class="source-strip__detail">{{ sourceDetail }}</span>
    </div>

    <div v-if="notice" class="tool-notice" :class="`tool-notice--${noticeTone}`" role="alert">
      <strong>{{ noticeTitle }}</strong>
      <span>{{ notice }}</span>
    </div>

    <form class="launch-panel" novalidate @submit.prevent="launchRun">
      <div class="launch-panel__heading">
        <div>
          <p class="section-kicker">Start a research workflow</p>
          <h2>New campaign goal</h2>
        </div>
        <span class="api-badge">POST /api/society/run</span>
      </div>
      <div class="launch-controls">
        <div class="field field--grow">
          <label for="campaign-goal">Research goal</label>
          <input
            id="campaign-goal"
            v-model="goal"
            class="tool-input"
            type="text"
            autocomplete="off"
            placeholder="Describe the mechanism or question to investigate"
            :disabled="isLaunching"
            :aria-invalid="Boolean(validationMessage)"
            :aria-describedby="validationMessage ? 'campaign-goal-help campaign-validation' : 'campaign-goal-help'"
            @input="validationMessage = ''"
          />
          <p id="campaign-goal-help" class="field-help">
            The Society planner chooses the backend workflow from this goal. The model field is passed to the run contract.
          </p>
          <p v-if="validationMessage" id="campaign-validation" class="field-error" role="alert">{{ validationMessage }}</p>
        </div>
        <div class="field field--model">
          <label for="campaign-model">Model name</label>
          <input
            id="campaign-model"
            v-model="modelName"
            class="tool-input"
            type="text"
            autocomplete="off"
            :disabled="isLaunching"
            aria-describedby="campaign-model-help"
          />
          <p id="campaign-model-help" class="field-help">Backend default: <code>gpt2</code></p>
        </div>
        <div class="launch-buttons">
          <button class="tool-button tool-button--primary" type="submit" :disabled="!canLaunch">
            <span v-if="isLaunching" class="spinner" aria-hidden="true" />
            <span v-else aria-hidden="true">＋</span>
            {{ isLaunching ? 'Starting…' : 'Start campaign' }}
          </button>
          <button v-if="isTracking" class="tool-button" type="button" @click="stopViewing">
            Stop viewing
          </button>
        </div>
      </div>
      <p v-if="trackingMessage" class="tracking-message" role="status">{{ trackingMessage }}</p>
    </form>

    <section class="metric-strip" aria-label="Observed campaign run counts">
      <div class="metric-tile">
        <span class="metric-tile__value">{{ runs.length }}</span>
        <span class="metric-tile__label">Recorded runs</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ activeRunCount }}</span>
        <span class="metric-tile__label">Running now</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ completedRunCount }}</span>
        <span class="metric-tile__label">Completed</span>
      </div>
      <div class="metric-tile metric-tile--note">
        <span class="metric-tile__label">Counts are ledger observations</span>
        <span class="metric-tile__caption">They are not scientific performance metrics.</span>
      </div>
    </section>

    <div class="workspace-grid">
      <section class="tool-panel run-list-panel" aria-labelledby="run-ledger-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Persistent backend records</p>
            <h2 id="run-ledger-title">Run ledger</h2>
          </div>
          <span class="count-label">{{ runs.length }} {{ runs.length === 1 ? 'record' : 'records' }}</span>
        </header>

        <div v-if="loadingRuns" class="state-block state-block--loading" role="status" aria-live="polite">
          <span class="spinner" aria-hidden="true" />
          <span>Reading the Society run ledger…</span>
        </div>
        <div v-else-if="offline" class="state-block" role="alert">
          <span class="state-mark" aria-hidden="true">!</span>
          <strong>Backend offline</strong>
          <span>Start the MECH runtime at <code>localhost:8000</code> to read or create campaigns.</span>
        </div>
        <div v-else-if="runs.length === 0" class="state-block">
          <span class="state-mark" aria-hidden="true">∅</span>
          <strong>No campaigns recorded</strong>
          <span>The run ledger is reachable, but no persisted Society runs were returned.</span>
        </div>
        <ul v-else class="run-list" aria-label="Research Society runs">
          <li v-for="run in runs" :key="run.id">
            <button
              class="run-row"
              :class="{ 'run-row--selected': run.id === selectedRunId }"
              type="button"
              :aria-pressed="run.id === selectedRunId"
              @click="selectRun(run.id)"
            >
              <span class="run-row__marker" :class="`run-row__marker--${statusTone(run.status)}`" aria-hidden="true" />
              <span class="run-row__body">
                <strong>{{ run.goal || 'Goal unavailable' }}</strong>
                <span class="run-row__meta">
                  <code>{{ run.id }}</code>
                  <span>{{ run.status || 'status unavailable' }}</span>
                  <span v-if="run.stepsCompleted">{{ run.stepsCompleted }} steps</span>
                </span>
              </span>
              <span class="run-row__time">{{ formatDate(run.created) }}</span>
            </button>
          </li>
        </ul>
      </section>

      <section class="tool-panel run-detail-panel" aria-labelledby="run-detail-title" :aria-busy="loadingDetail">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Selected record</p>
            <h2 id="run-detail-title">{{ selectedRun ? 'Run detail' : 'No run selected' }}</h2>
          </div>
          <span v-if="selectedRun" class="status-chip" :class="`status-chip--${statusTone(selectedRun.status)}`">
            {{ selectedRun.status || 'status unavailable' }}
          </span>
        </header>

        <div v-if="loadingDetail" class="state-block state-block--loading" role="status" aria-live="polite">
          <span class="spinner" aria-hidden="true" />
          <span>Loading the backend trace…</span>
        </div>
        <div v-else-if="!selectedRun" class="state-block">
          <span class="state-mark" aria-hidden="true">↖</span>
          <strong>Select a run</strong>
          <span>Choose a ledger record to inspect its trace and returned events.</span>
        </div>
        <div v-else class="detail-content">
          <dl class="record-grid">
            <div>
              <dt>Run ID</dt>
              <dd><code>{{ selectedRun.id }}</code></dd>
            </div>
            <div>
              <dt>Goal</dt>
              <dd>{{ selectedRun.goal || 'Unavailable' }}</dd>
            </div>
            <div>
              <dt>Model</dt>
              <dd>{{ selectedRun.model || 'Not returned' }}</dd>
            </div>
            <div>
              <dt>Created</dt>
              <dd>{{ formatDate(selectedRun.created) }}</dd>
            </div>
          </dl>

          <div class="detail-section">
            <div class="detail-section__heading">
              <h3>Pipeline trace</h3>
              <span>{{ selectedTrace.length }} {{ selectedTrace.length === 1 ? 'step' : 'steps' }}</span>
            </div>
            <ol v-if="selectedTrace.length" class="trace-list">
              <li v-for="(step, index) in selectedTrace" :key="`${step.node}-${index}`" class="trace-item">
                <span class="trace-item__index">{{ String(index + 1).padStart(2, '0') }}</span>
                <span class="trace-item__body">
                  <strong>{{ step.node || 'Unnamed node' }}</strong>
                  <span>{{ step.agent || 'Agent unavailable' }} · {{ step.status || 'status unavailable' }}</span>
                  <span v-if="step.reason || step.error" class="trace-item__detail">{{ step.error || step.reason }}</span>
                </span>
              </li>
            </ol>
            <p v-else class="inline-empty">The backend returned no trace steps for this run.</p>
          </div>

          <div class="detail-section">
            <div class="detail-section__heading">
              <h3>Returned events</h3>
              <span>{{ selectedEvents.length }} {{ selectedEvents.length === 1 ? 'event' : 'events' }}</span>
            </div>
            <ol v-if="selectedEvents.length" class="event-list" role="log" aria-live="polite">
              <li v-for="(event, index) in selectedEvents.slice(-12)" :key="`${event.timestamp || 'event'}-${index}`">
                <span class="event-list__type">{{ event.event_type || 'Unknown event' }}</span>
                <span class="event-list__label">{{ eventLabel(event) }}</span>
                <time v-if="event.timestamp" :datetime="event.timestamp">{{ formatDate(event.timestamp) }}</time>
              </li>
            </ol>
            <p v-else class="inline-empty">No event payload is available in this record.</p>
          </div>

          <div v-if="selectedReport" class="report-preview">
            <div class="detail-section__heading">
              <h3>Backend report payload</h3>
              <span>Returned with run</span>
            </div>
            <pre>{{ selectedReport }}</pre>
          </div>
          <p v-else class="inline-empty">No report payload is attached to this run.</p>
        </div>
      </section>
    </div>

    <footer class="tool-footer">
      <span>Source contract</span>
      <code>GET /api/society/runs</code>
      <code>GET /api/society/runs/:runId</code>
      <code>POST /api/society/run</code>
      <span class="tool-footer__note">Stopping this panel stops local tracking only; the API exposes no cancellation call.</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { apiUrl } from '../services/api';
import {
  getSocietyRun,
  startSocietyRun,
  streamSocietyRun,
  type SocietyEvent,
} from '../services/societyService';

type Phase = 'idle' | 'starting' | 'running' | 'stopped' | 'done' | 'failed';
type SourceTone = 'online' | 'offline' | 'loading' | 'partial';
type StatusTone = 'success' | 'running' | 'error' | 'neutral';

type JsonRecord = Record<string, unknown>;

interface RunSummary {
  id: string;
  goal: string;
  status: string;
  model: string;
  created: string;
  stepsCompleted: string;
}

interface TraceStep {
  node: string;
  agent: string;
  status: string;
  reason: string;
  error: string;
}

interface RunDetail extends RunSummary {
  events: SocietyEvent[];
  trace: TraceStep[];
  report: string;
}

const API_BASE = apiUrl('/api');
const POLL_INTERVAL_MS = 2000;

const goal = ref('');
const modelName = ref('gpt2');
const runs = ref<RunSummary[]>([]);
const selectedRunId = ref('');
const selectedRun = ref<RunDetail | null>(null);
const loadingRuns = ref(true);
const loadingDetail = ref(false);
const offline = ref(false);
const phase = ref<Phase>('idle');
const notice = ref('');
const noticeTone = ref<'warning' | 'error' | 'info'>('info');
const noticeTitle = ref('');
const validationMessage = ref('');
const trackingMessage = ref('');
const activeRunId = ref('');

let unsubscribeStream: (() => void) | null = null;
let pollTimer: number | null = null;
let requestVersion = 0;
let detailVersion = 0;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function text(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return fallback;
}

function reportText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (isRecord(value) && typeof value.markdown === 'string') return value.markdown;
  if (isRecord(value) && typeof value.report === 'string') return value.report;
  return '';
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function looksOffline(message: string): boolean {
  return /offline|failed to fetch|network|load|connection|timeout/i.test(message);
}

async function request<T>(path: string): Promise<T> {
  if (typeof navigator !== 'undefined' && navigator.onLine === false) {
    throw new Error('Browser is offline');
  }
  const response = await fetch(`${API_BASE}${path}`, { headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error(`GET ${path} returned ${response.status}`);
  return response.json() as Promise<T>;
}

function normalizeSummary(value: unknown): RunSummary | null {
  if (!isRecord(value)) return null;
  const id = text(value.run_id) || text(value.id);
  if (!id) return null;
  return {
    id,
    goal: text(value.goal),
    status: text(value.status),
    model: text(value.model_name) || text(value.model),
    created: text(value.created) || text(value.created_at) || text(value.published_at),
    stepsCompleted: text(value.steps_completed),
  };
}

function normalizeTrace(value: unknown): TraceStep[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!isRecord(item)) return [];
    const node = text(item.node);
    if (!node) return [];
    return [{
      node,
      agent: text(item.agent),
      status: text(item.status),
      reason: text(item.reason),
      error: text(item.error),
    }];
  });
}

function normalizeEvents(value: unknown): SocietyEvent[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!isRecord(item)) return [];
    return [{
      event_type: text(item.event_type, 'Unknown event'),
      payload: isRecord(item.payload) ? item.payload : {},
      ...(typeof item.timestamp === 'string' ? { timestamp: item.timestamp } : {}),
    }];
  });
}

function normalizeDetail(raw: unknown, fallback: RunSummary): RunDetail {
  const record = isRecord(raw) ? raw : {};
  const result = isRecord(record.result) ? record.result : record;
  const publication = isRecord(result.publication) ? result.publication : isRecord(record.publication) ? record.publication : null;
  const rawReport = publication?.report ?? result.report ?? record.report;
  return {
    ...fallback,
    goal: text(result.goal, fallback.goal),
    status: text(record.status, text(result.status, fallback.status)),
    model: text(record.model_name, text(result.model_name, fallback.model)),
    created: text(record.created, fallback.created),
    events: normalizeEvents(record.events ?? result.events),
    trace: normalizeTrace(result.trace ?? record.trace),
    report: reportText(rawReport),
  };
}

function statusTone(status: string): StatusTone {
  const normalized = status.toLowerCase();
  if (['completed', 'complete', 'ok', 'success', 'passed'].includes(normalized)) return 'success';
  if (['running', 'started', 'starting', 'queued'].includes(normalized)) return 'running';
  if (['error', 'failed', 'failure', 'cancelled', 'stopped'].includes(normalized)) return 'error';
  return 'neutral';
}

function formatDate(value: string): string {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
}

function eventLabel(event: SocietyEvent): string {
  const payload = isRecord(event.payload) ? event.payload : {};
  switch (event.event_type) {
    case 'ResearchStarted': return `Goal: ${text(payload.goal, 'unavailable')}`;
    case 'ExperimentQueued': return Array.isArray(payload.plan)
      ? `Queued: ${payload.plan.map((item) => text(item)).filter(Boolean).join(' → ') || 'plan unavailable'}`
      : 'Experiment queued';
    case 'DiscoveryCreated': return `Discovery ${text(payload.discovery_id, '(ID unavailable)')}`;
    case 'CircuitValidated': return `Validation event: ${text(payload.fidelity_pct, 'fidelity not returned')}`;
    case 'PublicationGenerated': return `Publication ${text(payload.experiment_id, '(ID unavailable)')}`;
    case 'ResearchFinished': return `Finished: ${text(payload.steps_completed, 'step count unavailable')}`;
    default: return text(event.event_type, 'Event payload');
  }
}

const isLaunching = computed(() => phase.value === 'starting');
const isTracking = computed(() => phase.value === 'starting' || phase.value === 'running');
const canLaunch = computed(() => Boolean(goal.value.trim()) && !isTracking.value);
const activeRunCount = computed(() => runs.value.filter((run) => statusTone(run.status) === 'running').length);
const completedRunCount = computed(() => runs.value.filter((run) => statusTone(run.status) === 'success').length);
const sourceTone = computed<SourceTone>(() => {
  if (loadingRuns.value || isLaunching.value) return 'loading';
  if (offline.value) return 'offline';
  if (noticeTone.value === 'error') return 'partial';
  return 'online';
});
const sourceLabel = computed(() => {
  if (sourceTone.value === 'loading') return 'Connecting';
  if (sourceTone.value === 'offline') return 'Offline / unavailable';
  if (sourceTone.value === 'partial') return 'Partially available';
  return runs.value.length ? 'Backend ledger live' : 'Backend reachable';
});
const selectedTrace = computed(() => selectedRun.value?.trace ?? []);
const selectedEvents = computed(() => selectedRun.value?.events ?? []);
const selectedReport = computed(() => selectedRun.value?.report ?? '');
const sourceDetail = computed(() => {
  if (sourceTone.value === 'loading') return 'Waiting for the run ledger response.';
  if (offline.value) return 'No run data can be read until the runtime is reachable.';
  if (sourceTone.value === 'partial') return 'The ledger responded, but at least one request needs attention.';
  return runs.value.length ? 'Run summaries are read from persisted backend evidence records.' : 'The endpoint responded without run records.';
});

function setNotice(message: string, tone: 'warning' | 'error' | 'info' = 'info', title = 'Workspace notice') {
  notice.value = message;
  noticeTone.value = tone;
  noticeTitle.value = title;
}

function mergeEvents(incoming: SocietyEvent[]): void {
  const current = selectedRun.value?.events ?? [];
  const merged = new Map<string, SocietyEvent>();
  for (const event of [...current, ...incoming]) merged.set(`${event.timestamp || ''}:${event.event_type}:${JSON.stringify(event.payload || {})}`, event);
  if (selectedRun.value) selectedRun.value.events = [...merged.values()].slice(-1000);
}

function applyResult(id: string, rawResult: unknown): void {
  const summary = runs.value.find((run) => run.id === id) ?? {
    id,
    goal: '',
    status: '',
    model: modelName.value,
    created: '',
    stepsCompleted: '',
  };
  const detail = normalizeDetail(rawResult, summary);
  detail.status = text(isRecord(rawResult) ? rawResult.status : '', 'ended');
  if (selectedRunId.value === id) selectedRun.value = detail;
  const listIndex = runs.value.findIndex((run) => run.id === id);
  if (listIndex >= 0) runs.value[listIndex] = { ...summary, status: detail.status, stepsCompleted: text(isRecord(rawResult) ? rawResult.steps_completed : summary.stepsCompleted, summary.stepsCompleted) };
  phase.value = statusTone(detail.status) === 'error' ? 'failed' : 'done';
  trackingMessage.value = 'The backend run ended. Live tracking in this panel is complete.';
}

function stopTransport(): void {
  unsubscribeStream?.();
  unsubscribeStream = null;
  if (pollTimer !== null) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function selectRun(id: string): Promise<void> {
  if (!id) return;
  selectedRunId.value = id;
  const version = ++detailVersion;
  loadingDetail.value = true;
  const fallback = runs.value.find((run) => run.id === id);
  try {
    const raw = await request<unknown>(`/society/runs/${encodeURIComponent(id)}`);
    if (version !== detailVersion) return;
    selectedRun.value = normalizeDetail(raw, fallback ?? { id, goal: '', status: '', model: '', created: '', stepsCompleted: '' });
  } catch (error) {
    if (version !== detailVersion) return;
    selectedRun.value = fallback ? { ...fallback, events: [], trace: [], report: '' } : null;
    setNotice(`Could not load run ${id}: ${errorMessage(error)}`, looksOffline(errorMessage(error)) ? 'warning' : 'error', 'Run detail unavailable');
  } finally {
    if (version === detailVersion) loadingDetail.value = false;
  }
}

async function refreshRuns(): Promise<void> {
  const version = ++requestVersion;
  loadingRuns.value = true;
  offline.value = false;
  notice.value = '';
  try {
    const raw = await request<unknown>('/society/runs');
    if (version !== requestVersion) return;
    if (isRecord(raw) && raw.status === 'error') throw new Error(text(raw.error, 'Run ledger returned an error'));
    const values = Array.isArray(raw) ? raw : isRecord(raw) && Array.isArray(raw.runs) ? raw.runs : [];
    runs.value = values.map(normalizeSummary).filter((run): run is RunSummary => Boolean(run));
    if (!runs.value.some((run) => run.id === selectedRunId.value)) selectedRunId.value = runs.value[0]?.id ?? '';
    if (selectedRunId.value) await selectRun(selectedRunId.value);
  } catch (error) {
    if (version !== requestVersion) return;
    offline.value = looksOffline(errorMessage(error));
    runs.value = [];
    selectedRunId.value = '';
    selectedRun.value = null;
    setNotice(
      offline.value ? 'The MECH runtime is not responding, so the campaign ledger is unavailable.' : errorMessage(error),
      offline.value ? 'warning' : 'error',
      offline.value ? 'Backend offline' : 'Could not load campaigns',
    );
  } finally {
    if (version === requestVersion) loadingRuns.value = false;
  }
}

function startPolling(id: string): void {
  if (pollTimer !== null) return;
  pollTimer = window.setInterval(async () => {
    if (phase.value !== 'running' || activeRunId.value !== id) return;
    try {
      const raw = await getSocietyRun(id);
      const detail = normalizeDetail(raw, runs.value.find((run) => run.id === id) ?? { id, goal: '', status: '', model: modelName.value, created: '', stepsCompleted: '' });
      if (selectedRunId.value === id) selectedRun.value = detail;
      const status = text(isRecord(raw) ? raw.status : '');
      if (status && status !== 'running') {
        const result = isRecord(raw) ? raw.result : raw;
        applyResult(id, result ?? raw);
        stopTransport();
      }
    } catch (error) {
      setNotice(`Live tracking fell back to polling: ${errorMessage(error)}`, 'warning', 'Tracking notice');
    }
  }, POLL_INTERVAL_MS);
}

function attachRun(id: string): void {
  stopTransport();
  try {
    unsubscribeStream = streamSocietyRun(id, {
      onEvent: (event) => {
        if (phase.value === 'running' && activeRunId.value === id) mergeEvents([event]);
      },
      onDone: (result) => {
        if (phase.value !== 'running' || activeRunId.value !== id) return;
        if (Object.keys(result || {}).length === 0) {
          startPolling(id);
          return;
        }
        applyResult(id, result);
        stopTransport();
        void refreshRuns();
      },
      onError: () => {
        if (phase.value === 'running' && activeRunId.value === id) {
          trackingMessage.value = 'The event stream was interrupted; polling the same backend run for updates.';
          startPolling(id);
        }
      },
    });
  } catch (error) {
    trackingMessage.value = `Live stream unavailable: ${errorMessage(error)}. Polling instead.`;
    startPolling(id);
  }
}

async function launchRun(): Promise<void> {
  const submittedGoal = goal.value.trim();
  if (!submittedGoal) {
    validationMessage.value = 'Enter a research goal before starting a campaign.';
    return;
  }
  if (typeof navigator !== 'undefined' && navigator.onLine === false) {
    setNotice('This browser is offline. Reconnect before starting a campaign.', 'warning', 'Browser offline');
    return;
  }
  if (isTracking.value) return;

  stopTransport();
  phase.value = 'starting';
  notice.value = '';
  validationMessage.value = '';
  trackingMessage.value = 'Submitting the goal to the Society run endpoint…';
  try {
    const started = await startSocietyRun(submittedGoal, modelName.value.trim() || 'gpt2');
    activeRunId.value = started.runId;
    phase.value = 'running';
    trackingMessage.value = 'The backend accepted the run. Listening for real Society events…';
    const optimistic: RunSummary = {
      id: started.runId,
      goal: submittedGoal,
      status: started.status || 'running',
      model: modelName.value.trim() || 'gpt2',
      created: new Date().toISOString(),
      stepsCompleted: '',
    };
    runs.value = [optimistic, ...runs.value.filter((run) => run.id !== optimistic.id)];
    selectedRunId.value = optimistic.id;
    selectedRun.value = { ...optimistic, events: [], trace: [], report: '' };
    attachRun(optimistic.id);
  } catch (error) {
    phase.value = 'failed';
    activeRunId.value = '';
    trackingMessage.value = '';
    const message = errorMessage(error);
    offline.value = looksOffline(message);
    setNotice(
      offline.value ? 'The run was not submitted because the MECH runtime is offline.' : message,
      offline.value ? 'warning' : 'error',
      offline.value ? 'Backend offline' : 'Campaign could not start',
    );
  }
}

function stopViewing(): void {
  if (!isTracking.value) return;
  requestVersion += 1;
  detailVersion += 1;
  stopTransport();
  phase.value = 'stopped';
  trackingMessage.value = 'This panel stopped receiving updates. The API has no cancellation endpoint, so the backend run may still be active.';
}

onMounted(() => {
  void refreshRuns();
});

onBeforeUnmount(() => {
  requestVersion += 1;
  detailVersion += 1;
  stopTransport();
});
</script>

<style scoped>
.campaign-tool {
  display: flex;
  width: 100%;
  min-height: 100%;
  flex-direction: column;
  gap: 14px;
  padding: 22px;
  color: var(--text);
  font-size: 13px;
}

.tool-header,
.tool-header__actions,
.panel-header,
.launch-panel__heading,
.detail-section__heading,
.tool-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.tool-header {
  align-items: flex-start;
}

.tool-header__copy { min-width: 0; }

.tool-eyebrow,
.section-kicker,
.source-strip__label,
.metric-tile__label,
.field label,
.api-badge,
.count-label,
.status-chip,
.source-pill,
.tool-footer > span:first-child {
  color: var(--text-muted);
  font: 700 10px/1.2 var(--font-mono);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.tool-eyebrow { margin: 0 0 6px; }
.section-kicker { margin: 0 0 5px; }

.tool-header h1 {
  margin: 0;
  color: var(--text);
  font-size: clamp(22px, 3vw, 30px);
  font-weight: 720;
  letter-spacing: -0.035em;
  line-height: 1.1;
}

.tool-intro {
  max-width: 760px;
  margin: 8px 0 0;
  color: var(--text-muted);
  line-height: 1.55;
}

.tool-header__actions {
  flex: 0 0 auto;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.source-pill,
.status-chip,
.api-badge {
  display: inline-flex;
  min-height: 25px;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  padding: 0 9px;
  white-space: nowrap;
}

.source-pill--online { border-color: #b9dfd3; background: var(--success-soft); color: var(--success); }
.source-pill--offline { border-color: #efc2c7; background: var(--danger-soft); color: var(--danger); }
.source-pill--partial { border-color: #ecd79c; background: var(--warning-soft); color: var(--warning); }
.source-pill--loading { border-color: #cbd8ef; background: var(--accent-soft); color: var(--primary-focus); }
.source-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.source-pill--loading .source-dot { animation: pulse 1.2s ease-in-out infinite; }

.tool-button {
  display: inline-flex;
  min-height: 34px;
  align-items: center;
  justify-content: center;
  gap: 7px;
  border: 1px solid var(--border-light);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  cursor: pointer;
  font: 650 12px/1.2 var(--font);
  padding: 0 12px;
  white-space: nowrap;
}
.tool-button:hover:not(:disabled) { border-color: var(--text); background: var(--surface-2); }
.tool-button:disabled { cursor: not-allowed; opacity: 0.45; }
.tool-button--primary { border-color: var(--primary); background: var(--primary); color: #fff; }
.tool-button--primary:hover:not(:disabled) { border-color: var(--primary-focus); background: var(--primary-focus); }

.source-strip {
  display: flex;
  min-height: 34px;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  padding: 7px 10px;
  color: var(--text-muted);
  font-size: 11px;
}
.source-strip code,
.tool-footer code,
.run-row code,
.record-grid code,
.field-help code { font-family: var(--font-mono); }
.source-strip__detail { margin-left: auto; }

.tool-notice {
  display: flex;
  flex-direction: column;
  gap: 3px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--primary);
  border-radius: 6px;
  background: var(--accent-soft);
  padding: 10px 12px;
  line-height: 1.45;
}
.tool-notice--warning { border-left-color: var(--warning); background: var(--warning-soft); color: var(--warning); }
.tool-notice--error { border-left-color: var(--danger); background: var(--danger-soft); color: var(--danger); }

.launch-panel,
.tool-panel,
.metric-strip {
  border: 1px solid var(--border);
  border-radius: 9px;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}
.launch-panel,
.tool-panel { padding: 14px; }
.launch-panel__heading { margin-bottom: 12px; }
.launch-panel h2,
.panel-header h2,
.detail-section h3 { margin: 0; color: var(--text); font-size: 14px; font-weight: 680; }
.api-badge { color: var(--text-muted); }
.launch-controls { display: grid; grid-template-columns: minmax(0, 1fr) 190px auto; align-items: start; gap: 12px; }
.field { display: flex; flex-direction: column; gap: 5px; }
.field label { color: var(--text-dim); }
.tool-input {
  width: 100%;
  min-height: 36px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  padding: 0 10px;
  font: inherit;
  font-size: 12px;
}
.tool-input:focus { border-color: var(--primary); box-shadow: 0 0 0 2px var(--accent-soft); outline: none; }
.field-help,
.field-error,
.tracking-message { margin: 0; color: var(--text-muted); font-size: 10px; line-height: 1.45; }
.field-error { color: var(--danger); font-weight: 650; }
.launch-buttons { display: flex; align-items: center; gap: 8px; padding-top: 21px; }
.tracking-message { margin-top: 10px; }

.metric-strip { display: grid; grid-template-columns: repeat(3, minmax(100px, 1fr)) minmax(180px, 1.5fr); overflow: hidden; }
.metric-tile { min-height: 66px; border-right: 1px solid var(--border); padding: 12px 14px; }
.metric-tile:last-child { border-right: 0; }
.metric-tile__value { display: block; margin-bottom: 3px; color: var(--text); font: 700 21px/1 var(--font-mono); }
.metric-tile__label { display: block; color: var(--text-muted); font-size: 9px; }
.metric-tile__caption { display: block; margin-top: 4px; color: var(--text-muted); font-size: 10px; }
.metric-tile--note { display: flex; flex-direction: column; justify-content: center; background: var(--surface-2); }

.workspace-grid { display: grid; grid-template-columns: minmax(280px, 0.82fr) minmax(420px, 1.45fr); gap: 14px; min-height: 430px; }
.run-list-panel,
.run-detail-panel { min-width: 0; }
.panel-header { min-height: 34px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.count-label { color: var(--text-muted); font-size: 9px; white-space: nowrap; }
.status-chip--success { border-color: #b9dfd3; background: var(--success-soft); color: var(--success); }
.status-chip--running { border-color: #cbd8ef; background: var(--accent-soft); color: var(--primary-focus); }
.status-chip--error { border-color: #efc2c7; background: var(--danger-soft); color: var(--danger); }
.status-chip--neutral { color: var(--text-muted); }

.state-block { display: flex; min-height: 220px; flex-direction: column; align-items: center; justify-content: center; gap: 7px; padding: 22px; color: var(--text-muted); text-align: center; line-height: 1.5; }
.state-block strong { color: var(--text); font-size: 13px; }
.state-block code { font-family: var(--font-mono); font-size: 11px; }
.state-block--loading { min-height: 180px; }
.state-mark { display: grid; width: 32px; height: 32px; place-items: center; border: 1px solid var(--border-light); border-radius: 50%; color: var(--text-muted); font: 700 15px/1 var(--font-mono); }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border-light); border-top-color: var(--primary); border-radius: 50%; animation: spin 0.75s linear infinite; }

.run-list,
.trace-list,
.event-list { display: flex; flex-direction: column; gap: 6px; margin: 0; padding: 0; list-style: none; }
.run-list { max-height: 500px; overflow-y: auto; }
.run-row { display: grid; width: 100%; grid-template-columns: 8px minmax(0, 1fr) auto; align-items: start; gap: 9px; border: 1px solid transparent; border-radius: 6px; background: transparent; color: var(--text); cursor: pointer; padding: 9px 8px; text-align: left; }
.run-row:hover { border-color: var(--border); background: var(--surface-2); }
.run-row--selected { border-color: var(--primary); background: var(--accent-soft); }
.run-row__marker { width: 7px; height: 7px; margin-top: 4px; border-radius: 50%; background: var(--text-muted); }
.run-row__marker--success { background: var(--success); }
.run-row__marker--running { background: var(--primary); animation: pulse 1.2s ease-in-out infinite; }
.run-row__marker--error { background: var(--danger); }
.run-row__body { display: flex; min-width: 0; flex-direction: column; gap: 4px; }
.run-row__body strong { overflow: hidden; font-size: 12px; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
.run-row__meta { display: flex; flex-wrap: wrap; gap: 4px 8px; color: var(--text-muted); font-size: 10px; }
.run-row__time { color: var(--text-muted); font: 10px/1.3 var(--font-mono); white-space: nowrap; }

.detail-content { display: flex; min-width: 0; flex-direction: column; gap: 16px; }
.record-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 0; }
.record-grid > div { min-width: 0; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 8px 10px; }
.record-grid dt { color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); letter-spacing: 0.06em; text-transform: uppercase; }
.record-grid dd { margin: 4px 0 0; overflow-wrap: anywhere; color: var(--text); font-size: 11px; line-height: 1.4; }
.detail-section { display: flex; flex-direction: column; gap: 8px; }
.detail-section__heading { min-height: 24px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.detail-section__heading h3 { font-size: 12px; }
.detail-section__heading > span { color: var(--text-muted); font-size: 10px; }
.trace-list,
.event-list { max-height: 250px; overflow-y: auto; }
.trace-item { display: grid; grid-template-columns: 25px minmax(0, 1fr); gap: 8px; border-bottom: 1px solid var(--border); padding: 7px 2px; }
.trace-item:last-child { border-bottom: 0; }
.trace-item__index { color: var(--text-muted); font: 10px/1.4 var(--font-mono); }
.trace-item__body { display: flex; min-width: 0; flex-direction: column; gap: 2px; }
.trace-item__body strong { font-size: 11px; }
.trace-item__body span { color: var(--text-muted); font-size: 10px; }
.trace-item__detail { color: var(--warning) !important; overflow-wrap: anywhere; }
.event-list li { display: grid; grid-template-columns: minmax(100px, 0.45fr) minmax(0, 1fr) auto; gap: 9px; border-bottom: 1px solid var(--border); padding: 7px 2px; font-size: 10px; line-height: 1.4; }
.event-list li:last-child { border-bottom: 0; }
.event-list__type { font-family: var(--font-mono); font-weight: 650; }
.event-list__label { color: var(--text-dim); overflow-wrap: anywhere; }
.event-list time { color: var(--text-muted); white-space: nowrap; }
.inline-empty { margin: 0; color: var(--text-muted); font-size: 11px; line-height: 1.45; }
.report-preview { display: flex; flex-direction: column; gap: 8px; }
.report-preview pre { max-height: 260px; margin: 0; overflow: auto; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 10px; color: var(--text-dim); font: 10px/1.55 var(--font-mono); white-space: pre-wrap; overflow-wrap: anywhere; }

.tool-footer { justify-content: flex-start; flex-wrap: wrap; color: var(--text-muted); font-size: 10px; }
.tool-footer > span:first-child { color: var(--text-dim); }
.tool-footer code { border: 1px solid var(--border); border-radius: 4px; background: var(--surface-2); padding: 3px 6px; font-size: 9px; }
.tool-footer__note { margin-left: auto; }

@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 50% { opacity: 0.35; } }

@media (max-width: 920px) {
  .launch-controls { grid-template-columns: minmax(0, 1fr) 180px; }
  .launch-buttons { grid-column: 1 / -1; padding-top: 0; }
  .workspace-grid { grid-template-columns: 1fr; }
  .run-detail-panel { min-height: 380px; }
}

@media (max-width: 620px) {
  .campaign-tool { padding: 14px; }
  .tool-header { flex-direction: column; }
  .tool-header__actions { width: 100%; justify-content: flex-start; }
  .source-strip__detail { width: 100%; margin-left: 0; }
  .launch-controls { grid-template-columns: 1fr; }
  .launch-buttons { grid-column: auto; }
  .metric-strip { grid-template-columns: repeat(3, 1fr); }
  .metric-tile--note { grid-column: 1 / -1; border-top: 1px solid var(--border); }
  .record-grid { grid-template-columns: 1fr; }
  .event-list li { grid-template-columns: 1fr; gap: 2px; }
  .tool-footer__note { width: 100%; margin-left: 0; }
}

@media (prefers-reduced-motion: reduce) {
  .spinner,
  .source-pill--loading .source-dot,
  .run-row__marker--running { animation: none; }
}
</style>
