<template>
  <section class="society" aria-labelledby="research-society-title">
    <header class="society__header">
      <div>
        <h2 id="research-society-title">Research Society</h2>
        <p class="society__intro">
          Run the backend planner, executor, inspector, discoverer, critic, and scribe workflow.
          No model needs to be loaded in this view first. Stages without explicit live provenance stop before scientific validation or publication.
        </p>
      </div>
      <a
        class="society__history-link"
        :href="runHistoryUrl"
        target="_blank"
        rel="noopener noreferrer"
      >
        Run history <span aria-hidden="true">↗</span>
        <span class="society__sr-only"> (backend JSON, opens in a new tab)</span>
      </a>
    </header>

    <form class="society__run-card" novalidate @submit.prevent="handleRun">
      <div class="society__field">
        <label for="society-goal">Research goal</label>
        <div class="society__controls">
          <input
            id="society-goal"
            v-model="goal"
            type="text"
            class="society__goal"
            placeholder="e.g. Reproduce IOI on gpt2-small and find causally important heads"
            :disabled="isActive"
            :aria-invalid="Boolean(validationMessage)"
            :aria-describedby="validationMessage ? 'society-goal-help society-validation' : 'society-goal-help'"
            autocomplete="off"
            @input="validationMessage = ''"
          />
          <button
            type="submit"
            class="society__button society__button--primary"
            :disabled="!canRun"
          >
            <span
              v-if="phase === 'starting'"
              class="society__spinner"
              aria-hidden="true"
            />
            {{ phase === 'starting' ? 'Starting…' : 'Run Society' }}
          </button>
          <button
            v-if="isActive"
            type="button"
            class="society__button society__button--stop"
            aria-describedby="society-stop-help"
            @click="handleStop"
          >
            Stop
          </button>
        </div>
        <p id="society-goal-help" class="society__help">
          A non-empty goal is required. The backend loads its configured model only if the workflow needs it.
        </p>
        <p
          v-if="validationMessage"
          id="society-validation"
          class="society__error-text"
          role="alert"
        >
          {{ validationMessage }}
        </p>
        <p id="society-stop-help" class="society__help">
          {{ stopHelp }}
        </p>
      </div>
    </form>

    <div class="society__status" role="status" aria-live="polite">
      <span class="society__status-dot" :class="`society__status-dot--${statusTone}`" aria-hidden="true" />
      <div>
        <strong>{{ statusHeading }}</strong>
        <span>{{ statusDetail }}</span>
      </div>
      <span class="society__backend-state">{{ backendStateLabel }}</span>
    </div>

    <div v-if="offlineNotice" class="society__notice society__notice--warning" role="alert">
      <strong>Backend unavailable</strong>
      <span>{{ offlineNotice }}</span>
    </div>

    <div v-if="errorMessage" class="society__notice society__notice--error" role="alert">
      <strong>Society request failed</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <p v-if="runId" class="society__run-meta">
      <span>Run <code>{{ runId }}</code></span>
      <span v-if="runStatus"> · Backend status: {{ runStatus }}</span>
      <span v-else-if="phase === 'done' || phase === 'failed'"> · Backend status unavailable</span>
      <span v-if="runSummary"> · {{ runSummary }}</span>
    </p>

    <div class="society__grid">
      <section class="society__card" aria-labelledby="society-pipeline-title" :aria-busy="isActive">
        <header class="society__card-header">
          <h3 id="society-pipeline-title">Pipeline trace</h3>
          <span>{{ steps.length }} {{ steps.length === 1 ? 'step' : 'steps' }}</span>
        </header>

        <div v-if="isActive && steps.length === 0" class="society__empty">
          <span class="society__spinner" aria-hidden="true" />
          <span>Waiting for backend trace steps…</span>
        </div>
        <ol v-else-if="steps.length" class="society__steps">
          <li v-for="step in steps" :key="`${step.node}-${step.agent ?? 'agent-unavailable'}`" class="society__step">
            <span
              class="society__step-dot"
              :class="`society__step-dot--${stepTone(step.status)}`"
              aria-hidden="true"
            />
            <div class="society__step-name">
              <strong>{{ step.node }}</strong>
              <span>{{ step.agent ?? 'Agent unavailable' }}</span>
            </div>
            <span class="society__step-status">{{ step.status ?? 'Status unavailable' }}</span>
            <span v-if="step.error || step.reason" class="society__step-detail">
              {{ step.error ?? step.reason }}
            </span>
          </li>
        </ol>
        <p v-else class="society__empty">
          {{ phase === 'stopped'
            ? 'Live tracking stopped before the backend returned trace steps.'
            : 'No backend trace steps are available.' }}
        </p>
      </section>

      <section class="society__card" aria-labelledby="society-events-title" :aria-busy="isActive">
        <header class="society__card-header">
          <h3 id="society-events-title">Live trace</h3>
          <span>{{ events.length }} {{ events.length === 1 ? 'event' : 'events' }}</span>
        </header>

        <ol v-if="events.length" class="society__events" role="log" aria-live="polite" aria-relevant="additions">
          <li v-for="(event, index) in events" :key="`${event.timestamp ?? 'no-time'}-${eventType(event)}-${index}`" class="society__event">
            <div>
              <strong>{{ eventType(event) }}</strong>
              <time v-if="event.timestamp" :datetime="event.timestamp">{{ formatTimestamp(event.timestamp) }}</time>
            </div>
            <span>{{ eventLabel(event) }}</span>
          </li>
        </ol>
        <div v-else class="society__empty">
          <span v-if="isActive" class="society__spinner" aria-hidden="true" />
          <span v-if="isActive">
            {{ transport === 'polling' ? 'Polling the backend for trace events…' : 'Waiting for backend trace events…' }}
          </span>
          <span v-else-if="phase === 'stopped'">Live tracking stopped before an event arrived.</span>
          <span v-else-if="phase === 'done' || phase === 'failed'">The backend returned no trace events.</span>
          <span v-else>No trace events are available yet.</span>
        </div>
      </section>
    </div>

    <section class="society__card" aria-labelledby="society-report-title" :aria-busy="isActive">
      <header class="society__card-header">
        <h3 id="society-report-title">Mechanistic report</h3>
        <a
          v-if="figureUrl && figurePatch"
          class="society__inline-link"
          :href="figureUrl"
          target="_blank"
          rel="noopener noreferrer"
        >
          Export attention figure L{{ figurePatch.layer }}H{{ figurePatch.head }} (PNG)
          <span aria-hidden="true">↗</span>
          <span class="society__sr-only"> (opens in a new tab)</span>
        </a>
      </header>

      <div v-if="isActive" class="society__empty">
        <span class="society__spinner" aria-hidden="true" />
        <span>Waiting for the backend report…</span>
      </div>
      <pre v-else-if="reportMarkdown && reportMarkdown.trim()" class="society__report">{{ reportMarkdown }}</pre>
      <p v-else-if="publicationBlockReason" class="society__empty">
        Publication blocked: {{ publicationBlockReason }}
      </p>
      <p v-else-if="reportMarkdown !== null" class="society__empty">
        The backend returned an empty mechanistic report.
      </p>
      <p v-else class="society__empty">
        {{ phase === 'stopped'
          ? 'A report is unavailable because live tracking stopped before the backend result arrived.'
          : 'No mechanistic report is available for this run.' }}
      </p>
    </section>

    <section class="society__card" aria-labelledby="society-gate-title" :aria-busy="isActive">
      <header class="society__card-header">
        <h3 id="society-gate-title">{{ gateHeading }}</h3>
      </header>

      <div v-if="isActive" class="society__empty">
        <span class="society__spinner" aria-hidden="true" />
        <span>Waiting for the backend validation gate…</span>
      </div>
      <div v-else-if="gate" class="society__gate">
        <span class="society__gate-verdict" :class="`society__gate-verdict--${gateTone}`">
          {{ gateVerdictLabel }}
        </span>
        <dl>
          <div>
            <dt>Fidelity</dt>
            <dd>{{ gateFidelity }}</dd>
          </div>
          <div>
            <dt>Confidence</dt>
            <dd>{{ gateConfidence }}</dd>
          </div>
          <div>
            <dt>Validated</dt>
            <dd>{{ gateValidated }}</dd>
          </div>
        </dl>
        <p v-if="gatePassed === false" class="society__help">
          The backend gate did not pass. Its reproducibility report contains the failing metrics.
        </p>
        <p v-else-if="gatePassed === null" class="society__help">
          The backend returned a gate but did not provide a pass/fail verdict.
        </p>
      </div>
      <p v-else class="society__empty">
        {{ phase === 'stopped'
          ? 'Validation is unavailable because live tracking stopped before the backend result arrived.'
          : 'No validation gate is available for this run.' }}
      </p>
    </section>
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
type Transport = 'idle' | 'stream' | 'polling';
type BackendState = 'unknown' | 'online' | 'offline';
type StatusTone = 'idle' | 'running' | 'success' | 'error' | 'offline';

interface TraceStep {
  node: string;
  agent?: string;
  status?: string;
  reason?: string;
  error?: string;
  layer?: number;
  head?: number;
}

const API_BASE = apiUrl('/api');
const SOCIETY_BASE = apiUrl('/api/society');
const POLL_INTERVAL_MS = 2000;
const runHistoryUrl = `${SOCIETY_BASE}/runs`;

const goal = ref('');
const phase = ref<Phase>('idle');
const runId = ref<string | null>(null);
const runGoal = ref('');
const runStatus = ref<string | null>(null);
const events = ref<SocietyEvent[]>([]);
const steps = ref<TraceStep[]>([]);
const reportMarkdown = ref<string | null>(null);
const publication = ref<Record<string, unknown> | null>(null);
const gate = ref<Record<string, unknown> | null>(null);
const errorMessage = ref('');
const validationMessage = ref('');
const transport = ref<Transport>('idle');
const backendState = ref<BackendState>('unknown');
const browserOnline = ref(typeof navigator === 'undefined' ? true : navigator.onLine);
const pollingNotice = ref('');

let unsubscribeSse: (() => void) | null = null;
let pollTimer: number | null = null;
let pollingInFlight = false;
let requestGeneration = 0;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asText(value: unknown, limit = Number.POSITIVE_INFINITY): string | undefined {
  if (typeof value === 'string') return value.slice(0, limit);
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return undefined;
}

function asNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function asIndex(value: unknown): number | undefined {
  const parsed = asNumber(value);
  return parsed !== undefined && Number.isInteger(parsed) && parsed >= 0 ? parsed : undefined;
}

function errorText(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return asText(error) ?? 'Unknown backend error';
}

function eventType(event: SocietyEvent): string {
  return asText(event.event_type) ?? 'Unknown event';
}

function eventLabel(event: SocietyEvent): string {
  const payload = isRecord(event.payload) ? event.payload : {};
  switch (eventType(event)) {
    case 'ResearchStarted':
      return `Goal: ${asText(payload.goal, 80) ?? 'unavailable'}`;
    case 'ExperimentQueued': {
      if (Array.isArray(payload.plan)) {
        const nodes = payload.plan.map((node) => asText(node)).filter((node): node is string => Boolean(node));
        return nodes.length ? `Plan queued: ${nodes.join(' → ')}` : 'Plan queued (node names unavailable)';
      }
      if (payload.replan !== undefined) {
        const failed = Array.isArray(payload.failed)
          ? payload.failed.map((node) => asText(node)).filter((node): node is string => Boolean(node)).join(', ')
          : asText(payload.failed);
        return `Replan #${asText(payload.replan) ?? '?'} after: ${failed ?? 'reason unavailable'}`;
      }
      return 'Experiment queued';
    }
    case 'DiscoveryCreated':
      return `Discovery ${asText(payload.discovery_id) ?? '(ID unavailable)'}`;
    case 'HypothesisRejected':
      return `Stage failed: ${asText(payload.node) ?? 'node unavailable'} — ${asText(payload.reason, 100) ?? 'reason unavailable'}`;
    case 'CircuitValidated': {
      if (payload.status === 'blocked' || payload.reason) {
        return `Validation blocked: ${asText(payload.reason) || 'live evidence unavailable'}`;
      }
      const successful = Array.isArray(payload.successful)
        ? payload.successful.map((node) => asText(node)).filter((node): node is string => Boolean(node))
        : [];
      return successful.length ? `Validated: ${successful.join(', ')}` : 'Validation event received';
    }
    case 'PublicationGenerated':
      return `Report ${asText(payload.experiment_id) ?? '(experiment ID unavailable)'} published`;
    case 'ResearchFinished':
      return `Finished (${asText(payload.steps_completed) ?? 'step count unavailable'} steps)`;
    default:
      return eventType(event);
  }
}

function eventKey(event: SocietyEvent): string {
  let payload = '';
  try {
    payload = JSON.stringify(event.payload ?? {});
  } catch {
    payload = 'unserializable-payload';
  }
  return `${event.timestamp ?? ''}:${eventType(event)}:${payload}`;
}

function mergeEvents(incoming: unknown): void {
  if (!Array.isArray(incoming)) return;
  const merged = new Map<string, SocietyEvent>();
  for (const event of [...events.value, ...incoming]) {
    if (!isRecord(event)) continue;
    const normalized: SocietyEvent = {
      event_type: asText(event.event_type) ?? 'Unknown event',
      payload: isRecord(event.payload) ? event.payload : {},
      ...(typeof event.timestamp === 'string' ? { timestamp: event.timestamp } : {}),
    };
    merged.set(eventKey(normalized), normalized);
  }
  events.value = [...merged.values()].slice(-1000);
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? timestamp : date.toLocaleTimeString();
}

function normalizeTrace(value: unknown): TraceStep[] {
  if (!Array.isArray(value)) return [];
  const normalized: TraceStep[] = [];
  for (const item of value) {
    if (!isRecord(item)) continue;
    const node = asText(item.node);
    if (!node) continue;
    normalized.push({
      node,
      agent: asText(item.agent),
      status: asText(item.status),
      reason: asText(item.reason),
      error: asText(item.error),
      layer: asIndex(item.layer),
      head: asIndex(item.head),
    });
  }
  return normalized;
}

function isTrackingRun(id: string): boolean {
  return phase.value === 'running' && runId.value === id;
}

function stopTransport(): void {
  unsubscribeSse?.();
  unsubscribeSse = null;
  if (pollTimer !== null) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
  transport.value = 'idle';
  pollingNotice.value = '';
}

function applyDone(id: string, rawResult: unknown): void {
  if (!isRecord(rawResult)) {
    beginPolling(id, 'The live stream ended without a readable result. Checking backend status…');
    return;
  }

  stopTransport();
  backendState.value = 'online';
  runId.value = id;
  runGoal.value = asText(rawResult.goal) ?? runGoal.value;
  runStatus.value = asText(rawResult.status) ?? null;
  steps.value = normalizeTrace(rawResult.trace);
  mergeEvents(rawResult.events);
  publication.value = isRecord(rawResult.publication) ? rawResult.publication : null;

  const report = publication.value?.report ?? rawResult.report;
  if (typeof report === 'string') {
    reportMarkdown.value = report;
  } else if (isRecord(report) && typeof report.markdown === 'string') {
    reportMarkdown.value = report.markdown;
  } else {
    reportMarkdown.value = null;
  }

  const rawGate = publication.value?.gate ?? rawResult.gate;
  gate.value = isRecord(rawGate) && Object.keys(rawGate).length > 0 ? rawGate : null;

  const resultError = asText(rawResult.error)
    ?? (publication.value?.status === 'blocked' ? asText(publication.value.reason) : undefined)
    ?? asText(rawResult.reason);
  const failedStatus = ['error', 'failed', 'cancelled', 'blocked'].includes(runStatus.value ?? '');
  if (failedStatus || publication.value?.status === 'blocked' || (resultError && runStatus.value !== 'completed')) {
    phase.value = 'failed';
    errorMessage.value = resultError ?? `The backend ended this run with status “${runStatus.value ?? 'unavailable'}”.`;
  } else {
    phase.value = 'done';
    errorMessage.value = resultError ?? '';
  }
}

async function pollRun(id: string): Promise<void> {
  if (!isTrackingRun(id) || pollingInFlight) return;
  pollingInFlight = true;
  try {
    const status = await getSocietyRun(id);
    if (!isTrackingRun(id)) return;
    backendState.value = 'online';
    runStatus.value = asText(status.status) ?? runStatus.value;
    errorMessage.value = '';
    mergeEvents(status.events);
    if (status.status !== 'running') {
      if (isRecord(status.result)) {
        applyDone(id, status.result);
      } else {
        stopTransport();
        phase.value = 'failed';
        runStatus.value = asText(status.status) ?? null;
        const statusError = isRecord(status) ? asText(status.error) : undefined;
        errorMessage.value = statusError ?? 'The backend run ended without a result payload.';
      }
    }
  } catch (error) {
    if (!isTrackingRun(id)) return;
    backendState.value = 'offline';
    errorMessage.value = `Could not read run status: ${errorText(error)}`;
  } finally {
    pollingInFlight = false;
  }
}

function beginPolling(id: string, notice = 'Live stream unavailable; polling the backend every 2 seconds.'): void {
  if (!isTrackingRun(id)) return;
  unsubscribeSse?.();
  unsubscribeSse = null;
  if (pollTimer !== null) return;
  transport.value = 'polling';
  pollingNotice.value = notice;
  pollTimer = window.setInterval(() => {
    void pollRun(id);
  }, POLL_INTERVAL_MS);
  void pollRun(id);
}

function attachRun(id: string): void {
  if (!isTrackingRun(id)) return;
  transport.value = 'stream';
  try {
    unsubscribeSse = streamSocietyRun(id, {
      onEvent: (event) => {
        if (!isTrackingRun(id)) return;
        backendState.value = 'online';
        mergeEvents([event]);
      },
      onDone: (result) => {
        if (!isTrackingRun(id)) return;
        if (isRecord(result) && Object.keys(result).length === 0) {
          beginPolling(id, 'The live stream ended without a result payload. Checking backend status…');
          return;
        }
        applyDone(id, result);
      },
      onError: () => {
        if (!isTrackingRun(id)) return;
        beginPolling(id);
      },
    });
  } catch (error) {
    errorMessage.value = `Live stream could not start: ${errorText(error)}`;
    beginPolling(id);
  }
}

async function handleRun(): Promise<void> {
  const submittedGoal = goal.value.trim();
  if (!submittedGoal) {
    validationMessage.value = 'Enter a research goal before running the Society.';
    return;
  }
  if (!browserOnline.value) {
    errorMessage.value = 'This browser is offline. Reconnect before starting a Society run.';
    return;
  }
  if (isActive.value) return;

  const generation = ++requestGeneration;
  stopTransport();
  phase.value = 'starting';
  runId.value = null;
  runGoal.value = submittedGoal;
  runStatus.value = null;
  events.value = [];
  steps.value = [];
  reportMarkdown.value = null;
  publication.value = null;
  gate.value = null;
  errorMessage.value = '';
  validationMessage.value = '';

  try {
    const started = await startSocietyRun(submittedGoal);
    if (generation !== requestGeneration || phase.value !== 'starting') return;
    runId.value = started.runId;
    runStatus.value = asText(started.status) ?? null;
    phase.value = 'running';
    backendState.value = 'online';
    attachRun(started.runId);
  } catch (error) {
    if (generation !== requestGeneration || phase.value !== 'starting') return;
    backendState.value = 'offline';
    phase.value = 'failed';
    runStatus.value = null;
    errorMessage.value = `Could not start the Society: ${errorText(error)}`;
  }
}

function handleStop(): void {
  if (!isActive.value) return;
  requestGeneration += 1;
  stopTransport();
  phase.value = 'stopped';
  errorMessage.value = '';
}

function handleOnline(): void {
  browserOnline.value = true;
  if (backendState.value === 'offline') backendState.value = 'unknown';
}

function handleOffline(): void {
  browserOnline.value = false;
  backendState.value = 'offline';
}

const isActive = computed(() => phase.value === 'starting' || phase.value === 'running');
const canRun = computed(() => Boolean(goal.value.trim()) && browserOnline.value && !isActive.value);

const backendStateLabel = computed(() => {
  if (!browserOnline.value) return 'Browser offline';
  if (backendState.value === 'online') return 'Backend reachable';
  if (backendState.value === 'offline') return 'Backend unavailable';
  return 'Backend status unknown';
});

const offlineNotice = computed(() => {
  if (!browserOnline.value) return 'Reconnect this browser to start or update a Society run.';
  if (backendState.value !== 'offline') return '';
  return `The backend at ${API_BASE} is not responding. Existing backend data remains visible.`;
});

const stopHelp = computed(() => {
  if (phase.value === 'starting') {
    return 'Stop ends this panel’s wait. The backend has no run-cancellation endpoint, so a submitted run may continue.';
  }
  if (phase.value === 'running') {
    return 'Stop ends live updates in this panel only. The backend run may continue because this API has no cancellation endpoint.';
  }
  return 'Run starts a new backend workflow. Stop is available only while this panel is waiting or receiving events.';
});

const statusTone = computed<StatusTone>(() => {
  if (!browserOnline.value || backendState.value === 'offline') return 'offline';
  if (phase.value === 'failed') return 'error';
  if (phase.value === 'done') return 'success';
  if (phase.value === 'starting' || phase.value === 'running') return 'running';
  return 'idle';
});

const statusHeading = computed(() => {
  if (!browserOnline.value) return 'Browser offline';
  switch (phase.value) {
    case 'starting': return 'Starting backend run';
    case 'running': return transport.value === 'polling' ? 'Polling backend status' : 'Society run in progress';
    case 'stopped': return 'Live tracking stopped';
    case 'done': return runStatus.value === 'completed' ? 'Society run completed' : 'Society run ended';
    case 'failed': return 'Society run failed';
    default: return 'Ready for a research goal';
  }
});

const statusDetail = computed(() => {
  if (!browserOnline.value) return 'No backend requests can be sent until connectivity returns.';
  switch (phase.value) {
    case 'starting': return 'Submitting the goal to the Society run endpoint.';
    case 'running': return pollingNotice.value || 'Listening for real backend trace events.';
    case 'stopped': return runId.value
      ? 'The backend run was not cancelled; this panel is no longer receiving updates.'
      : 'This panel stopped waiting; any run created by the in-flight request was not cancelled.';
    case 'done': return runStatus.value === 'completed' ? 'Backend result received.' : 'The backend stream ended, but no completion status was provided.';
    case 'failed': return errorMessage.value || 'No successful backend result was returned.';
    default: return `Backend status is unknown until a request reaches ${SOCIETY_BASE}.`;
  }
});

const publicationBlockReason = computed(() => (
  publication.value?.status === 'blocked'
    ? asText(publication.value.reason) ?? 'Live evidence is unavailable.'
    : ''
));

const runSummary = computed(() => {
  const parts: string[] = [];
  const completed = publication.value ? asText(publication.value.steps_completed) : undefined;
  const experimentId = publication.value ? asText(publication.value.experiment_id) : undefined;
  if (completed) parts.push(`${completed} steps`);
  if (experimentId) parts.push(experimentId);
  return parts.join(' · ');
});

const figurePatch = computed(() => steps.value.find((step) => (
  step.node === 'patch' && step.layer !== undefined && step.head !== undefined
)));

const figureUrl = computed(() => {
  if (!figurePatch.value || !runGoal.value) return '';
  const url = new URL(apiUrl('/api/figures/attention'), window.location.origin);
  url.searchParams.set('prompt', runGoal.value);
  url.searchParams.set('layer', String(figurePatch.value.layer));
  url.searchParams.set('head', String(figurePatch.value.head));
  return url.toString();
});

const gateHeading = computed(() => {
  const threshold = gate.value ? asNumber(gate.value.threshold) : undefined;
  if (threshold === undefined) return 'Validation gate (threshold unavailable)';
  const percentage = threshold <= 1 ? threshold * 100 : threshold;
  return `Validation gate (${percentage}% threshold)`;
});

const gatePassed = computed<boolean | null>(() => {
  if (!gate.value) return null;
  return typeof gate.value.passed === 'boolean' ? gate.value.passed : null;
});

const gateTone = computed(() => gatePassed.value === true ? 'passed' : gatePassed.value === false ? 'failed' : 'unknown');
const gateVerdictLabel = computed(() => {
  if (gatePassed.value === true) return 'Passed';
  if (gatePassed.value === false) return 'Failed';
  return 'Verdict unavailable';
});

const gateFidelity = computed(() => {
  const value = gate.value ? asNumber(gate.value.value) : undefined;
  return value === undefined ? 'Unavailable' : `${value}%`;
});

const gateConfidence = computed(() => {
  const value = gate.value ? asNumber(gate.value.confidence) : undefined;
  return value === undefined ? 'Unavailable' : value.toFixed(2);
});

const gateValidated = computed(() => {
  if (!gate.value || gate.value.status === 'blocked') return 'Unavailable';
  if (typeof gate.value.validated === 'boolean') return gate.value.validated ? 'Yes' : 'No';
  return 'Unavailable';
});

function stepTone(status: string | undefined): StatusTone {
  if (status === 'completed' || status === 'ok' || status === 'loaded') return 'success';
  if (status === 'running') return 'running';
  if (status === 'error' || status === 'failed' || status === 'unavailable' || status === 'blocked') return 'error';
  return 'idle';
}

onMounted(() => {
  window.addEventListener('online', handleOnline);
  window.addEventListener('offline', handleOffline);
});

onBeforeUnmount(() => {
  requestGeneration += 1;
  stopTransport();
  window.removeEventListener('online', handleOnline);
  window.removeEventListener('offline', handleOffline);
});
</script>

<style scoped>
.society {
  display: flex;
  flex-direction: column;
  gap: 14px;
  width: 100%;
  max-width: 1120px;
  color: var(--text);
  font-size: 13px;
}

.society__header,
.society__card-header,
.society__status,
.society__run-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.society h2,
.society h3,
.society p {
  margin: 0;
}

.society h2 {
  font-size: 18px;
  line-height: 1.3;
  font-weight: 650;
}

.society h3 {
  font-size: 13px;
  line-height: 1.4;
  font-weight: 650;
}

.society__intro,
.society__help,
.society__empty {
  color: var(--text-muted);
}

.society__intro {
  max-width: 720px;
  margin-top: 4px !important;
  line-height: 1.5;
}

.society__history-link,
.society__inline-link {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-height: 32px;
  color: var(--text);
  font-weight: 600;
  text-decoration: none;
  white-space: nowrap;
}

.society__history-link:hover,
.society__inline-link:hover {
  text-decoration: underline;
  text-underline-offset: 3px;
}

.society :is(button, input, a):focus-visible {
  outline: 2px solid var(--text);
  outline-offset: 2px;
}

.society__run-card,
.society__card,
.society__status,
.society__notice {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-elev);
}

.society__run-card,
.society__card {
  padding: 14px;
}

.society__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.society label {
  font-size: 11px;
  font-weight: 650;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.society__controls {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  gap: 8px;
}

.society__goal {
  min-height: 36px;
}

.society__button {
  min-height: 36px;
  padding: 7px 13px;
  border: 1px solid var(--border-light);
  border-radius: var(--radius);
  background: var(--bg);
  color: var(--text);
  font: inherit;
  font-size: 12px;
  font-weight: 600;
}

.society__button:hover:not(:disabled) {
  background: var(--bg-hover);
}

.society__button--primary {
  border-color: var(--text);
  background: var(--text);
  color: var(--bg);
}

.society__button--primary:hover:not(:disabled) {
  background: var(--text-dim);
  color: var(--bg);
}

.society__button:disabled {
  cursor: not-allowed;
  opacity: 0.42;
}

.society__help,
.society__error-text {
  font-size: 11px;
  line-height: 1.45;
}

.society__error-text {
  color: var(--text);
  font-weight: 600;
}

.society__status {
  min-height: 48px;
  padding: 10px 12px;
}

.society__status > div {
  display: flex;
  flex: 1;
  min-width: 0;
  flex-wrap: wrap;
  gap: 3px 8px;
  line-height: 1.4;
}

.society__status > div span {
  color: var(--text-muted);
}

.society__status-dot,
.society__step-dot {
  display: inline-block;
  flex: 0 0 auto;
  border-radius: 999px;
  background: var(--text-muted);
}

.society__status-dot {
  width: 9px;
  height: 9px;
}

.society__status-dot--running,
.society__step-dot--running {
  background: var(--text);
  animation: society-pulse 1.2s ease-in-out infinite;
}

.society__status-dot--success,
.society__step-dot--success {
  background: var(--text);
  box-shadow: inset 0 0 0 2px var(--bg), 0 0 0 1px var(--text);
}

.society__status-dot--error,
.society__status-dot--offline,
.society__step-dot--error {
  background: var(--bg);
  box-shadow: inset 0 0 0 2px var(--text);
}

.society__backend-state {
  flex: 0 0 auto;
  padding: 2px 8px;
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--text-muted);
  font-size: 10px;
  white-space: nowrap;
}

.society__notice {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 10px 12px;
  line-height: 1.45;
}

.society__notice--warning,
.society__notice--error {
  border-left: 3px solid var(--text);
  background: var(--bg-elev-2);
}

.society__run-meta {
  justify-content: flex-start;
  flex-wrap: wrap;
  color: var(--text-muted);
  font-size: 11px;
}

.society code,
.society__report {
  font-family: var(--font-mono);
}

.society__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.society__card-header {
  min-height: 28px;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

.society__card-header > span {
  color: var(--text-muted);
  font-size: 10px;
  white-space: nowrap;
}

.society__empty {
  display: flex;
  min-height: 72px;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 16px;
  text-align: center;
  line-height: 1.45;
}

.society__spinner {
  display: inline-block;
  width: 13px;
  height: 13px;
  flex: 0 0 auto;
  border: 2px solid var(--border-light);
  border-top-color: var(--text);
  border-radius: 999px;
  animation: society-spin 0.7s linear infinite;
}

.society__steps,
.society__events {
  display: flex;
  flex-direction: column;
  gap: 7px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.society__step {
  display: grid;
  grid-template-columns: 9px minmax(90px, 1fr) auto;
  align-items: center;
  gap: 7px 9px;
  min-width: 0;
  padding: 7px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg);
}

.society__step-dot {
  width: 8px;
  height: 8px;
}

.society__step-name {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  gap: 3px 7px;
}

.society__step-name strong {
  overflow-wrap: anywhere;
}

.society__step-name span,
.society__step-status,
.society__step-detail {
  color: var(--text-muted);
  font-size: 10px;
}

.society__step-status {
  white-space: nowrap;
}

.society__step-detail {
  grid-column: 2 / -1;
  overflow-wrap: anywhere;
}

.society__events {
  max-height: 300px;
  overflow-y: auto;
}

.society__event {
  display: grid;
  grid-template-columns: minmax(120px, 0.42fr) minmax(0, 1fr);
  gap: 10px;
  padding: 7px 8px;
  border-bottom: 1px solid var(--border);
  line-height: 1.4;
}

.society__event:last-child {
  border-bottom: 0;
}

.society__event > div {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 1px;
}

.society__event strong,
.society__event > span {
  overflow-wrap: anywhere;
}

.society__event time,
.society__event > span {
  color: var(--text-muted);
  font-size: 10px;
}

.society__report {
  max-height: 460px;
  margin: 0;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  color: var(--text-dim);
  font-size: 11px;
  line-height: 1.6;
}

.society__gate {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 10px;
}

.society__gate-verdict {
  display: inline-flex;
  min-height: 26px;
  align-items: center;
  padding: 3px 10px;
  border: 1px solid var(--text);
  border-radius: 999px;
  color: var(--text);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.society__gate-verdict--failed,
.society__gate-verdict--unknown {
  border-style: dashed;
}

.society__gate dl {
  display: grid;
  width: 100%;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin: 0;
}

.society__gate dl > div {
  padding: 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg);
}

.society__gate dt {
  color: var(--text-muted);
  font-size: 9px;
  font-weight: 650;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.society__gate dd {
  margin: 3px 0 0;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 12px;
}

.society__sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@keyframes society-spin {
  to { transform: rotate(360deg); }
}

@keyframes society-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}

@media (max-width: 760px) {
  .society__header,
  .society__status {
    align-items: flex-start;
  }

  .society__header {
    flex-direction: column;
  }

  .society__controls,
  .society__grid {
    grid-template-columns: 1fr;
  }

  .society__button {
    width: 100%;
  }

  .society__status {
    flex-wrap: wrap;
  }

  .society__backend-state {
    margin-left: 21px;
  }

  .society__event {
    grid-template-columns: 1fr;
    gap: 3px;
  }
}

@media (max-width: 480px) {
  .society__gate dl {
    grid-template-columns: 1fr;
  }
}

@media (prefers-reduced-motion: reduce) {
  .society__spinner,
  .society__status-dot--running,
  .society__step-dot--running {
    animation: none;
  }
}
</style>
