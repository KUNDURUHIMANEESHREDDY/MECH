<template>
  <section class="jobs-surface" data-testid="jobs-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Develop / Runtime queue</p>
        <h2 class="surface-title">Jobs</h2>
        <p class="surface-description">
          The backend's persisted job queue (<code>GET /api/v1/jobs</code>). Rows are the
          backend's own records: status, timestamps, error and result are shown as returned.
          The queue reports no progress, ETA or stage, so none is displayed.
        </p>
      </div>
      <div class="header-actions">
        <span class="surface-status" :class="{ 'is-connected': isConnected }" role="status" aria-live="polite" data-testid="jobs-status">
          <span class="status-dot" aria-hidden="true" />
          {{ isLoading ? 'Reading queue' : isConnected ? 'Queue reachable' : 'Queue unavailable' }}
        </span>
        <button class="quiet-button" type="button" data-testid="jobs-refresh" :disabled="isLoading" @click="loadJobs()">
          {{ isLoading ? 'Refreshing…' : 'Refresh queue' }}
        </button>
      </div>
    </header>

    <div
      v-if="failure"
      class="notice"
      :class="failure.kind === 'error' ? 'notice-error' : 'notice-offline'"
      role="alert"
      :data-testid="`jobs-${failure.kind}`"
    >
      <strong>{{ failureTitle }}</strong>
      <p>{{ failure.message }}</p>
      <p class="notice-foot">
        No local job list, cached row or sample entry is substituted. Polling is stopped until a
        request succeeds.
      </p>
      <button class="quiet-button" type="button" data-testid="jobs-retry" :disabled="isLoading" @click="loadJobs()">
        {{ isLoading ? 'Retrying…' : 'Retry' }}
      </button>
    </div>

    <p v-else-if="isLoading && !hasLoaded" class="notice" role="status" aria-live="polite" data-testid="jobs-loading">
      <strong>Reading the backend job queue…</strong>
      <p>Listing persisted jobs; this view waits for the backend's answer.</p>
    </p>

    <section v-if="hasLoaded && !failure" class="panel bounds-panel" aria-labelledby="jobs-bounds-title">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">Reported bounds</p>
          <h3 id="jobs-bounds-title">Queue limits and storage</h3>
          <p class="panel-description">
            Facts from the listing response's <code>bounds</code> object; not client-side guesses.
          </p>
        </div>
        <span class="source-badge" data-testid="jobs-provenance">PROVENANCE: {{ provenance || 'NOT RETURNED' }}</span>
      </header>

      <dl v-if="bounds" class="bounds-list" data-testid="jobs-bounds">
        <div>
          <dt>max_concurrency</dt>
          <dd class="mono">{{ numberOrNotReturned(bounds.max_concurrency) }}</dd>
        </div>
        <div>
          <dt>max_queued</dt>
          <dd class="mono">{{ numberOrNotReturned(bounds.max_queued) }}</dd>
        </div>
        <div class="bounds-dir">
          <dt>job_dir</dt>
          <dd class="mono break">{{ bounds.job_dir || 'not returned' }}</dd>
        </div>
        <div>
          <dt>kinds reported by the backend</dt>
          <dd class="mono">{{ kindsReported.length ? kindsReported.join(', ') : 'not returned' }}</dd>
        </div>
      </dl>
      <p v-else class="panel-message">The backend did not return a <code>bounds</code> object for this listing.</p>
    </section>

    <section v-if="hasLoaded && !failure" class="panel jobs-panel" aria-labelledby="jobs-list-title">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">Persisted records</p>
          <h3 id="jobs-list-title">{{ jobs.length }} job(s) in this listing</h3>
          <p class="panel-description">
            Newest first, in the backend's order. A job counts as finished only when the backend
            reports a terminal status.
          </p>
        </div>
        <span class="source-badge">SOURCE: API</span>
      </header>

      <p v-if="countMismatch" class="notice notice-warn" role="status" data-testid="jobs-count-mismatch">
        The backend reported <code>count: {{ count }}</code> but returned
        {{ jobs.length }} readable record(s). Nothing was added to fill the difference.
      </p>

      <p v-if="isLoading && jobs.length" class="panel-message" role="status">Refreshing the listing…</p>

      <ol v-if="jobs.length" class="job-list" data-testid="jobs-list" aria-label="Jobs returned by the backend">
        <li v-for="job in jobs" :key="job.job_id" class="job-item" :data-testid="`job-${job.job_id}`">
          <div class="job-head">
            <div class="job-title">
              <span class="job-kind">{{ labelForKind(job.kind) }}</span>
              <code class="job-id">{{ job.job_id }}</code>
            </div>
            <div class="job-head-actions">
              <span class="status-badge" :class="statusClass(job.status)" :data-testid="`job-status-${job.job_id}`">
                {{ job.status || 'not returned' }}
              </span>
              <button
                v-if="!isTerminal(job.status)"
                class="quiet-button"
                type="button"
                :data-testid="`job-cancel-${job.job_id}`"
                :disabled="cancellingId === job.job_id"
                :aria-label="`Request cancellation of job ${job.job_id}`"
                @click="requestCancel(job)"
              >
                {{ cancellingId === job.job_id ? 'Requesting…' : 'Cancel job' }}
              </button>
            </div>
          </div>

          <dl class="job-facts">
            <div>
              <dt>Created</dt>
              <dd>{{ formatTimestamp(job.created_at) }}</dd>
            </div>
            <div>
              <dt>Started</dt>
              <dd>{{ formatTimestamp(job.started_at) }}</dd>
            </div>
            <div>
              <dt>Finished</dt>
              <dd>{{ formatTimestamp(job.finished_at) }}</dd>
            </div>
            <div>
              <dt>Cancellation requested</dt>
              <dd>{{ job.cancel_requested ? 'yes' : 'no' }}</dd>
            </div>
            <div class="job-params">
              <dt>Submitted params</dt>
              <dd class="mono break">{{ paramsText(job) }}</dd>
            </div>
          </dl>

          <p v-if="job.persist_error" class="job-warning" role="status">
            <strong>Persistence error recorded by the backend:</strong> {{ job.persist_error }}
          </p>

          <div v-if="job.error" class="job-error" role="status" data-testid="job-error">
            <strong>{{ isTerminal(job.status) ? 'Backend reason' : 'Backend note' }}</strong>
            <p>{{ job.error }}</p>
          </div>

          <details v-if="hasResult(job)" class="job-result">
            <summary>Result returned by the backend</summary>
            <pre>{{ resultText(job) }}</pre>
          </details>
        </li>
      </ol>

      <p v-else class="panel-message" data-testid="jobs-empty">
        No jobs have been submitted. The backend returned an empty queue; nothing is listed locally.
      </p>

      <footer v-if="lastRefreshed" class="panel-footnote">
        Last read <time :datetime="lastRefreshed">{{ formatTimestamp(lastRefreshed) }}</time>.
        <span v-if="hasActiveJobs">Re-reading every 2 s while a job is queued or running.</span>
      </footer>
    </section>

    <section class="panel submit-panel" aria-labelledby="jobs-submit-title">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">Submit work</p>
          <h3 id="jobs-submit-title">Queue a job</h3>
          <p class="panel-description">
            Only the two kinds the backend implements are offered. Inputs are validated with the
            backend's own rules before submission; the backend remains the authority.
          </p>
        </div>
      </header>

      <p v-if="failure" class="panel-message" data-testid="jobs-submit-blocked">
        Submission is disabled while the queue cannot be read. A job is never queued locally.
      </p>

      <form class="entry-form" @submit.prevent="submitJobForm">
        <div class="field">
          <label for="jobs-kind">Kind</label>
          <select id="jobs-kind" v-model="kind" data-testid="jobs-kind" :disabled="Boolean(failure) || isSubmitting">
            <option v-for="option in JOB_KINDS" :key="option" :value="option">{{ labelForKind(option) }} ({{ option }})</option>
          </select>
        </div>

        <div v-if="kind === 'benchmark_sweep'" class="field">
          <label for="jobs-benchmarks">Benchmark names <span class="required-mark" aria-hidden="true">*</span></label>
          <input
            id="jobs-benchmarks"
            v-model="benchmarksText"
            type="text"
            autocomplete="off"
            placeholder="Comma-separated names, e.g. IOI, GreaterThan"
            :disabled="Boolean(failure) || isSubmitting"
            data-testid="jobs-benchmarks"
          >
          <p class="field-hint">The backend requires a non-empty list (<code>benchmarks</code>); at most 16 names are run.</p>
        </div>

        <div v-if="kind === 'benchmark_sweep'" class="field">
          <label for="jobs-n-prompts">n_prompts <span class="optional-label">optional</span></label>
          <input
            id="jobs-n-prompts"
            v-model="nPromptsText"
            type="number"
            min="1"
            max="1000"
            step="1"
            inputmode="numeric"
            placeholder="backend default (100)"
            :disabled="Boolean(failure) || isSubmitting"
            data-testid="jobs-n-prompts"
          >
          <p class="field-hint">1–1000. Left blank, the field is not sent and the backend applies its default.</p>
        </div>

        <div v-if="kind === 'ioi_stability'" class="field">
          <label for="jobs-n-seeds">n_seeds <span class="optional-label">optional</span></label>
          <input
            id="jobs-n-seeds"
            v-model="nSeedsText"
            type="number"
            min="1"
            max="10"
            step="1"
            inputmode="numeric"
            placeholder="backend default (5)"
            :disabled="Boolean(failure) || isSubmitting"
            data-testid="jobs-n-seeds"
          >
          <p class="field-hint">Integer 1–10. Left blank, the field is not sent and the backend applies its default.</p>
        </div>

        <div v-if="kind === 'ioi_stability'" class="field">
          <label for="jobs-n-prompts-ioi">n_prompts <span class="optional-label">optional</span></label>
          <input
            id="jobs-n-prompts-ioi"
            v-model="nPromptsText"
            type="number"
            min="1"
            max="1000"
            step="1"
            inputmode="numeric"
            placeholder="backend default (100)"
            :disabled="Boolean(failure) || isSubmitting"
            data-testid="jobs-n-prompts-ioi"
          >
          <p class="field-hint">1–1000. Left blank, the field is not sent and the backend applies its default.</p>
        </div>

        <div v-if="formError" class="notice notice-error" role="alert" data-testid="jobs-form-error">
          <strong>Not submitted</strong>
          <p>{{ formError }}</p>
        </div>

        <div class="form-actions">
          <button class="primary-button" type="submit" data-testid="jobs-submit" :disabled="Boolean(failure) || isSubmitting">
            {{ isSubmitting ? 'Submitting…' : 'Submit job' }}
          </button>
          <span class="form-hint">A 503 means the queue is full and the job was not accepted.</span>
        </div>
      </form>

      <div v-if="actionMessage" class="notice notice-success" role="status" aria-live="polite" data-testid="jobs-action-message">
        <p>{{ actionMessage }}</p>
      </div>
      <div v-if="actionError" class="notice notice-error" role="alert" data-testid="jobs-action-error">
        <strong>Job action refused by the backend</strong>
        <p>{{ actionError }}</p>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import {
  JOB_KINDS,
  cancelJob,
  describeJobsError,
  getJob,
  isActiveJobStatus,
  isJobsHttpError,
  isTerminalJobStatus,
  listJobs,
  submitJob,
  type JobBounds,
  type JobKind,
  type JobRecord,
} from '../services/jobs';

type FailureKind = 'offline' | 'unavailable' | 'error';

interface Failure {
  kind: FailureKind;
  message: string;
}

const POLL_INTERVAL_MS = 2000;

const jobs = ref<JobRecord[]>([]);
const bounds = ref<JobBounds | null>(null);
const kindsReported = ref<string[]>([]);
const count = ref<number | null>(null);
const provenance = ref('');
const failure = ref<Failure | null>(null);
const isLoading = ref(false);
const hasLoaded = ref(false);
const lastRefreshed = ref('');

const kind = ref<JobKind>('benchmark_sweep');
const benchmarksText = ref('');
const nPromptsText = ref('');
const nSeedsText = ref('');
const isSubmitting = ref(false);
const formError = ref('');
const actionMessage = ref('');
const actionError = ref('');
const cancellingId = ref('');

const isTerminal = isTerminalJobStatus;
const isConnected = computed(() => hasLoaded.value && failure.value === null);
const hasActiveJobs = computed(() => jobs.value.some((job) => isActiveJobStatus(job.status)));
const countMismatch = computed(() => count.value !== null && count.value !== jobs.value.length);
const failureTitle = computed(() => {
  if (!failure.value) return '';
  if (failure.value.kind === 'offline') return 'API unreachable';
  if (failure.value.kind === 'unavailable') return 'Job queue unavailable';
  return 'The job queue request was refused';
});

watch(hasActiveJobs, () => syncPolling());

onMounted(() => {
  void loadJobs();
});

onUnmounted(() => {
  stopPolling();
});

async function loadJobs(background = false): Promise<void> {
  if (!background) isLoading.value = true;
  try {
    const listing = await listJobs({ limit: 100 });
    jobs.value = listing.jobs;
    bounds.value = listing.bounds;
    kindsReported.value = listing.kinds;
    count.value = listing.count;
    provenance.value = listing.provenance;
    failure.value = null;
    hasLoaded.value = true;
    lastRefreshed.value = new Date().toISOString();
  } catch (error) {
    jobs.value = [];
    bounds.value = null;
    kindsReported.value = [];
    count.value = null;
    provenance.value = '';
    failure.value = classifyFailure(error);
    hasLoaded.value = true;
  } finally {
    isLoading.value = false;
    syncPolling();
  }
}

function classifyFailure(error: unknown): Failure {
  if (isJobsHttpError(error)) {
    if (error.isUnreachable) return { kind: 'offline', message: error.message };
    // A 5xx with no backend `detail` is the dev proxy reporting "connection
    // refused", not the queue answering. Say so instead of inventing a reason.
    if (error.status >= 500 && !error.detail) {
      return {
        kind: 'unavailable',
        message: `${error.message} — no detail was returned, so either the API is not running or it failed before answering.`,
      };
    }
    return { kind: 'error', message: error.message };
  }
  return { kind: 'offline', message: describeJobsError(error) };
}

let pollTimer: ReturnType<typeof setInterval> | null = null;

function syncPolling(): void {
  if (hasActiveJobs.value && failure.value === null) {
    if (pollTimer === null) {
      pollTimer = setInterval(() => {
        void loadJobs(true);
      }, POLL_INTERVAL_MS);
    }
    return;
  }
  stopPolling();
}

function stopPolling(): void {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function submitJobForm(): Promise<void> {
  formError.value = '';
  actionError.value = '';
  actionMessage.value = '';
  if (failure.value) {
    formError.value = 'The queue cannot be read, so nothing was submitted. Retry once the API answers.';
    return;
  }
  const params = buildParams();
  if (!params) return;

  isSubmitting.value = true;
  try {
    const result = await submitJob(kind.value, params);
    actionMessage.value = `The backend accepted job ${result.job.job_id} with status ${result.job.status || 'not returned'}.`;
    nPromptsText.value = '';
    nSeedsText.value = '';
    await loadJobs(true);
  } catch (error) {
    if (isJobsHttpError(error) && error.isQueueFull) {
      actionError.value = `The queue is full, so the job was NOT accepted. ${error.message}`;
    } else {
      actionError.value = describeJobsError(error);
    }
  } finally {
    isSubmitting.value = false;
    syncPolling();
  }
}

function buildParams(): Record<string, unknown> | null {
  const params: Record<string, unknown> = {};

  if (kind.value === 'benchmark_sweep') {
    const names = benchmarksText.value
      .split(',')
      .map((name) => name.trim())
      .filter((name) => name.length > 0);
    if (names.length === 0) {
      formError.value = 'benchmarks must be a non-empty list of names (the backend rejects an empty list).';
      return null;
    }
    params.benchmarks = names;
  }

  if (nPromptsText.value.trim() !== '') {
    const nPrompts = parseIntegerField(nPromptsText.value);
    if (nPrompts === null || nPrompts < 1 || nPrompts > 1000) {
      formError.value = 'n_prompts must be an integer between 1 and 1000.';
      return null;
    }
    params.n_prompts = nPrompts;
  }

  if (kind.value === 'ioi_stability' && nSeedsText.value.trim() !== '') {
    const nSeeds = parseIntegerField(nSeedsText.value);
    if (nSeeds === null || nSeeds < 1 || nSeeds > 10) {
      formError.value = 'n_seeds must be an integer between 1 and 10.';
      return null;
    }
    params.n_seeds = nSeeds;
  }

  return params;
}

function parseIntegerField(value: string): number | null {
  const trimmed = value.trim();
  if (!/^\d+$/.test(trimmed)) return null;
  return Number.parseInt(trimmed, 10);
}

async function requestCancel(job: JobRecord): Promise<void> {
  cancellingId.value = job.job_id;
  actionError.value = '';
  actionMessage.value = '';
  try {
    // The cancel response is the backend's; re-read the job so the row shows
    // the state the backend persisted, not the state we hoped for.
    await cancelJob(job.job_id);
    const reRead = await getJob(job.job_id);
    replaceJob(reRead.job);
    actionMessage.value = reRead.job.status === 'cancelled'
      ? `The backend reports job ${reRead.job.job_id} as cancelled.`
      : `Cancellation requested; the backend now reports ${reRead.job.status || 'status not returned'}. A running job stops cooperatively and may still finish.`;
  } catch (error) {
    actionError.value = isJobsHttpError(error) && error.isAlreadyFinished
      ? `The backend refused the cancellation, so the row is unchanged: ${error.detail || error.message}`
      : `Cancellation did not complete: ${describeJobsError(error)}`;
  } finally {
    cancellingId.value = '';
    syncPolling();
  }
}

function replaceJob(updated: JobRecord): void {
  const index = jobs.value.findIndex((job) => job.job_id === updated.job_id);
  if (index < 0) {
    jobs.value = [...jobs.value, updated];
    return;
  }
  jobs.value = [
    ...jobs.value.slice(0, index),
    updated,
    ...jobs.value.slice(index + 1),
  ];
}

function labelForKind(value: string): string {
  if (value === 'benchmark_sweep') return 'Benchmark sweep';
  if (value === 'ioi_stability') return 'IOI stability';
  return value || 'kind not returned';
}

function statusClass(status: string): string {
  if (isTerminalJobStatus(status)) return `is-${status}`;
  if (status === 'queued' || status === 'running') return `is-${status}`;
  return 'is-unknown';
}

function formatTimestamp(value: string | null): string {
  if (!value) return 'not returned';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function numberOrNotReturned(value: number | null): string {
  return value === null ? 'not returned' : String(value);
}

function paramsText(job: JobRecord): string {
  if (!job.payload) return 'not returned';
  return truncate(safeJson(job.payload), 240);
}

function hasResult(job: JobRecord): boolean {
  return job.result !== null && job.result !== undefined;
}

function resultText(job: JobRecord): string {
  if (!hasResult(job)) return 'not returned';
  return truncate(safeJson(job.result), 1600);
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function truncate(text: string, limit: number): string {
  return text.length > limit ? `${text.slice(0, limit)}\n… (truncated)` : text;
}
</script>

<style scoped>
.jobs-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.panel-heading,
.header-actions,
.job-head,
.job-head-actions,
.form-actions {
  display: flex;
  gap: 10px;
}

.surface-header,
.panel-heading {
  align-items: flex-start;
  justify-content: space-between;
  flex-wrap: wrap;
}

.surface-header {
  margin-bottom: 18px;
  align-items: center;
}

.header-copy {
  flex: 1 1 340px;
  min-width: 0;
}

.surface-kicker,
.panel-kicker {
  margin: 0 0 5px;
  color: var(--text-muted);
  font: 700 10px/1.2 var(--font-mono);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.surface-title {
  margin: 0;
  font-size: clamp(22px, 3vw, 30px);
  font-weight: 750;
  letter-spacing: -0.035em;
}

.surface-description,
.panel-description {
  margin: 7px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.55;
}

.header-actions {
  align-items: center;
  flex-wrap: wrap;
}

.surface-status {
  display: inline-flex;
  min-height: 28px;
  align-items: center;
  gap: 7px;
  padding: 0 9px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2, var(--bg-elev));
  color: var(--text-muted);
  font: 650 10px/1 var(--font-mono);
  white-space: nowrap;
}

.surface-status.is-connected {
  color: var(--text-dim);
}

.status-dot {
  width: 7px;
  height: 7px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: var(--text-muted);
}

.is-connected .status-dot {
  background: var(--success);
}

.quiet-button,
.primary-button {
  min-height: 32px;
  padding: 0 11px;
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
  font: 700 11px/1 var(--font);
}

.quiet-button {
  background: var(--surface);
  color: var(--text-muted);
}

.quiet-button:hover:not(:disabled) {
  color: var(--text);
  border-color: var(--border-light);
}

.quiet-button:disabled,
.primary-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.primary-button {
  border-color: var(--primary);
  background: var(--primary);
  color: #ffffff;
}

.notice {
  margin: 0 0 14px;
  padding: 11px 13px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--border-light);
  border-radius: 6px;
  background: var(--surface-2, var(--bg-elev));
  color: var(--text-dim);
  font-size: 12px;
  line-height: 1.5;
}

.notice strong {
  display: block;
  margin-bottom: 3px;
  color: var(--text);
}

.notice p {
  margin: 0;
}

.notice-error {
  border-left-color: var(--danger);
}

.notice-warn {
  border-left-color: var(--warning);
}

.notice-success {
  border-left-color: var(--success);
}

.notice-offline {
  border-style: dashed;
  border-left-color: var(--danger);
}

.notice-foot {
  margin-top: 6px !important;
  color: var(--text-muted);
  font-size: 11px;
}

.notice .quiet-button {
  margin-top: 8px;
}

.panel {
  margin-bottom: 14px;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.panel-heading {
  margin-bottom: 12px;
}

.panel-heading h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 700;
}

.panel-message {
  margin: 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.panel-footnote {
  margin-top: 12px;
  color: var(--text-muted);
  font-size: 11px;
}

.source-badge {
  padding: 3px 6px;
  border: 1px solid var(--border-light);
  border-radius: 999px;
  color: var(--text-muted);
  font: 700 9px/1 var(--font-mono);
  letter-spacing: 0.06em;
  white-space: nowrap;
}

.bounds-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin: 0;
}

.bounds-list div {
  padding: 9px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2, var(--bg-elev));
}

.bounds-list dt,
.job-facts dt {
  margin-bottom: 3px;
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.bounds-list dd,
.job-facts dd {
  margin: 0;
  color: var(--text);
  font-size: 12px;
  line-height: 1.45;
}

.bounds-dir {
  grid-column: 1 / -1;
}

.job-list {
  display: grid;
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.job-item {
  padding: 11px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2, var(--bg-elev));
}

.job-head {
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
}

.job-title {
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
}

.job-kind {
  font-size: 13px;
  font-weight: 700;
}

.job-id {
  color: var(--text-muted);
  font: 600 11px/1 var(--font-mono);
}

.job-head-actions {
  align-items: center;
  flex-wrap: wrap;
}

.status-badge.is-queued,
.status-badge.is-running {
  color: var(--warning);
  border-color: color-mix(in srgb, var(--warning) 25%, transparent);
}

.status-badge.is-completed {
  color: var(--success);
  border-color: color-mix(in srgb, var(--success) 25%, transparent);
}

.status-badge.is-failed,
.status-badge.is-interrupted {
  color: var(--danger);
  border-color: color-mix(in srgb, var(--danger) 25%, transparent);
}

.status-badge.is-unavailable,
.status-badge.is-cancelled,
.status-badge.is-unknown {
  color: var(--text-muted);
}

.job-facts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px 14px;
  margin: 10px 0 0;
}

.job-params {
  grid-column: 1 / -1;
}

.job-warning,
.job-error {
  margin: 10px 0 0;
  padding-top: 8px;
  border-top: 1px solid var(--border);
  color: var(--text-dim);
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
}

.job-warning strong,
.job-error strong {
  display: block;
  color: var(--text-muted);
  font-size: 10px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.job-result {
  margin-top: 10px;
}

.job-result summary {
  color: var(--text-muted);
  font-size: 11px;
  cursor: pointer;
}

.job-result pre {
  max-height: 320px;
  margin: 8px 0 0;
  padding: 10px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text-dim);
  font: 11px/1.5 var(--font-mono);
  white-space: pre-wrap;
}

.entry-form {
  display: grid;
  gap: 10px;
  max-width: 520px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.field label {
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.field input,
.field select {
  min-height: 32px;
  padding: 0 9px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  font: 12px/1.4 var(--font);
}

.field input:focus-visible,
.field select:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 1px;
}

.field-hint,
.form-hint {
  margin: 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.45;
}

.field-hint code {
  font-family: var(--font-mono);
}

.form-actions {
  align-items: center;
  flex-wrap: wrap;
}

.required-mark {
  color: var(--danger);
}

.optional-label {
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 500;
  text-transform: none;
}

.mono {
  font-family: var(--font-mono);
}

.break {
  word-break: break-word;
  overflow-wrap: anywhere;
}
</style>
