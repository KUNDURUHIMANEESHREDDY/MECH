<template>
  <main class="experiments-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Research / Backend records</p>
        <h2 class="surface-title">Experiments</h2>
        <p class="surface-description">Create, inspect, refresh, and remove experiment records stored by the MECH backend.</p>
      </div>
      <div class="header-actions">
        <span class="surface-status" :class="{ 'is-connected': isConnected }" role="status" aria-live="polite">
          <span class="status-dot" aria-hidden="true" />
          {{ isLoading ? 'Loading backend' : isConnected ? 'Backend connected' : 'Backend unavailable' }}
        </span>
        <button class="quiet-button" type="button" :disabled="isLoading" @click="refreshExperiments">
          {{ isLoading ? 'Refreshing…' : 'Refresh records' }}
        </button>
      </div>
    </header>

    <div v-if="loadError" class="surface-notice notice-error" role="alert">
      <strong>Experiments could not be loaded</strong>
      <p>{{ loadError }}</p>
    </div>

    <div v-if="actionError" class="surface-notice notice-error" role="alert">
      <strong>Experiment action failed</strong>
      <p>{{ actionError }}</p>
    </div>

    <div v-if="actionMessage" class="surface-notice notice-success" role="status" aria-live="polite">
      <p>{{ actionMessage }}</p>
    </div>

    <div class="experiments-layout">
      <section class="surface-panel records-panel" aria-labelledby="experiment-records-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Application storage</p>
            <h3 id="experiment-records-title">Backend experiment records</h3>
            <p class="panel-description">Records returned by <code>GET /api/experiments</code>; no local experiment data is substituted.</p>
          </div>
          <span class="source-badge">SOURCE: API</span>
        </header>

        <div v-if="isLoading" class="panel-message" role="status" aria-live="polite">
          <span class="loading-mark" aria-hidden="true" />
          Loading experiment records…
        </div>

        <ul v-else-if="experiments.length" class="experiment-list" aria-label="Backend experiment records">
          <li v-for="experiment in experiments" :key="experiment.id" class="experiment-item">
            <button
              class="experiment-select"
              type="button"
              :class="{ 'is-selected': selectedExperimentId === experiment.id }"
              :aria-pressed="selectedExperimentId === experiment.id"
              @click="selectExperiment(experiment.id)"
            >
              <span class="experiment-heading">
                <span class="experiment-mark" aria-hidden="true">E</span>
                <span class="experiment-title">{{ experiment.title }}</span>
                <span v-if="experiment.status" class="status-badge" :class="statusClass(experiment.status)">{{ experiment.status }}</span>
              </span>
              <span class="experiment-detail">Target: {{ experiment.targetCircuit || 'Not supplied' }}</span>
              <span class="experiment-meta">Updated {{ formatTimestamp(experiment.updatedAt || experiment.createdAt) }}</span>
            </button>
            <button
              v-if="experiment.id"
              class="icon-button"
              type="button"
              :aria-label="`Delete experiment ${experiment.title}`"
              title="Delete experiment"
              :disabled="isDeleting"
              @click="deleteExperiment(experiment)"
            >
              <span aria-hidden="true">×</span>
            </button>
          </li>
        </ul>

        <p v-else-if="loadError" class="panel-message panel-message-error">
          Experiment records are unavailable for this request.
        </p>
        <p v-else-if="hasLoaded" class="panel-message">
          The backend returned no experiment records. Create one only when you have a real record to save.
        </p>
        <p v-else class="panel-message">
          Experiment records require a reachable MECH backend.
        </p>

        <footer v-if="lastRefreshed" class="panel-footnote">
          Last refreshed <time :datetime="lastRefreshed">{{ formatTimestamp(lastRefreshed) }}</time>.
        </footer>
      </section>

      <section class="surface-panel form-panel" aria-labelledby="create-experiment-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Write record</p>
            <h3 id="create-experiment-title">Create experiment</h3>
            <p class="panel-description">The record is sent to the backend only when you save it.</p>
          </div>
        </header>

        <form class="entry-form" @submit.prevent="createExperiment">
          <div class="field">
            <label for="experiment-title">Title <span class="required-mark" aria-hidden="true">*</span></label>
            <input id="experiment-title" v-model="newTitle" type="text" autocomplete="off" placeholder="A descriptive experiment name" required>
          </div>
          <div class="field">
            <label for="experiment-target">Target circuit <span class="optional-label">optional</span></label>
            <input id="experiment-target" v-model="newTarget" type="text" autocomplete="off" placeholder="e.g. L8_N402">
          </div>
          <div class="field">
            <label for="experiment-status">Status <span class="optional-label">optional</span></label>
            <select id="experiment-status" v-model="newStatus">
              <option value="pending">Pending</option>
              <option value="running">Running</option>
              <option value="complete">Complete</option>
              <option value="failed">Failed</option>
            </select>
          </div>
          <p v-if="formError" class="form-message" role="alert">{{ formError }}</p>
          <div class="form-actions">
            <button class="primary-button" type="submit" :disabled="isSaving">
              {{ isSaving ? 'Saving…' : 'Save experiment' }}
            </button>
            <button class="quiet-button" type="button" :disabled="isSaving" @click="clearForm">Clear</button>
          </div>
        </form>
      </section>
    </div>

    <section v-if="selectedExperiment" class="surface-panel detail-panel" aria-labelledby="experiment-detail-title">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">Selected backend record</p>
          <h3 id="experiment-detail-title">{{ selectedExperiment.title }}</h3>
          <p class="panel-description">Fields are shown exactly as returned by the backend.</p>
        </div>
        <button class="quiet-button" type="button" @click="selectedExperimentId = ''">Clear selection</button>
      </header>
      <dl class="detail-list">
        <div><dt>ID</dt><dd class="mono">{{ selectedExperiment.id || 'Not supplied' }}</dd></div>
        <div><dt>Title</dt><dd>{{ selectedExperiment.title }}</dd></div>
        <div><dt>Target circuit</dt><dd>{{ selectedExperiment.targetCircuit || 'Not supplied' }}</dd></div>
        <div><dt>Status</dt><dd>{{ selectedExperiment.status || 'Not supplied' }}</dd></div>
        <div><dt>Created</dt><dd>{{ formatTimestamp(selectedExperiment.createdAt) }}</dd></div>
        <div><dt>Updated</dt><dd>{{ formatTimestamp(selectedExperiment.updatedAt) }}</dd></div>
      </dl>
      <details class="raw-details">
        <summary>Raw backend record</summary>
        <pre>{{ selectedRawText }}</pre>
      </details>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { api } from '../services/api';

type UnknownRecord = Record<string, unknown>;

interface ExperimentRecord {
  id: string;
  title: string;
  targetCircuit: string;
  status: string;
  createdAt: string;
  updatedAt: string;
  raw: unknown;
}

const experiments = ref<ExperimentRecord[]>([]);
const selectedExperimentId = ref('');
const newTitle = ref('');
const newTarget = ref('');
const newStatus = ref('pending');
const isLoading = ref(false);
const isSaving = ref(false);
const isDeleting = ref(false);
const hasLoaded = ref(false);
const loadError = ref('');
const actionError = ref('');
const formError = ref('');
const actionMessage = ref('');
const lastRefreshed = ref('');

const isConnected = computed(() => hasLoaded.value && !loadError.value);
const selectedExperiment = computed(() => experiments.value.find((item) => item.id === selectedExperimentId.value) ?? null);
const selectedRawText = computed(() => {
  if (!selectedExperiment.value) return '';
  try {
    return JSON.stringify(selectedExperiment.value.raw, null, 2);
  } catch {
    return String(selectedExperiment.value.raw);
  }
});

onMounted(() => {
  void refreshExperiments();
});

async function refreshExperiments() {
  isLoading.value = true;
  loadError.value = '';
  try {
    const result = await api.listExperiments();
    experiments.value = (Array.isArray(result) ? result : [])
      .map((value, index) => normalizeExperiment(value, index))
      .filter((value): value is ExperimentRecord => value !== null);
    hasLoaded.value = true;
    lastRefreshed.value = new Date().toISOString();
    if (selectedExperimentId.value && !selectedExperiment.value) selectedExperimentId.value = '';
  } catch (error) {
    experiments.value = [];
    hasLoaded.value = true;
    loadError.value = errorMessage(error);
  } finally {
    isLoading.value = false;
  }
}

function selectExperiment(id: string) {
  selectedExperimentId.value = id;
}

async function createExperiment() {
  formError.value = '';
  actionError.value = '';
  actionMessage.value = '';
  const title = newTitle.value.trim();
  if (!title) {
    formError.value = 'Enter an experiment title before saving.';
    return;
  }

  isSaving.value = true;
  try {
    const record = {
      id: makeId('experiment'),
      title,
      targetCircuit: newTarget.value.trim(),
      status: newStatus.value,
      createdAt: new Date().toISOString(),
      results: {},
    };
    const result = await api.createExperiment(record);
    if (isRecord(result) && result.error !== undefined) throw new Error(stringValue(result.error) || 'The backend rejected the experiment record.');
    await refreshExperiments();
    clearForm();
    actionMessage.value = 'Experiment record saved to the backend.';
  } catch (error) {
    actionError.value = errorMessage(error);
  } finally {
    isSaving.value = false;
  }
}

async function deleteExperiment(experiment: ExperimentRecord) {
  if (!experiment.id || typeof window !== 'undefined' && !window.confirm(`Delete backend experiment “${experiment.title}”?`)) return;
  isDeleting.value = true;
  actionError.value = '';
  actionMessage.value = '';
  try {
    const result = await api.deleteExperiment(experiment.id);
    if (isRecord(result) && stringValue(result.status).toLocaleLowerCase() === 'not_found') {
      throw new Error('The backend no longer contains that experiment record.');
    }
    experiments.value = experiments.value.filter((item) => item.id !== experiment.id);
    if (selectedExperimentId.value === experiment.id) selectedExperimentId.value = '';
    actionMessage.value = `Deleted ${experiment.title} from the backend.`;
  } catch (error) {
    actionError.value = errorMessage(error);
  } finally {
    isDeleting.value = false;
  }
}

function clearForm() {
  newTitle.value = '';
  newTarget.value = '';
  newStatus.value = 'pending';
  formError.value = '';
}

function normalizeExperiment(value: unknown, index: number): ExperimentRecord | null {
  if (!isRecord(value)) return null;
  const title = stringValue(value.title) || stringValue(value.name) || stringValue(value.experiment_name);
  const id = stringValue(value.id) || stringValue(value.item_id) || stringValue(value.experiment_id);
  if (!title && !id) return null;
  return {
    id: id || `experiment-record-${index}`,
    title: title || id || `Experiment ${index + 1}`,
    targetCircuit: stringValue(value.targetCircuit) || stringValue(value.target_circuit),
    status: stringValue(value.status),
    createdAt: stringValue(value.createdAt) || stringValue(value.created_at),
    updatedAt: stringValue(value.updatedAt) || stringValue(value.updated_at),
    raw: value,
  };
}

function makeId(prefix: string): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return `${prefix}-${crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function statusClass(status: string): string {
  return `status-${status.trim().toLocaleLowerCase().replace(/[^a-z0-9]+/g, '-') || 'unknown'}`;
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

function formatTimestamp(value: string): string {
  if (!value) return 'an unspecified time';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === 'string' && error.trim()) return error;
  if (isRecord(error) && stringValue(error.error)) return stringValue(error.error);
  return 'The MECH backend could not complete that experiment request.';
}
</script>

<style scoped>
.experiments-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.header-actions,
.panel-heading,
.experiment-heading,
.form-actions,
.panel-footnote {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.surface-header {
  align-items: center;
  margin-bottom: 20px;
}

.header-copy {
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
  color: var(--text);
  font-size: clamp(22px, 3vw, 30px);
  font-weight: 750;
  letter-spacing: -0.035em;
  line-height: 1.1;
}

.surface-description,
.panel-description {
  margin: 7px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.header-actions {
  flex: 0 0 auto;
  align-items: center;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.surface-status {
  display: inline-flex;
  min-height: 28px;
  align-items: center;
  gap: 7px;
  padding: 0 9px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
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

.surface-status.is-connected .status-dot {
  background: var(--success);
}

.quiet-button,
.primary-button {
  min-height: 30px;
  padding: 0 10px;
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
  border-color: var(--border-light);
  color: var(--text);
}

.primary-button {
  border-color: var(--primary);
  background: var(--primary);
  color: #ffffff;
}

.primary-button:hover:not(:disabled) {
  border-color: var(--primary-focus);
  background: var(--primary-focus);
}

.quiet-button:disabled,
.primary-button:disabled,
.icon-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.surface-notice {
  margin-bottom: 14px;
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--border-light);
  border-radius: var(--radius);
  background: var(--surface-2);
  color: var(--text-dim);
  font-size: 12px;
  line-height: 1.5;
}

.surface-notice strong {
  display: block;
  margin-bottom: 3px;
  color: var(--text);
}

.surface-notice p {
  margin: 0;
}

.notice-error {
  border-left-color: var(--danger);
}

.notice-success {
  border-left-color: var(--success);
}

.experiments-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(260px, 0.65fr);
  gap: 14px;
  align-items: start;
}

.surface-panel {
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.records-panel,
.form-panel,
.detail-panel {
  padding: clamp(16px, 2.2vw, 22px);
}

.panel-heading {
  align-items: flex-start;
  margin-bottom: 18px;
}

.panel-heading h3 {
  margin: 0;
  color: var(--text);
  font-size: 17px;
  font-weight: 720;
  letter-spacing: -0.02em;
}

.source-badge,
.status-badge {
  display: inline-flex;
  min-height: 21px;
  align-items: center;
  padding: 0 7px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-muted);
  font: 700 9px/1 var(--font-mono);
  letter-spacing: 0.03em;
  white-space: nowrap;
}

.status-running,
.status-complete,
.status-completed {
  border-color: var(--border-light);
  color: var(--text-dim);
}

.status-failed,
.status-error {
  border-color: var(--danger);
  color: var(--danger);
}

.experiment-list {
  display: grid;
  gap: 7px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.experiment-item {
  display: flex;
  min-width: 0;
  gap: 5px;
}

.experiment-select {
  display: block;
  min-width: 0;
  flex: 1;
  padding: 11px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text);
  cursor: pointer;
  text-align: left;
}

.experiment-select:hover,
.experiment-select.is-selected {
  border-color: var(--primary);
  background: var(--accent-soft);
}

.experiment-heading {
  align-items: center;
  justify-content: flex-start;
  gap: 8px;
}

.experiment-mark {
  display: inline-grid;
  width: 25px;
  height: 25px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 5px;
  background: var(--surface);
  color: var(--primary);
  font: 750 9px/1 var(--font-mono);
}

.experiment-title {
  min-width: 0;
  overflow: hidden;
  color: var(--text);
  font-size: 12px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.experiment-detail,
.experiment-meta {
  display: block;
  margin: 8px 0 0 33px;
  color: var(--text-dim);
  font-size: 11px;
  line-height: 1.4;
}

.experiment-meta {
  margin-top: 5px;
  color: var(--text-muted);
  font-size: 10px;
}

.icon-button {
  width: 30px;
  min-height: 30px;
  padding: 0;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text-muted);
  cursor: pointer;
  font-size: 16px;
}

.icon-button:hover:not(:disabled) {
  border-color: var(--danger);
  color: var(--danger);
}

.panel-message {
  display: flex;
  min-height: 86px;
  align-items: center;
  gap: 9px;
  margin: 0;
  padding: 12px;
  border: 1px dashed var(--border-light);
  border-radius: 6px;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.panel-message-error {
  border-style: solid;
  color: var(--danger);
}

.loading-mark {
  width: 12px;
  height: 12px;
  flex: 0 0 auto;
  border: 1px solid var(--border-light);
  border-top-color: var(--text);
  border-radius: 50%;
  animation: experiments-spin 800ms linear infinite;
}

@keyframes experiments-spin {
  to { transform: rotate(360deg); }
}

.panel-footnote {
  margin: 16px 0 0;
  color: var(--text-muted);
  font-size: 10px;
}

.form-panel {
  min-width: 0;
}

.entry-form {
  display: grid;
  gap: 14px;
}

.field {
  min-width: 0;
}

.field label {
  display: block;
  margin-bottom: 6px;
  color: var(--text-dim);
  font-size: 12px;
  font-weight: 700;
}

.field input,
.field select {
  display: block;
  width: 100%;
  min-height: 38px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  outline: 0;
  background: var(--surface);
  color: var(--text);
  font: 12px/1.45 var(--font);
}

.field input:focus,
.field select:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--accent-soft);
}

.field input:focus-visible,
.field select:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

.optional-label {
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 500;
}

.required-mark {
  color: var(--danger);
}

.form-message {
  margin: 0;
  color: var(--danger);
  font-size: 11px;
  line-height: 1.45;
}

.form-actions {
  align-items: center;
  justify-content: flex-end;
  flex-wrap: wrap;
}

.detail-panel {
  margin-top: 14px;
}

.detail-list {
  display: grid;
  gap: 0;
  margin: 0;
  border-top: 1px solid var(--border);
}

.detail-list div {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  padding: 9px 0;
  border-bottom: 1px solid var(--border);
}

.detail-list dt,
.detail-list dd {
  margin: 0;
  font-size: 11px;
}

.detail-list dt {
  color: var(--text-muted);
}

.detail-list dd {
  color: var(--text-dim);
  font-weight: 650;
  text-align: right;
  overflow-wrap: anywhere;
}

.mono {
  font-family: var(--font-mono);
}

.raw-details {
  margin-top: 15px;
  color: var(--text-muted);
  font-size: 11px;
}

.raw-details summary {
  cursor: pointer;
  font-weight: 700;
}

.raw-details pre {
  max-height: 260px;
  margin: 9px 0 0;
  padding: 10px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text-dim);
  font: 11px/1.5 var(--font-mono);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

@media (max-width: 820px) {
  .experiments-layout {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 580px) {
  .surface-header,
  .header-actions,
  .panel-heading,
  .form-actions {
    align-items: flex-start;
    flex-direction: column;
  }

  .header-actions {
    width: 100%;
  }

  .header-actions .quiet-button,
  .header-actions .surface-status {
    align-self: flex-start;
  }

  .form-actions .primary-button,
  .form-actions .quiet-button {
    width: 100%;
  }
}

@media (prefers-reduced-motion: reduce) {
  .loading-mark {
    animation: none;
  }
}
</style>
