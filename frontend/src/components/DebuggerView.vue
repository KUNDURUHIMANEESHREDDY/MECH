<template>
  <main class="debugger-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Develop / Runtime diagnostics</p>
        <h2 class="surface-title">Debugger</h2>
        <p class="surface-description">Send a prompt to the backend diagnostic contract and inspect the returned model metadata without adding synthetic traces.</p>
      </div>
      <div class="header-actions">
        <span class="surface-status" :class="{ 'is-connected': backendAvailable }" role="status" aria-live="polite">
          <span class="status-dot" aria-hidden="true" />
          {{ backendStatusLabel }}
        </span>
        <button class="quiet-button" type="button" :disabled="isChecking" @click="refreshContracts">
          {{ isChecking ? 'Checking…' : 'Check contracts' }}
        </button>
      </div>
    </header>

    <div v-if="backendUnavailable" class="surface-notice notice-warning" role="status">
      <strong>Backend diagnostics are unavailable</strong>
      <p>
        The local bridge does not provide a model debugger endpoint. This surface can only show diagnostics when the existing
        backend <code>/api/infer</code> and <code>/api/models/:name</code> contracts respond.
      </p>
    </div>

    <div v-if="backendError" class="surface-notice notice-error" role="alert">
      <strong>Backend contract check failed</strong>
      <p>{{ backendError }}</p>
    </div>

    <div v-if="diagnosticError" class="surface-notice notice-error" role="alert">
      <strong>Prompt diagnostic failed</strong>
      <p>{{ diagnosticError }}</p>
    </div>

    <div v-if="actionMessage" class="surface-notice notice-success" role="status" aria-live="polite">
      <p>{{ actionMessage }}</p>
    </div>

    <div class="debugger-layout">
      <section class="surface-panel run-panel" aria-labelledby="run-diagnostic-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Prompt diagnostic</p>
            <h3 id="run-diagnostic-title">Run a model response</h3>
            <p class="panel-description">The prompt is sent only after you choose “Run diagnostic”.</p>
          </div>
          <span class="source-badge">API CONTRACT</span>
        </header>

        <form class="diagnostic-form" @submit.prevent="runDiagnostic">
          <div class="field">
            <label for="debug-prompt">Prompt <span class="required-mark" aria-hidden="true">*</span></label>
            <textarea
              id="debug-prompt"
              v-model="prompt"
              rows="8"
              placeholder="Enter the text to diagnose."
              required
              :disabled="!backendAvailable"
              aria-describedby="debug-prompt-help"
            />
            <p id="debug-prompt-help" class="field-help">No prompt is persisted by this surface.</p>
          </div>

          <div class="field">
            <label for="debug-model">Model <span class="optional-label">optional</span></label>
            <select id="debug-model" v-model="selectedModel" :disabled="!backendAvailable || !modelOptions.length">
              <option value="">Backend default</option>
              <option v-for="model in modelOptions" :key="model" :value="model">{{ model }}</option>
            </select>
            <p class="field-help">Model choices come from the backend registry; a blank value uses the backend default.</p>
          </div>

          <div class="run-actions">
            <span class="run-context">Endpoint: <code>POST /api/infer</code></span>
            <button class="primary-button" type="submit" :disabled="!canRun">
              {{ isRunning ? 'Running…' : 'Run diagnostic' }}
            </button>
          </div>
          <p v-if="!backendAvailable" class="action-help">The run action is disabled until the backend is reachable.</p>
        </form>

        <div v-if="isRunning" class="panel-message" role="status" aria-live="polite">
          <span class="loading-mark" aria-hidden="true" />
          Waiting for the backend diagnostic response…
        </div>
      </section>

      <aside class="surface-panel contract-panel" aria-labelledby="debug-contract-title">
        <p class="panel-kicker">Available contracts</p>
        <h3 id="debug-contract-title">What can be diagnosed</h3>
        <dl class="contract-list">
          <div>
            <dt>Model list</dt>
            <dd>{{ backendAvailable ? 'Available' : 'Unavailable' }}</dd>
          </div>
          <div>
            <dt>Prompt response</dt>
            <dd>{{ backendAvailable ? 'Available' : 'Unavailable' }}</dd>
          </div>
          <div>
            <dt>Model metadata</dt>
            <dd>{{ backendAvailable ? 'On demand' : 'Unavailable' }}</dd>
          </div>
        </dl>
        <p class="contract-note">
          This is a model-response inspector, not a step debugger. Layer, neuron, and attention step controls remain unavailable
          until a backend contract exposes them.
        </p>
      </aside>
    </div>

    <div v-if="diagnosticResult || isLoadingInfo || metadataError" class="result-grid">
      <section class="surface-panel response-panel" aria-labelledby="debug-response-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Backend response</p>
            <h3 id="debug-response-title">Prompt result</h3>
            <p class="panel-description">Raw fields returned by the diagnostic endpoint.</p>
          </div>
          <span class="source-badge" :class="{ 'is-warning': resultProvenance !== 'live' }">{{ resultProvenanceLabel }}</span>
        </header>
        <dl v-if="resultSummary.length" class="summary-list">
          <div v-for="entry in resultSummary" :key="entry.label">
            <dt>{{ entry.label }}</dt>
            <dd :class="{ mono: entry.mono }">{{ entry.value }}</dd>
          </div>
        </dl>
        <pre class="result-code">{{ formattedResult }}</pre>
      </section>

      <section class="surface-panel metadata-panel" aria-labelledby="debug-metadata-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Model contract</p>
            <h3 id="debug-metadata-title">Model metadata</h3>
            <p class="panel-description">Requested separately from <code>GET /api/models/:name</code>.</p>
          </div>
          <button class="quiet-button" type="button" :disabled="!selectedModel || isLoadingInfo" @click="inspectModel">
            {{ isLoadingInfo ? 'Loading…' : 'Inspect model' }}
          </button>
        </header>
        <div v-if="isLoadingInfo" class="panel-message" role="status" aria-live="polite">
          <span class="loading-mark" aria-hidden="true" />
          Loading model metadata…
        </div>
        <p v-else-if="metadataError" class="panel-message panel-message-error" role="alert">{{ metadataError }}</p>
        <dl v-else-if="metadataEntries.length" class="summary-list">
          <div v-for="entry in metadataEntries" :key="entry.label">
            <dt>{{ entry.label }}</dt>
            <dd :class="{ mono: entry.mono }">{{ entry.value }}</dd>
          </div>
        </dl>
        <p v-else class="panel-message">Select a model and inspect its backend metadata.</p>
      </section>
    </div>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { api } from '../services/api';

type UnknownRecord = Record<string, unknown>;
type BackendState = 'checking' | 'available' | 'unavailable';
type SummaryEntry = { label: string; value: string; mono: boolean };

const prompt = ref('');
const selectedModel = ref('');
const modelOptions = ref<string[]>([]);
const backendState = ref<BackendState>('checking');
const isChecking = ref(false);
const isRunning = ref(false);
const isLoadingInfo = ref(false);
const backendError = ref('');
const diagnosticError = ref('');
const metadataError = ref('');
const actionMessage = ref('');
const diagnosticResult = ref<UnknownRecord | null>(null);
const modelInfo = ref<UnknownRecord | null>(null);

const backendAvailable = computed(() => backendState.value === 'available');
const backendStatusLabel = computed(() => {
  if (backendState.value === 'available') return 'Backend available';
  if (backendState.value === 'checking') return 'Checking backend';
  return 'Backend unavailable';
});
const canRun = computed(() => backendAvailable.value && Boolean(prompt.value.trim()) && !isRunning.value);
const resultProvenance = computed(() => diagnosticResult.value ? stringValue(diagnosticResult.value.provenance).toLocaleLowerCase() : '');
const resultProvenanceLabel = computed(() => {
  if (resultProvenance.value === 'live') return 'LIVE BACKEND';
  if (resultProvenance.value === 'seeded') return 'SEEDED BACKEND';
  return resultProvenance.value ? 'BACKEND PROVENANCE' : 'BACKEND RESPONSE';
});
const formattedResult = computed(() => {
  if (!diagnosticResult.value) return '';
  try {
    return JSON.stringify(diagnosticResult.value, null, 2);
  } catch {
    return String(diagnosticResult.value);
  }
});
const resultSummary = computed<SummaryEntry[]>(() => {
  if (!diagnosticResult.value) return [];
  const record = diagnosticResult.value;
  const entries: SummaryEntry[] = [];
  const modelName = stringValue(record.model_name) || selectedModel.value;
  if (modelName) entries.push({ label: 'Model', value: modelName, mono: true });
  if (record.provenance !== undefined) entries.push({ label: 'Provenance', value: displayValue(record.provenance), mono: true });
  if (record.provenance_note !== undefined) entries.push({ label: 'Provenance note', value: displayValue(record.provenance_note), mono: false });
  if (record.generated_text !== undefined) entries.push({ label: 'Generated text', value: displayValue(record.generated_text), mono: false });
  return entries;
});
const metadataEntries = computed<SummaryEntry[]>(() => {
  if (!modelInfo.value) return [];
  const record = modelInfo.value;
  const preferred: Array<[string, string]> = [
    ['model_name', 'Model name'],
    ['layers', 'Layers'],
    ['hidden_size', 'Hidden size'],
    ['vocab_size', 'Vocabulary size'],
    ['num_heads', 'Attention heads'],
  ];
  const entries: SummaryEntry[] = [];
  for (const [key, label] of preferred) {
    if (record[key] !== undefined && record[key] !== null) entries.push({ label, value: displayValue(record[key]), mono: key !== 'model_name' });
  }
  return entries.length ? entries : Object.entries(record).map(([key, value]) => ({
    label: humanize(key),
    value: displayValue(value),
    mono: typeof value === 'string' && /name|id|path|model/i.test(key),
  }));
});

onMounted(() => {
  void refreshContracts();
});

async function refreshContracts() {
  isChecking.value = true;
  backendError.value = '';
  try {
    const result = await api.listModels();
    modelOptions.value = Array.isArray(result.models)
      ? result.models.filter((value): value is string => typeof value === 'string' && Boolean(value.trim()))
      : [];
    backendState.value = 'available';
  } catch (error) {
    modelOptions.value = [];
    backendState.value = 'unavailable';
    backendError.value = errorMessage(error);
  } finally {
    isChecking.value = false;
  }
}

async function runDiagnostic() {
  if (!canRun.value) return;
  isRunning.value = true;
  diagnosticError.value = '';
  metadataError.value = '';
  actionMessage.value = '';
  try {
    const result = await api.infer(prompt.value.trim(), selectedModel.value || undefined);
    if (isRecord(result) && (result.error !== undefined || stringValue(result.status).toLocaleLowerCase() === 'error')) {
      throw new Error(stringValue(result.error) || 'The backend returned an error for this diagnostic.');
    }
    diagnosticResult.value = result;
    if (isRecord(result) && stringValue(result.model_name)) selectedModel.value = stringValue(result.model_name);
    backendState.value = 'available';
    actionMessage.value = 'The backend returned a prompt diagnostic.';
  } catch (error) {
    diagnosticError.value = errorMessage(error);
    backendState.value = 'unavailable';
  } finally {
    isRunning.value = false;
  }
}

async function inspectModel() {
  if (!selectedModel.value || isLoadingInfo.value) return;
  isLoadingInfo.value = true;
  metadataError.value = '';
  try {
    modelInfo.value = await api.getModelInfo(selectedModel.value);
  } catch (error) {
    modelInfo.value = null;
    metadataError.value = errorMessage(error);
  } finally {
    isLoadingInfo.value = false;
  }
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return 'Not supplied';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function humanize(value: string): string {
  return value.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim();
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === 'string' && error.trim()) return error;
  if (isRecord(error) && stringValue(error.error)) return stringValue(error.error);
  return 'The MECH backend could not complete that diagnostic request.';
}
</script>

<style scoped>
.debugger-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.header-actions,
.panel-heading,
.run-actions,
.summary-list div,
.result-panel .panel-heading {
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
  border-color: var(--text);
  background: var(--text);
  color: var(--surface);
}

.primary-button:hover:not(:disabled) {
  background: var(--text-dim);
}

.quiet-button:disabled,
.primary-button:disabled {
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

.notice-warning {
  border-left-color: var(--text-dim);
}

.notice-error {
  border-left-color: var(--danger);
}

.notice-success {
  border-left-color: var(--success);
}

.debugger-layout,
.result-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(240px, 0.65fr);
  gap: 14px;
  align-items: start;
}

.result-grid {
  margin-top: 14px;
}

.surface-panel {
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.run-panel,
.contract-panel,
.response-panel,
.metadata-panel {
  padding: clamp(16px, 2.2vw, 22px);
}

.panel-heading {
  align-items: flex-start;
  margin-bottom: 18px;
}

.panel-heading h3,
.contract-panel h3 {
  margin: 0;
  color: var(--text);
  font-size: 17px;
  font-weight: 720;
  letter-spacing: -0.02em;
}

.source-badge {
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

.source-badge.is-warning {
  border-color: var(--border-light);
  color: var(--text-dim);
}

.diagnostic-form {
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

.field textarea,
.field select {
  display: block;
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 6px;
  outline: 0;
  background: var(--surface);
  color: var(--text);
  font: 12px/1.45 var(--font);
}

.field textarea {
  min-height: 150px;
  padding: 9px 10px;
  resize: vertical;
}

.field select {
  min-height: 38px;
  padding: 0 10px;
}

.field textarea:focus,
.field select:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--accent-soft);
}

.field textarea:focus-visible,
.field select:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

.field textarea:disabled,
.field select:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.required-mark {
  color: var(--danger);
}

.optional-label {
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 500;
}

.field-help,
.action-help {
  margin: 6px 0 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.45;
}

.run-actions {
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
}

.run-context {
  color: var(--text-muted);
  font-size: 10px;
}

.panel-message {
  display: flex;
  min-height: 76px;
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
  animation: debugger-spin 800ms linear infinite;
}

@keyframes debugger-spin {
  to { transform: rotate(360deg); }
}

.contract-list,
.summary-list {
  display: grid;
  gap: 0;
  margin: 18px 0 0;
  border-top: 1px solid var(--border);
}

.contract-list div,
.summary-list div {
  align-items: baseline;
  padding: 9px 0;
  border-bottom: 1px solid var(--border);
}

.contract-list dt,
.contract-list dd,
.summary-list dt,
.summary-list dd {
  margin: 0;
  font-size: 11px;
}

.contract-list dt,
.summary-list dt {
  color: var(--text-muted);
}

.contract-list dd,
.summary-list dd {
  color: var(--text-dim);
  font-weight: 650;
  text-align: right;
  overflow-wrap: anywhere;
}

.contract-note {
  margin: 17px 0 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.55;
}

.result-code {
  max-height: 380px;
  margin: 15px 0 0;
  padding: 12px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text-dim);
  font: 11px/1.55 var(--font-mono);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

@media (max-width: 800px) {
  .debugger-layout,
  .result-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 580px) {
  .surface-header,
  .header-actions,
  .panel-heading,
  .run-actions {
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

  .run-actions .primary-button {
    width: 100%;
  }
}

@media (prefers-reduced-motion: reduce) {
  .loading-mark {
    animation: none;
  }
}
</style>
