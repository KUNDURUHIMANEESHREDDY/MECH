<template>
  <main class="models-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Backend / Model registry</p>
        <h2 class="surface-title">Models</h2>
        <p class="surface-description">Inspect the model registry and request a model load from the active MECH runtime.</p>
      </div>
      <div class="header-actions">
        <span class="surface-status" :class="{ 'is-connected': isConnected }" role="status" aria-live="polite">
          <span class="status-dot" aria-hidden="true" />
          {{ isConnected ? 'Backend connected' : isLoading ? 'Checking backend' : 'Backend unavailable' }}
        </span>
        <button class="quiet-button" type="button" :disabled="isLoading" @click="refreshModels">
          {{ isLoading ? 'Refreshing…' : 'Refresh models' }}
        </button>
      </div>
    </header>

    <div v-if="loadError" class="surface-notice notice-error" role="alert">
      <strong>Models could not be loaded</strong>
      <p>{{ loadError }}</p>
    </div>

    <div v-if="actionMessage" class="surface-notice notice-success" role="status" aria-live="polite">
      <p>{{ actionMessage }}</p>
    </div>

    <section class="surface-panel registry-panel" aria-labelledby="model-registry-title">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">Backend registry</p>
          <h3 id="model-registry-title">Available models</h3>
          <p class="panel-description">Records returned by <code>GET /api/models</code>; no catalogue entries are inferred locally.</p>
        </div>
        <span class="source-badge">SOURCE: API</span>
      </header>

      <div v-if="isLoading" class="panel-message" role="status" aria-live="polite">
        <span class="loading-mark" aria-hidden="true" />
        Loading the model registry…
      </div>

      <ul v-else-if="models.length" class="model-grid" aria-label="Available backend models">
        <li v-for="model in models" :key="model.name" class="model-card">
          <div class="model-card-heading">
            <span class="model-mark" aria-hidden="true">M</span>
            <div class="model-heading-copy">
              <h4>{{ model.name }}</h4>
              <p>Returned by the backend registry.</p>
            </div>
            <span v-if="model.loaded" class="record-badge">LOADED</span>
          </div>
          <dl class="model-meta">
            <div>
              <dt>Model ID</dt>
              <dd class="mono">{{ model.name }}</dd>
            </div>
            <div>
              <dt>Source</dt>
              <dd>Backend API</dd>
            </div>
          </dl>
          <div class="model-actions">
            <button class="quiet-button" type="button" @click="inspectModel(model.name)">
              Inspect metadata
            </button>
            <button
              class="primary-button"
              type="button"
              :disabled="isLoadingModel || model.name === loadingModel"
              @click="loadModel(model.name)"
            >
              {{ model.name === loadingModel && isLoadingModel ? 'Loading…' : 'Load model' }}
            </button>
          </div>
        </li>
      </ul>

      <p v-else-if="loadError" class="panel-message panel-message-error">
        Model records are unavailable for this request.
      </p>
      <p v-else-if="hasLoaded" class="panel-message">
        The backend returned an empty model registry. No local substitutes are shown.
      </p>
      <p v-else class="panel-message">
        Model records require a reachable MECH backend.
      </p>

      <footer v-if="lastRefreshed" class="panel-footnote">
        Last refreshed <time :datetime="lastRefreshed">{{ formatTimestamp(lastRefreshed) }}</time>.
      </footer>
    </section>

    <div class="models-lower-grid">
      <section v-if="selectedModel" class="surface-panel detail-panel" aria-labelledby="model-detail-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Backend metadata</p>
            <h3 id="model-detail-title">{{ selectedModel }}</h3>
            <p class="panel-description">Values returned by <code>GET /api/models/:name</code>.</p>
          </div>
          <button class="quiet-button" type="button" @click="clearSelection">Clear</button>
        </header>
        <div v-if="isLoadingInfo" class="panel-message" role="status" aria-live="polite">
          <span class="loading-mark" aria-hidden="true" />
          Loading model metadata…
        </div>
        <div v-else-if="modelInfoError" class="panel-message panel-message-error" role="alert">
          {{ modelInfoError }}
        </div>
        <dl v-else class="detail-list">
          <div v-for="entry in infoEntries" :key="entry.label">
            <dt>{{ entry.label }}</dt>
            <dd :class="{ mono: entry.mono }">{{ entry.value }}</dd>
          </div>
        </dl>
        <p v-if="modelProvenance" class="provenance-note">
          <span class="source-badge">PROVENANCE</span>
          {{ modelProvenance }}
        </p>
      </section>
      <section v-else class="surface-panel detail-panel detail-empty" aria-labelledby="model-detail-empty-title">
        <p class="panel-kicker">Backend metadata</p>
        <h3 id="model-detail-empty-title">Select a model</h3>
        <p>Choose “Inspect metadata” to request the backend’s model record for a registry entry.</p>
      </section>

      <aside class="surface-panel contract-panel" aria-labelledby="model-contract-title">
        <p class="panel-kicker">Runtime contract</p>
        <h3 id="model-contract-title">What this surface can do</h3>
        <dl class="contract-list">
          <div>
            <dt>Registry</dt>
            <dd>Read from <code>api.listModels()</code></dd>
          </div>
          <div>
            <dt>Metadata</dt>
            <dd>Read on demand from <code>api.getModelInfo()</code></dd>
          </div>
          <div>
            <dt>Load</dt>
            <dd>Sent through <code>api.loadModel()</code> only on request</dd>
          </div>
        </dl>
        <p class="help-copy">No model availability, architecture, or scientific result is assumed before the backend returns it.</p>
      </aside>
    </div>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { api } from '../services/api';

type UnknownRecord = Record<string, unknown>;

interface ModelRecord {
  name: string;
  loaded: boolean;
}

interface InfoEntry {
  label: string;
  value: string;
  mono: boolean;
}

const models = ref<ModelRecord[]>([]);
const selectedModel = ref('');
const modelInfo = ref<UnknownRecord | null>(null);
const isLoading = ref(false);
const isLoadingInfo = ref(false);
const isLoadingModel = ref(false);
const loadingModel = ref('');
const hasLoaded = ref(false);
const loadError = ref('');
const modelInfoError = ref('');
const actionMessage = ref('');
const lastRefreshed = ref('');

const isConnected = computed(() => hasLoaded.value && !loadError.value);
const infoEntries = computed<InfoEntry[]>(() => {
  if (!modelInfo.value) return [];
  const preferred: Array<[string, string]> = [
    ['model_name', 'Model name'],
    ['layers', 'Layers'],
    ['hidden_size', 'Hidden size'],
    ['vocab_size', 'Vocabulary size'],
    ['num_heads', 'Attention heads'],
  ];
  const entries: InfoEntry[] = [];
  for (const [key, label] of preferred) {
    if (modelInfo.value[key] !== undefined && modelInfo.value[key] !== null) {
      entries.push({ label, value: displayValue(modelInfo.value[key]), mono: key !== 'model_name' });
    }
  }
  return entries.length ? entries : Object.entries(modelInfo.value).map(([key, value]) => ({
    label: humanize(key),
    value: displayValue(value),
    mono: typeof value === 'string' && /id|name|path|model/i.test(key),
  }));
});
const modelProvenance = computed(() => {
  if (!modelInfo.value) return '';
  return stringValue(modelInfo.value.provenance_note) || stringValue(modelInfo.value.provenance);
});

onMounted(() => {
  void refreshModels();
});

async function refreshModels() {
  isLoading.value = true;
  loadError.value = '';
  try {
    const result = await api.listModels();
    const values = Array.isArray(result.models) ? result.models : [];
    models.value = values
      .map((value) => (typeof value === 'string' ? value.trim() : ''))
      .filter((value): value is string => Boolean(value))
      .map((name) => ({ name, loaded: false }));
    hasLoaded.value = true;
    lastRefreshed.value = new Date().toISOString();
  } catch (error) {
    models.value = [];
    hasLoaded.value = true;
    loadError.value = errorMessage(error);
  } finally {
    isLoading.value = false;
  }
}

async function inspectModel(name: string) {
  selectedModel.value = name;
  modelInfo.value = null;
  modelInfoError.value = '';
  isLoadingInfo.value = true;
  try {
    const result = await api.getModelInfo(name);
    modelInfo.value = result;
  } catch (error) {
    modelInfoError.value = errorMessage(error);
  } finally {
    isLoadingInfo.value = false;
  }
}

async function loadModel(name: string) {
  isLoadingModel.value = true;
  loadingModel.value = name;
  actionMessage.value = '';
  loadError.value = '';
  try {
    const result = await api.loadModel(name);
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(stringValue(result.error) || 'The backend rejected the model load request.');
    }
    models.value = models.value.map((model) => ({ ...model, loaded: model.name === name }));
    const returnedName = isRecord(result) ? stringValue(result.model_name) : '';
    actionMessage.value = `Load request accepted for ${returnedName || name}.`;
  } catch (error) {
    loadError.value = errorMessage(error);
  } finally {
    isLoadingModel.value = false;
    loadingModel.value = '';
  }
}

function clearSelection() {
  selectedModel.value = '';
  modelInfo.value = null;
  modelInfoError.value = '';
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

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === 'string' && error.trim()) return error;
  if (isRecord(error) && stringValue(error.error)) return stringValue(error.error);
  return 'The MECH backend could not complete the model request.';
}
</script>

<style scoped>
.models-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.header-actions,
.panel-heading,
.model-card-heading,
.model-actions,
.models-lower-grid,
.provenance-note {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.surface-header {
  align-items: center;
  margin-bottom: 20px;
}

.header-copy,
.model-heading-copy {
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

.is-connected .status-dot {
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

.notice-error {
  border-left-color: var(--danger);
}

.notice-success {
  border-left-color: var(--success);
}

.surface-panel {
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.registry-panel,
.detail-panel,
.contract-panel {
  padding: clamp(16px, 2.2vw, 22px);
}

.panel-heading {
  align-items: flex-start;
  margin-bottom: 18px;
}

.panel-heading h3,
.contract-panel h3,
.detail-empty h3 {
  margin: 0;
  color: var(--text);
  font-size: 17px;
  font-weight: 720;
  letter-spacing: -0.02em;
}

.source-badge,
.record-badge {
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

.record-badge {
  border-color: var(--border-light);
  color: var(--text-dim);
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.model-card {
  min-width: 0;
  padding: 13px;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--surface-2);
}

.model-card-heading {
  align-items: flex-start;
  gap: 9px;
}

.model-card-heading h4 {
  margin: 0;
  overflow: hidden;
  color: var(--text);
  font-size: 13px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-card-heading p {
  margin: 3px 0 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.4;
}

.model-mark {
  display: inline-grid;
  width: 28px;
  height: 28px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--primary);
  font: 750 11px/1 var(--font-mono);
}

.model-meta,
.detail-list,
.contract-list {
  display: grid;
  gap: 0;
  margin: 15px 0 0;
  border-top: 1px solid var(--border);
}

.model-meta div,
.detail-list div,
.contract-list div {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
}

.model-meta dt,
.model-meta dd,
.detail-list dt,
.detail-list dd,
.contract-list dt,
.contract-list dd {
  margin: 0;
  font-size: 11px;
}

.model-meta dt,
.detail-list dt,
.contract-list dt {
  color: var(--text-muted);
}

.model-meta dd,
.detail-list dd,
.contract-list dd {
  color: var(--text-dim);
  font-weight: 650;
  text-align: right;
  overflow-wrap: anywhere;
}

.mono {
  font-family: var(--font-mono);
}

.model-actions {
  align-items: center;
  justify-content: flex-end;
  flex-wrap: wrap;
  margin-top: 12px;
}

.model-actions .primary-button,
.model-actions .quiet-button {
  min-height: 29px;
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
  animation: models-spin 800ms linear infinite;
}

@keyframes models-spin {
  to { transform: rotate(360deg); }
}

.panel-footnote {
  margin: 16px 0 0;
  color: var(--text-muted);
  font-size: 10px;
}

.models-lower-grid {
  align-items: stretch;
  margin-top: 14px;
}

.models-lower-grid > * {
  flex: 1 1 0;
}

.detail-empty p:last-child {
  margin: 10px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.provenance-note {
  align-items: flex-start;
  margin: 15px 0 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.5;
}

.contract-panel {
  min-width: 0;
}

.help-copy {
  margin: 16px 0 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.55;
}

@media (max-width: 760px) {
  .models-lower-grid {
    flex-direction: column;
  }

  .models-lower-grid > * {
    flex: auto;
  }
}

@media (max-width: 580px) {
  .surface-header,
  .header-actions,
  .panel-heading {
    align-items: flex-start;
    flex-direction: column;
  }

  .header-actions {
    width: 100%;
    align-items: stretch;
  }

  .header-actions .quiet-button,
  .header-actions .surface-status {
    align-self: flex-start;
  }

  .model-actions {
    justify-content: stretch;
  }

  .model-actions button {
    flex: 1 1 0;
  }
}

@media (prefers-reduced-motion: reduce) {
  .loading-mark {
    animation: none;
  }
}
</style>
