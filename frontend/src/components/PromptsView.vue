<template>
  <main class="prompts-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Develop / Prompt workspace</p>
        <h2 class="surface-title">Prompts</h2>
        <p class="surface-description">Keep reusable prompt drafts in this browser, then run a selected draft only when the backend is available.</p>
      </div>
      <div class="header-actions">
        <span class="surface-status" :class="{ 'is-connected': backendAvailable }" role="status" aria-live="polite">
          <span class="status-dot" aria-hidden="true" />
          {{ backendStatusLabel }}
        </span>
        <button class="quiet-button" type="button" :disabled="backendChecking" @click="checkBackend">
          {{ backendChecking ? 'Checking…' : 'Check backend' }}
        </button>
      </div>
    </header>

    <div v-if="backendUnavailable" class="surface-notice notice-warning" role="status">
      <strong>Backend prompt execution is unavailable</strong>
      <p>
        Drafts below are local user notes stored in this browser. MECH will not substitute sample prompts or pretend that a
        local draft was executed while the backend is offline.
      </p>
    </div>

    <div v-if="backendError" class="surface-notice notice-error" role="alert">
      <strong>Backend check failed</strong>
      <p>{{ backendError }}</p>
    </div>

    <div v-if="storageError" class="surface-notice notice-error" role="alert">
      <strong>Local prompt storage is unavailable</strong>
      <p>{{ storageError }}</p>
    </div>

    <div v-if="actionMessage" class="surface-notice notice-success" role="status" aria-live="polite">
      <p>{{ actionMessage }}</p>
    </div>

    <div class="prompts-layout">
      <section class="surface-panel library-panel" aria-labelledby="prompt-library-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">This browser only</p>
            <h3 id="prompt-library-title">Local prompt drafts</h3>
            <p class="panel-description">User-created drafts in <code>localStorage</code>; they do not sync to the backend.</p>
          </div>
          <div class="toolbar-actions">
            <span class="source-badge">LOCAL</span>
            <button class="quiet-button" type="button" @click="createDraft">New draft</button>
          </div>
        </header>

        <div class="search-field">
          <label for="prompt-search">Find a draft</label>
          <input
            id="prompt-search"
            v-model="searchQuery"
            type="search"
            autocomplete="off"
            placeholder="Search by name or text"
            :disabled="!prompts.length"
          >
        </div>

        <div v-if="isLoading" class="panel-message" role="status" aria-live="polite">
          <span class="loading-mark" aria-hidden="true" />
          Loading local prompt drafts…
        </div>

        <ul v-else-if="filteredPrompts.length" class="prompt-list" aria-label="Local prompt drafts">
          <li v-for="prompt in filteredPrompts" :key="prompt.id" class="prompt-item">
            <button
              class="prompt-select"
              type="button"
              :class="{ 'is-selected': selectedPromptId === prompt.id }"
              :aria-pressed="selectedPromptId === prompt.id"
              @click="selectPrompt(prompt.id)"
            >
              <span class="prompt-select-heading">
                <span class="prompt-mark" aria-hidden="true">P</span>
                <span class="prompt-name">{{ prompt.name }}</span>
                <span class="local-record-badge">LOCAL DRAFT</span>
              </span>
              <span class="prompt-preview">{{ prompt.text || 'Empty draft' }}</span>
              <span class="prompt-meta">Updated {{ formatTimestamp(prompt.updatedAt) }}</span>
            </button>
            <button class="icon-button" type="button" :aria-label="`Delete ${prompt.name}`" title="Delete draft" @click="deletePrompt(prompt)">
              <span aria-hidden="true">×</span>
            </button>
          </li>
        </ul>

        <p v-else-if="storageError" class="panel-message panel-message-error">
          Local prompt drafts could not be read from this browser.
        </p>
        <p v-else-if="prompts.length" class="panel-message">
          No local drafts match “{{ searchQuery }}”.
        </p>
        <p v-else class="panel-message">
          No local prompt drafts have been created in this browser. Use “New draft” to start one.
        </p>
      </section>

      <section class="surface-panel editor-panel" aria-labelledby="prompt-editor-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Local editor</p>
            <h3 id="prompt-editor-title">{{ selectedPromptId ? 'Edit prompt draft' : 'Create prompt draft' }}</h3>
            <p class="panel-description">This editor is always local. Backend execution is a separate, explicit action.</p>
          </div>
          <span v-if="selectedPromptId" class="source-badge">LOCAL DRAFT</span>
        </header>

        <form class="editor-form" @submit.prevent="saveDraft">
          <div class="field">
            <label for="prompt-name">Draft name <span class="required-mark" aria-hidden="true">*</span></label>
            <input
              id="prompt-name"
              v-model="draftName"
              type="text"
              autocomplete="off"
              placeholder="e.g. Check a causal claim"
              required
              :disabled="!storageAvailable"
              aria-describedby="prompt-name-help"
            >
            <p id="prompt-name-help" class="field-help">Use a name you can recognize in the local library.</p>
          </div>

          <div class="field">
            <label for="prompt-model">Model for an explicit run <span class="optional-label">optional</span></label>
            <select id="prompt-model" v-model="draftModel" :disabled="!backendAvailable || !selectedPromptId">
              <option value="">Use backend default</option>
              <option v-for="model in modelOptions" :key="model" :value="model">{{ model }}</option>
            </select>
            <p class="field-help">
              {{ backendAvailable ? 'Model names come from the backend registry.' : 'Model names will appear when the backend is reachable.' }}
            </p>
          </div>

          <div class="field">
            <label for="prompt-text">Prompt text <span class="required-mark" aria-hidden="true">*</span></label>
            <textarea
              id="prompt-text"
              v-model="draftText"
              rows="9"
              placeholder="Write the prompt you want to keep or inspect."
              required
              :disabled="!storageAvailable"
              aria-describedby="prompt-text-help"
            />
            <p id="prompt-text-help" class="field-help">Saved text is never sent anywhere until you choose “Run diagnostic”.</p>
          </div>

          <p v-if="formError" class="form-message" role="alert">{{ formError }}</p>
          <div class="editor-actions">
            <button class="primary-button" type="submit" :disabled="!storageAvailable">
              Save draft
            </button>
            <button
              class="quiet-button"
              type="button"
              :disabled="!canRun || isRunning"
              :aria-describedby="!backendAvailable ? 'backend-unavailable-help' : undefined"
              @click="runDraft"
            >
              {{ isRunning ? 'Running…' : 'Run diagnostic' }}
            </button>
          </div>
          <p v-if="!backendAvailable" id="backend-unavailable-help" class="action-help">
            Execution is disabled until the backend is reachable.
          </p>
        </form>
      </section>
    </div>

    <section v-if="runResult" class="surface-panel result-panel" aria-labelledby="prompt-result-title">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">Backend response</p>
          <h3 id="prompt-result-title">Diagnostic result</h3>
          <p class="panel-description">Raw response from <code>api.infer()</code>; no interpretation is added here.</p>
        </div>
        <span class="source-badge" :class="{ 'is-local': resultProvenance === 'local' }">{{ resultProvenanceLabel }}</span>
      </header>
      <pre class="result-code">{{ formattedResult }}</pre>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { api } from '../services/api';

type UnknownRecord = Record<string, unknown>;
type BackendState = 'checking' | 'available' | 'unavailable';

interface PromptDraft {
  id: string;
  name: string;
  text: string;
  model: string;
  createdAt: string;
  updatedAt: string;
}

const STORAGE_KEY = 'mech.local.prompt-drafts.v1';
const prompts = ref<PromptDraft[]>([]);
const selectedPromptId = ref('');
const draftName = ref('');
const draftText = ref('');
const draftModel = ref('');
const searchQuery = ref('');
const modelOptions = ref<string[]>([]);
const backendState = ref<BackendState>('checking');
const backendError = ref('');
const isLoading = ref(true);
const isRunning = ref(false);
const formError = ref('');
const storageError = ref('');
const actionMessage = ref('');
const storageAvailable = ref(true);
const runResult = ref<UnknownRecord | string | null>(null);

const selectedPrompt = computed(() => prompts.value.find((prompt) => prompt.id === selectedPromptId.value) ?? null);
const filteredPrompts = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase();
  if (!query) return prompts.value;
  return prompts.value.filter((prompt) => `${prompt.name} ${prompt.text}`.toLocaleLowerCase().includes(query));
});
const backendAvailable = computed(() => backendState.value === 'available');
const backendStatusLabel = computed(() => {
  if (backendState.value === 'available') return 'Backend available';
  if (backendState.value === 'checking') return 'Checking backend';
  return 'Backend unavailable';
});
const canRun = computed(() => backendAvailable.value && Boolean(draftText.value.trim()) && !isRunning.value);
const resultProvenance = computed(() => {
  if (!isRecord(runResult.value)) return 'local';
  return stringValue(runResult.value.provenance).toLocaleLowerCase();
});
const resultProvenanceLabel = computed(() => {
  if (resultProvenance.value === 'live') return 'LIVE BACKEND';
  if (resultProvenance.value === 'seeded') return 'SEEDED BACKEND';
  return 'BACKEND RESPONSE';
});
const formattedResult = computed(() => {
  if (typeof runResult.value === 'string') return runResult.value;
  try {
    return JSON.stringify(runResult.value, null, 2);
  } catch {
    return String(runResult.value);
  }
});

onMounted(() => {
  loadPrompts();
  void checkBackend();
});

function loadPrompts() {
  isLoading.value = true;
  if (typeof localStorage === 'undefined') {
    storageAvailable.value = false;
    storageError.value = 'This runtime does not expose browser local storage.';
    isLoading.value = false;
    return;
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      prompts.value = [];
      isLoading.value = false;
      return;
    }
    const parsed: unknown = JSON.parse(stored);
    prompts.value = Array.isArray(parsed)
      ? parsed.map((value, index) => normalizePrompt(value, index)).filter((value): value is PromptDraft => value !== null)
      : [];
  } catch {
    prompts.value = [];
    storageError.value = 'The saved local prompt drafts could not be read; no sample drafts were loaded.';
  } finally {
    isLoading.value = false;
  }
}

function persistPrompts(): boolean {
  if (typeof localStorage === 'undefined') {
    storageAvailable.value = false;
    storageError.value = 'This runtime does not expose browser local storage.';
    return false;
  }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prompts.value));
    storageAvailable.value = true;
    return true;
  } catch {
    storageError.value = 'The browser could not save this local prompt draft.';
    return false;
  }
}

function createDraft() {
  selectedPromptId.value = '';
  draftName.value = '';
  draftText.value = '';
  draftModel.value = '';
  formError.value = '';
  actionMessage.value = '';
}

function selectPrompt(id: string) {
  const prompt = prompts.value.find((item) => item.id === id);
  if (!prompt) return;
  selectedPromptId.value = id;
  draftName.value = prompt.name;
  draftText.value = prompt.text;
  draftModel.value = prompt.model;
  formError.value = '';
}

function saveDraft() {
  formError.value = '';
  actionMessage.value = '';
  const name = draftName.value.trim();
  const text = draftText.value.trim();
  if (!name || !text) {
    formError.value = 'Enter a draft name and prompt text before saving.';
    return;
  }

  const now = new Date().toISOString();
  if (selectedPromptId.value) {
    prompts.value = prompts.value.map((prompt) => prompt.id === selectedPromptId.value
      ? { ...prompt, name, text, model: draftModel.value, updatedAt: now }
      : prompt);
  } else {
    const draft: PromptDraft = {
      id: makeId(),
      name,
      text,
      model: draftModel.value,
      createdAt: now,
      updatedAt: now,
    };
    prompts.value = [draft, ...prompts.value];
    selectedPromptId.value = draft.id;
  }

  if (!persistPrompts()) return;
  actionMessage.value = 'Prompt draft saved in this browser only.';
}

function deletePrompt(prompt: PromptDraft) {
  if (typeof window !== 'undefined' && !window.confirm(`Delete the local prompt draft “${prompt.name}”?`)) return;
  prompts.value = prompts.value.filter((item) => item.id !== prompt.id);
  if (!persistPrompts()) return;
  if (selectedPromptId.value === prompt.id) createDraft();
  actionMessage.value = `Deleted ${prompt.name} from this browser.`;
}

async function checkBackend() {
  backendState.value = 'checking';
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
  }
}

async function runDraft() {
  if (!canRun.value) return;
  isRunning.value = true;
  actionMessage.value = '';
  try {
    const result = await api.infer(draftText.value.trim(), draftModel.value || undefined);
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(stringValue(result.error) || 'The backend returned an error for this prompt.');
    }
    runResult.value = result;
    backendState.value = 'available';
    actionMessage.value = 'The backend accepted the prompt and returned a response.';
  } catch (error) {
    backendError.value = errorMessage(error);
    backendState.value = 'unavailable';
  } finally {
    isRunning.value = false;
  }
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return `local-prompt-${crypto.randomUUID()}`;
  return `local-prompt-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function normalizePrompt(value: unknown, index: number): PromptDraft | null {
  if (!isRecord(value)) return null;
  const name = stringValue(value.name) || stringValue(value.title);
  const text = stringValue(value.text) || stringValue(value.prompt) || stringValue(value.content);
  if (!name && !text) return null;
  return {
    id: stringValue(value.id) || `local-prompt-${index}-${name}`,
    name: name || `Draft ${index + 1}`,
    text,
    model: stringValue(value.model),
    createdAt: stringValue(value.createdAt) || stringValue(value.created_at),
    updatedAt: stringValue(value.updatedAt) || stringValue(value.updated_at) || new Date().toISOString(),
  };
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
  return 'The MECH backend could not complete that prompt request.';
}
</script>

<style scoped>
.prompts-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.header-actions,
.panel-heading,
.toolbar-actions,
.prompt-select-heading,
.editor-actions,
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

.prompts-layout {
  display: grid;
  grid-template-columns: minmax(240px, 0.72fr) minmax(0, 1.28fr);
  gap: 14px;
  align-items: start;
}

.surface-panel {
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.library-panel,
.editor-panel,
.result-panel {
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
.local-record-badge {
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

.local-record-badge {
  margin-left: auto;
  border-color: var(--border-light);
  color: var(--text-dim);
}

.toolbar-actions {
  align-items: center;
  flex: 0 0 auto;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.search-field {
  margin-bottom: 14px;
}

.search-field label,
.field label {
  display: block;
  margin-bottom: 6px;
  color: var(--text-dim);
  font-size: 12px;
  font-weight: 700;
}

.search-field input,
.field input,
.field select,
.field textarea {
  display: block;
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 6px;
  outline: 0;
  background: var(--surface);
  color: var(--text);
  font: 12px/1.45 var(--font);
}

.search-field input,
.field input,
.field select {
  min-height: 38px;
  padding: 0 10px;
}

.field textarea {
  min-height: 180px;
  padding: 9px 10px;
  resize: vertical;
}

.search-field input:focus,
.field input:focus,
.field select:focus,
.field textarea:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--accent-soft);
}

.search-field input:focus-visible,
.field input:focus-visible,
.field select:focus-visible,
.field textarea:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

.search-field input:disabled,
.field input:disabled,
.field select:disabled,
.field textarea:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.prompt-list {
  display: grid;
  gap: 7px;
  max-height: 490px;
  margin: 0;
  padding: 0 2px 0 0;
  overflow-y: auto;
  list-style: none;
}

.prompt-item {
  display: flex;
  min-width: 0;
  align-items: stretch;
  gap: 5px;
}

.prompt-select {
  display: block;
  min-width: 0;
  flex: 1;
  padding: 10px 11px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text);
  cursor: pointer;
  text-align: left;
}

.prompt-select:hover,
.prompt-select.is-selected {
  border-color: var(--primary);
  background: var(--accent-soft);
}

.prompt-select-heading {
  align-items: center;
  justify-content: flex-start;
  gap: 7px;
}

.prompt-mark {
  display: inline-grid;
  width: 23px;
  height: 23px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 5px;
  background: var(--surface);
  color: var(--primary);
  font: 750 9px/1 var(--font-mono);
}

.prompt-name {
  min-width: 0;
  overflow: hidden;
  color: var(--text);
  font-size: 12px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.prompt-preview {
  display: -webkit-box;
  margin: 8px 0 0;
  overflow: hidden;
  color: var(--text-dim);
  font-size: 11px;
  line-height: 1.4;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.prompt-meta {
  display: block;
  margin-top: 7px;
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

.icon-button:hover {
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
  animation: prompts-spin 800ms linear infinite;
}

@keyframes prompts-spin {
  to { transform: rotate(360deg); }
}

.editor-form {
  display: grid;
  gap: 14px;
}

.field {
  min-width: 0;
}

.optional-label {
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 500;
}

.required-mark {
  color: var(--danger);
}

.field-help,
.action-help {
  margin: 6px 0 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.45;
}

.form-message {
  margin: 0;
  color: var(--danger);
  font-size: 11px;
  line-height: 1.45;
}

.editor-actions {
  align-items: center;
  justify-content: flex-end;
  flex-wrap: wrap;
}

.result-panel {
  margin-top: 14px;
}

.result-code {
  max-height: 360px;
  margin: 0;
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

@media (max-width: 820px) {
  .prompts-layout {
    grid-template-columns: minmax(0, 1fr);
  }

  .prompt-list {
    max-height: none;
  }
}

@media (max-width: 580px) {
  .surface-header,
  .header-actions,
  .panel-heading,
  .editor-actions {
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

  .toolbar-actions {
    width: 100%;
    justify-content: space-between;
  }

  .prompt-select-heading {
    flex-wrap: wrap;
  }

  .local-record-badge {
    margin-left: 0;
  }
}

@media (prefers-reduced-motion: reduce) {
  .loading-mark {
    animation: none;
  }
}
</style>
