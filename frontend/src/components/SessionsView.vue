<template>
  <main class="sessions-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Research / Backend records</p>
        <h2 class="surface-title">Sessions</h2>
        <p class="surface-description">Create and inspect session records kept by the MECH backend, including the prompt and metadata you provide.</p>
      </div>
      <div class="header-actions">
        <span class="surface-status" :class="{ 'is-connected': isConnected }" role="status" aria-live="polite">
          <span class="status-dot" aria-hidden="true" />
          {{ isLoading ? 'Loading backend' : isConnected ? 'Backend connected' : 'Backend unavailable' }}
        </span>
        <button class="quiet-button" type="button" :disabled="isLoading" @click="refreshSessions">
          {{ isLoading ? 'Refreshing…' : 'Refresh records' }}
        </button>
      </div>
    </header>

    <div v-if="loadError" class="surface-notice notice-error" role="alert">
      <strong>Sessions could not be loaded</strong>
      <p>{{ loadError }}</p>
    </div>

    <div v-if="actionError" class="surface-notice notice-error" role="alert">
      <strong>Session action failed</strong>
      <p>{{ actionError }}</p>
    </div>

    <div v-if="actionMessage" class="surface-notice notice-success" role="status" aria-live="polite">
      <p>{{ actionMessage }}</p>
    </div>

    <div class="sessions-layout">
      <section class="surface-panel records-panel" aria-labelledby="session-records-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Application storage</p>
            <h3 id="session-records-title">Backend session records</h3>
            <p class="panel-description">Records returned by <code>GET /api/sessions</code>; no local session data is substituted.</p>
          </div>
          <span class="source-badge">SOURCE: API</span>
        </header>

        <div v-if="isLoading" class="panel-message" role="status" aria-live="polite">
          <span class="loading-mark" aria-hidden="true" />
          Loading session records…
        </div>

        <ul v-else-if="sessions.length" class="session-list" aria-label="Backend session records">
          <li v-for="session in sessions" :key="session.id" class="session-item">
            <button
              class="session-select"
              type="button"
              :class="{ 'is-selected': selectedSessionId === session.id }"
              :aria-pressed="selectedSessionId === session.id"
              @click="selectSession(session.id)"
            >
              <span class="session-heading">
                <span class="session-mark" aria-hidden="true">S</span>
                <span class="session-title">{{ session.name }}</span>
                <span class="record-badge">BACKEND</span>
              </span>
              <span class="session-detail">Model: {{ session.model || 'Not supplied' }}</span>
              <span class="session-prompt">{{ session.prompt || 'No prompt stored' }}</span>
              <span class="session-meta">Updated {{ formatTimestamp(session.updatedAt || session.createdAt) }}</span>
            </button>
            <button
              class="icon-button"
              type="button"
              :aria-label="`Delete session ${session.name}`"
              title="Delete session"
              :disabled="isDeleting"
              @click="deleteSession(session)"
            >
              <span aria-hidden="true">×</span>
            </button>
          </li>
        </ul>

        <p v-else-if="loadError" class="panel-message panel-message-error">
          Session records are unavailable for this request.
        </p>
        <p v-else-if="hasLoaded" class="panel-message">
          The backend returned no session records. Create one only when you have a real record to save.
        </p>
        <p v-else class="panel-message">
          Session records require a reachable MECH backend.
        </p>

        <footer v-if="lastRefreshed" class="panel-footnote">
          Last refreshed <time :datetime="lastRefreshed">{{ formatTimestamp(lastRefreshed) }}</time>.
        </footer>
      </section>

      <section class="surface-panel form-panel" aria-labelledby="create-session-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Write record</p>
            <h3 id="create-session-title">Create session</h3>
            <p class="panel-description">Saved fields are sent to the backend only when you choose “Save session”.</p>
          </div>
        </header>

        <form class="entry-form" @submit.prevent="createSession">
          <div class="field">
            <label for="session-name">Name <span class="required-mark" aria-hidden="true">*</span></label>
            <input id="session-name" v-model="newName" type="text" autocomplete="off" placeholder="A descriptive session name" required>
          </div>
          <div class="field">
            <label for="session-model">Model <span class="optional-label">optional</span></label>
            <input id="session-model" v-model="newModel" type="text" autocomplete="off" placeholder="e.g. gpt2-small">
          </div>
          <div class="field">
            <label for="session-prompt">Prompt <span class="optional-label">optional</span></label>
            <textarea id="session-prompt" v-model="newPrompt" rows="5" placeholder="Record the prompt associated with this session." />
          </div>
          <div class="field">
            <label for="session-metadata">Metadata <span class="optional-label">optional JSON</span></label>
            <textarea id="session-metadata" v-model="newMetadata" rows="4" placeholder='{"source":"user note"}' />
          </div>
          <p v-if="formError" class="form-message" role="alert">{{ formError }}</p>
          <div class="form-actions">
            <button class="primary-button" type="submit" :disabled="isSaving">
              {{ isSaving ? 'Saving…' : 'Save session' }}
            </button>
            <button class="quiet-button" type="button" :disabled="isSaving" @click="clearForm">Clear</button>
          </div>
        </form>
      </section>
    </div>

    <section v-if="selectedSession" class="surface-panel detail-panel" aria-labelledby="session-detail-title">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">Selected backend record</p>
          <h3 id="session-detail-title">{{ selectedSession.name }}</h3>
          <p class="panel-description">Fields are shown exactly as returned by the backend.</p>
        </div>
        <button class="quiet-button" type="button" @click="selectedSessionId = ''">Clear selection</button>
      </header>
      <dl class="detail-list">
        <div><dt>ID</dt><dd class="mono">{{ selectedSession.id }}</dd></div>
        <div><dt>Name</dt><dd>{{ selectedSession.name }}</dd></div>
        <div><dt>Model</dt><dd>{{ selectedSession.model || 'Not supplied' }}</dd></div>
        <div><dt>Prompt</dt><dd>{{ selectedSession.prompt || 'Not supplied' }}</dd></div>
        <div><dt>Created</dt><dd>{{ formatTimestamp(selectedSession.createdAt) }}</dd></div>
        <div><dt>Updated</dt><dd>{{ formatTimestamp(selectedSession.updatedAt) }}</dd></div>
      </dl>
      <details v-if="selectedMetadataText" class="raw-details">
        <summary>Stored metadata</summary>
        <pre>{{ selectedMetadataText }}</pre>
      </details>
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

interface SessionRecord {
  id: string;
  name: string;
  model: string;
  prompt: string;
  metadata: unknown;
  createdAt: string;
  updatedAt: string;
  raw: unknown;
}

const sessions = ref<SessionRecord[]>([]);
const selectedSessionId = ref('');
const newName = ref('');
const newModel = ref('');
const newPrompt = ref('');
const newMetadata = ref('');
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
const selectedSession = computed(() => sessions.value.find((item) => item.id === selectedSessionId.value) ?? null);
const selectedRawText = computed(() => formatJson(selectedSession.value?.raw));
const selectedMetadataText = computed(() => formatJson(selectedSession.value?.metadata));

onMounted(() => {
  void refreshSessions();
});

async function refreshSessions() {
  isLoading.value = true;
  loadError.value = '';
  try {
    const result = await api.listSessions();
    sessions.value = (Array.isArray(result) ? result : [])
      .map((value, index) => normalizeSession(value, index))
      .filter((value): value is SessionRecord => value !== null);
    hasLoaded.value = true;
    lastRefreshed.value = new Date().toISOString();
    if (selectedSessionId.value && !selectedSession.value) selectedSessionId.value = '';
  } catch (error) {
    sessions.value = [];
    hasLoaded.value = true;
    loadError.value = errorMessage(error);
  } finally {
    isLoading.value = false;
  }
}

function selectSession(id: string) {
  selectedSessionId.value = id;
}

async function createSession() {
  formError.value = '';
  actionError.value = '';
  actionMessage.value = '';
  const name = newName.value.trim();
  if (!name) {
    formError.value = 'Enter a session name before saving.';
    return;
  }

  let metadata: unknown = {};
  if (newMetadata.value.trim()) {
    try {
      metadata = JSON.parse(newMetadata.value);
    } catch {
      formError.value = 'Metadata must be valid JSON when provided.';
      return;
    }
  }

  isSaving.value = true;
  try {
    const result = await api.createSession({
      id: makeId('session'),
      name,
      model: newModel.value.trim(),
      prompt: newPrompt.value.trim(),
      createdAt: new Date().toISOString(),
      metadata,
    });
    if (isRecord(result) && result.error !== undefined) throw new Error(stringValue(result.error) || 'The backend rejected the session record.');
    await refreshSessions();
    clearForm();
    actionMessage.value = 'Session record saved to the backend.';
  } catch (error) {
    actionError.value = errorMessage(error);
  } finally {
    isSaving.value = false;
  }
}

async function deleteSession(session: SessionRecord) {
  if (typeof window !== 'undefined' && !window.confirm(`Delete backend session “${session.name}”?`)) return;
  isDeleting.value = true;
  actionError.value = '';
  actionMessage.value = '';
  try {
    const result = await api.deleteSession(session.id);
    if (isRecord(result) && stringValue(result.status).toLocaleLowerCase() === 'not_found') {
      throw new Error('The backend no longer contains that session record.');
    }
    sessions.value = sessions.value.filter((item) => item.id !== session.id);
    if (selectedSessionId.value === session.id) selectedSessionId.value = '';
    actionMessage.value = `Deleted ${session.name} from the backend.`;
  } catch (error) {
    actionError.value = errorMessage(error);
  } finally {
    isDeleting.value = false;
  }
}

function clearForm() {
  newName.value = '';
  newModel.value = '';
  newPrompt.value = '';
  newMetadata.value = '';
  formError.value = '';
}

function normalizeSession(value: unknown, index: number): SessionRecord | null {
  if (!isRecord(value)) return null;
  const id = stringValue(value.id) || stringValue(value.item_id) || stringValue(value.session_id);
  const name = stringValue(value.name) || stringValue(value.title) || stringValue(value.session_name);
  if (!id && !name) return null;
  return {
    id: id || `session-record-${index}`,
    name: name || id || `Session ${index + 1}`,
    model: stringValue(value.model),
    prompt: stringValue(value.prompt),
    metadata: value.metadata,
    createdAt: stringValue(value.createdAt) || stringValue(value.created_at),
    updatedAt: stringValue(value.updatedAt) || stringValue(value.updated_at),
    raw: value,
  };
}

function makeId(prefix: string): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return `${prefix}-${crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function formatJson(value: unknown): string {
  if (value === undefined || value === null || value === '') return '';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
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
  return 'The MECH backend could not complete that session request.';
}
</script>

<style scoped>
.sessions-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.header-actions,
.panel-heading,
.session-heading,
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

.sessions-layout {
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
  margin-left: auto;
  border-color: var(--border-light);
  color: var(--text-dim);
}

.session-list {
  display: grid;
  gap: 7px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.session-item {
  display: flex;
  min-width: 0;
  gap: 5px;
}

.session-select {
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

.session-select:hover,
.session-select.is-selected {
  border-color: var(--primary);
  background: var(--accent-soft);
}

.session-heading {
  align-items: center;
  justify-content: flex-start;
  gap: 8px;
}

.session-mark {
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

.session-title {
  min-width: 0;
  overflow: hidden;
  color: var(--text);
  font-size: 12px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-detail,
.session-prompt,
.session-meta {
  display: block;
  margin: 8px 0 0 33px;
  color: var(--text-dim);
  font-size: 11px;
  line-height: 1.4;
}

.session-prompt {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-meta {
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
  animation: sessions-spin 800ms linear infinite;
}

@keyframes sessions-spin {
  to { transform: rotate(360deg); }
}

.panel-footnote {
  margin: 16px 0 0;
  color: var(--text-muted);
  font-size: 10px;
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

.field input {
  min-height: 38px;
  padding: 0 10px;
}

.field textarea {
  padding: 9px 10px;
  resize: vertical;
}

.field input:focus,
.field textarea:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--accent-soft);
}

.field input:focus-visible,
.field textarea:focus-visible {
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
  .sessions-layout {
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
