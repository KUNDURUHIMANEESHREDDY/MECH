<template>
  <main class="recent-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Workspace / History</p>
        <h2 class="surface-title">Recent Files</h2>
        <p class="surface-description">Files opened by the desktop application, with browser-only entries kept separate.</p>
      </div>
      <span
        class="surface-status"
        :class="{ 'is-connected': bridgeAvailable }"
        role="status"
        aria-live="polite"
      >
        <span class="status-dot" aria-hidden="true" />
        {{ bridgeAvailable ? 'Bridge detected' : 'Bridge unavailable' }}
      </span>
    </header>

    <div v-if="!bridgeAvailable" class="surface-notice notice-warning" role="status">
      <strong>Local application bridge unavailable</strong>
      <p>
        Open MECH in the Electron desktop app to load application recent-file history. Files added below are
        user-created browser records and are not synced to the desktop database.
      </p>
    </div>

    <div v-if="bridgeError" class="surface-notice notice-error" role="alert">
      <strong>Recent files could not be loaded</strong>
      <p>{{ bridgeError }}</p>
    </div>

    <div v-if="actionMessage" class="surface-notice notice-success" role="status" aria-live="polite">
      <p>{{ actionMessage }}</p>
    </div>

    <div class="recent-layout">
      <section class="surface-panel records-panel" aria-labelledby="application-files-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Application history</p>
            <h3 id="application-files-title">Desktop recent files</h3>
            <p class="panel-description">Records returned by the active application bridge.</p>
          </div>
          <div v-if="bridgeCanList || bridgeCanClear" class="toolbar-actions">
            <button
              v-if="bridgeCanList"
              class="quiet-button"
              type="button"
              :disabled="isLoading"
              @click="refreshFiles"
            >
              {{ isLoading ? 'Refreshing…' : 'Refresh' }}
            </button>
            <button
              v-if="bridgeCanClear"
              class="quiet-button"
              type="button"
              :disabled="isClearing"
              @click="clearApplicationFiles"
            >
              {{ isClearing ? 'Clearing…' : 'Clear history' }}
            </button>
          </div>
        </header>

        <div v-if="isLoading" class="panel-message" role="status" aria-live="polite">
          <span class="loading-mark" aria-hidden="true" />
          Loading recent files…
        </div>

        <ul v-else-if="applicationFiles.length" class="record-list" aria-label="Desktop recent files">
          <li v-for="file in applicationFiles" :key="file.id" class="record-item">
            <div class="record-main">
              <div class="record-title-line">
                <span class="file-mark" aria-hidden="true">□</span>
                <h4>{{ file.label }}</h4>
                <span class="record-badge">APPLICATION</span>
              </div>
              <p class="record-path" :title="file.path">{{ file.path }}</p>
              <p class="record-meta">
                Opened <time v-if="file.openedAt" :datetime="file.openedAt">{{ formatTimestamp(file.openedAt) }}</time>
                <span v-else>at an unspecified time</span>
              </p>
            </div>
            <button
              v-if="bridgeCanReveal"
              class="icon-button"
              type="button"
              :aria-label="`Show ${file.label} in folder`"
              title="Show in folder"
              @click="showInFolder(file)"
            >
              <span aria-hidden="true">↗</span>
            </button>
          </li>
        </ul>

        <p v-else-if="bridgeError" class="panel-message panel-message-error">
          Application file history is unavailable for this request.
        </p>
        <p v-else-if="bridgeAvailable && bridgeCanList" class="panel-message">
          No application recent files have been recorded.
        </p>
        <p v-else-if="bridgeAvailable" class="panel-message">
          The detected bridge does not expose a recent-file reader. No application records are shown.
        </p>
        <p v-else class="panel-message">
          Application file history requires the local application bridge.
        </p>
      </section>

      <section class="surface-panel entry-panel" aria-labelledby="add-file-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Create</p>
            <h3 id="add-file-title">Add recent file</h3>
            <p class="panel-description">Only record a path here; this surface does not claim to open or validate it.</p>
          </div>
        </header>

        <form class="entry-form" @submit.prevent="addFile">
          <div class="field">
            <label for="file-label">Label <span class="optional-label">optional</span></label>
            <input
              id="file-label"
              v-model="fileLabel"
              type="text"
              autocomplete="off"
              placeholder="Defaults to the file name"
            >
          </div>
          <div class="field">
            <label for="file-path">File path</label>
            <input
              id="file-path"
              v-model="filePath"
              type="text"
              autocomplete="off"
              placeholder="C:\\research\\notes.md"
              required
            >
          </div>

          <div class="destination-note" role="note">
            <span class="destination-badge">{{ bridgeCanAdd ? 'APPLICATION BRIDGE' : 'LOCAL BROWSER' }}</span>
            <span v-if="bridgeCanAdd">This entry will be sent to the active application bridge.</span>
            <span v-else>This entry will be stored only in this browser until a bridge is available.</span>
          </div>

          <p v-if="formError" class="form-message form-message-error" role="alert">{{ formError }}</p>
          <div class="form-actions">
            <button class="primary-button" type="submit" :disabled="isSaving">
              {{ isSaving ? 'Adding…' : 'Add file' }}
            </button>
          </div>
        </form>
      </section>
    </div>

    <section class="surface-panel local-panel" aria-labelledby="local-files-title">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">This browser only</p>
          <h3 id="local-files-title">Local recent files</h3>
          <p class="panel-description">
            User-created path records stored in this browser's local storage. They are not application history and do not sync.
          </p>
        </div>
        <span class="local-badge">LOCAL</span>
      </header>

      <div data-testid="recent-files-list" class="local-list">
        <ul v-if="localFiles.length" class="record-list" aria-label="Local browser recent files">
          <li v-for="file in localFiles" :key="file.id" class="record-item">
            <div class="record-main">
              <div class="record-title-line">
                <span class="file-mark" aria-hidden="true">□</span>
                <h4>{{ file.label }}</h4>
                <span class="record-badge record-badge-local">LOCAL BROWSER</span>
              </div>
              <p class="record-path" :title="file.path">{{ file.path }}</p>
              <p class="record-meta">
                Added <time v-if="file.openedAt" :datetime="file.openedAt">{{ formatTimestamp(file.openedAt) }}</time>
                <span v-else>at an unspecified time</span>
              </p>
            </div>
            <button
              class="remove-button"
              type="button"
              :aria-label="`Remove local recent file ${file.label}`"
              @click="removeLocalFile(file)"
            >
              Remove
            </button>
          </li>
        </ul>
        <p v-else class="panel-message">No local recent files have been created in this browser.</p>
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';

type UnknownRecord = Record<string, unknown>;
type Bridge = Record<string, unknown>;
type BridgeMethod = (...args: unknown[]) => unknown;
type FileSource = 'application' | 'local';

interface FileRecord {
  id: string;
  label: string;
  path: string;
  openedAt: string;
  source: FileSource;
}

const LOCAL_FILES_KEY = 'mech.local.recent-files.v1';
const bridge = ref<Bridge | null>(resolveBridge());
const applicationFiles = ref<FileRecord[]>([]);
const localFiles = ref<FileRecord[]>([]);
const fileLabel = ref('');
const filePath = ref('');
const isLoading = ref(false);
const isSaving = ref(false);
const isClearing = ref(false);
const bridgeError = ref('');
const formError = ref('');
const actionMessage = ref('');

const bridgeAvailable = computed(() => Boolean(bridge.value));
const bridgeCanList = computed(() => hasMethod(bridge.value, 'listRecentFiles'));
const bridgeCanAdd = computed(() => hasMethod(bridge.value, 'addRecentFile'));
const bridgeCanClear = computed(() => hasMethod(bridge.value, 'clearRecentFiles'));
const bridgeCanReveal = computed(() => hasMethod(bridge.value, 'showInFolder'));

onMounted(() => {
  loadLocalFiles();
  void loadApplicationFiles();
});

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function resolveBridge(): Bridge | null {
  if (typeof window !== 'undefined') {
    const windowApi = (window as Window & { appApi?: unknown }).appApi;
    if (isRecord(windowApi)) return windowApi;
  }
  return null;
}

function hasMethod(value: Bridge | null, name: string): value is Bridge & Record<string, BridgeMethod> {
  return Boolean(value && typeof value[name] === 'function');
}

function getMethod(value: Bridge | null, name: string): BridgeMethod | null {
  if (!hasMethod(value, name)) return null;
  return value[name];
}

function stringValue(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value.trim() : fallback;
}

function recordArray(value: unknown, key: string): unknown[] {
  if (Array.isArray(value)) return value;
  if (isRecord(value)) {
    if (Array.isArray(value[key])) return value[key];
    if (value.error !== undefined) {
      throw new Error(typeof value.error === 'string' ? value.error : 'The recent-file bridge returned an error.');
    }
  }
  return [];
}

function normalizeFile(value: unknown, _index: number, source: FileSource): FileRecord | null {
  if (!isRecord(value)) return null;
  const id = stringValue(value.id);
  const path = stringValue(value.path);
  const label = stringValue(value.label) || stringValue(value.name) || path;
  if (!id || !path || !label) return null;

  return {
    id,
    label,
    path,
    openedAt: stringValue(value.openedAt) || stringValue(value.lastOpenedAt) || stringValue(value.updatedAt),
    source,
  };
}

function normalizeFiles(values: unknown[], source: FileSource): FileRecord[] {
  return values
    .map((value, index) => normalizeFile(value, index, source))
    .filter((value): value is FileRecord => value !== null);
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `local-file-${crypto.randomUUID()}`;
  }
  return `local-file-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function loadLocalFiles() {
  if (typeof localStorage === 'undefined') return;
  try {
    const stored = localStorage.getItem(LOCAL_FILES_KEY);
    if (!stored) return;
    const parsed: unknown = JSON.parse(stored);
    if (Array.isArray(parsed)) localFiles.value = normalizeFiles(parsed, 'local');
  } catch {
    localFiles.value = [];
  }
}

function persistLocalFiles() {
  if (typeof localStorage === 'undefined') return false;
  try {
    const records = localFiles.value.map(({ source: _source, ...record }) => record);
    localStorage.setItem(LOCAL_FILES_KEY, JSON.stringify(records));
    return true;
  } catch {
    formError.value = 'The browser could not save this local recent-file record.';
    return false;
  }
}

async function loadApplicationFiles() {
  const list = getMethod(bridge.value, 'listRecentFiles');
  if (!list) return;

  isLoading.value = true;
  bridgeError.value = '';
  try {
    const result = await list();
    if (!Array.isArray(result) && !isRecord(result)) {
      throw new Error('The recent-file bridge returned no record list.');
    }
    applicationFiles.value = normalizeFiles(recordArray(result, 'files'), 'application');
  } catch (error) {
    applicationFiles.value = [];
    bridgeError.value = errorMessage(error);
  } finally {
    isLoading.value = false;
  }
}

async function refreshFiles() {
  if (!bridgeCanList.value) return;
  await loadApplicationFiles();
}

async function addFile() {
  formError.value = '';
  actionMessage.value = '';
  const path = filePath.value.trim();
  if (!path) {
    formError.value = 'Enter a file path before saving.';
    return;
  }
  const label = fileLabel.value.trim() || path.split(/[\\/]/).pop() || path;

  if (bridgeCanAdd.value) {
    await addApplicationFile(path, label);
    return;
  }

  const record: FileRecord = {
    id: makeId(),
    label,
    path,
    openedAt: new Date().toISOString(),
    source: 'local',
  };
  localFiles.value = [record, ...localFiles.value];
  if (!persistLocalFiles()) return;
  fileLabel.value = '';
  filePath.value = '';
  actionMessage.value = 'File saved as a local browser record.';
}

async function addApplicationFile(path: string, label: string) {
  const add = getMethod(bridge.value, 'addRecentFile');
  if (!add) return;

  isSaving.value = true;
  try {
    const result = await add({ path, label });
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(typeof result.error === 'string' ? result.error : 'The recent-file bridge returned an error.');
    }
    const returned = normalizeFile(result, Date.now(), 'application');
    if (returned) {
      applicationFiles.value = [returned, ...applicationFiles.value.filter((file) => file.id !== returned.id)];
    } else {
      await loadApplicationFiles();
    }
    fileLabel.value = '';
    filePath.value = '';
    actionMessage.value = 'File added through the local application bridge.';
  } catch (error) {
    formError.value = errorMessage(error);
  } finally {
    isSaving.value = false;
  }
}

async function clearApplicationFiles() {
  const clear = getMethod(bridge.value, 'clearRecentFiles');
  if (!clear || isClearing.value) return;

  isClearing.value = true;
  try {
    const result = await clear();
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(typeof result.error === 'string' ? result.error : 'The recent-file bridge returned an error.');
    }
    applicationFiles.value = [];
    actionMessage.value = 'Application recent-file history cleared through the local bridge.';
  } catch (error) {
    bridgeError.value = errorMessage(error);
  } finally {
    isClearing.value = false;
  }
}

function removeLocalFile(file: FileRecord) {
  localFiles.value = localFiles.value.filter((item) => item.id !== file.id);
  if (!persistLocalFiles()) return;
  actionMessage.value = `Removed ${file.label} from this browser.`;
}

async function showInFolder(file: FileRecord) {
  const reveal = getMethod(bridge.value, 'showInFolder');
  if (!reveal || !file.path) return;
  try {
    const result = await reveal(file.path);
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(typeof result.error === 'string' ? result.error : 'The local application bridge returned an error.');
    }
  } catch (error) {
    formError.value = errorMessage(error);
  }
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === 'string' && error.trim()) return error;
  if (isRecord(error) && typeof error.error === 'string' && error.error.trim()) return error.error;
  return 'The local application bridge could not complete that recent-file action.';
}
</script>

<style scoped>
.recent-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.panel-heading,
.form-actions,
.record-title-line {
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
.record-main {
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
  background: var(--text);
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
  font-weight: 700;
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

.recent-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
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
.entry-panel,
.local-panel {
  padding: clamp(16px, 2.2vw, 22px);
}

.local-panel {
  margin-top: 14px;
}

.panel-heading {
  margin-bottom: 18px;
}

.panel-heading h3 {
  margin: 0;
  color: var(--text);
  font-size: 17px;
  font-weight: 720;
  letter-spacing: -0.02em;
}

.toolbar-actions {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 6px;
}

.record-badge,
.local-badge,
.destination-badge {
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

.local-badge {
  background: var(--surface);
  color: var(--text-dim);
}

.record-title-line {
  align-items: center;
  justify-content: flex-start;
  gap: 8px;
}

.file-mark {
  display: inline-grid;
  width: 22px;
  height: 22px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid var(--border-light);
  border-radius: 5px;
  color: var(--text-muted);
  font: 700 12px/1 var(--font-mono);
}

.record-title-line h4 {
  min-width: 0;
  margin: 0;
  overflow: hidden;
  color: var(--text);
  font-size: 13px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.record-badge-local {
  border-color: var(--border-light);
}

.record-list {
  display: grid;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.record-item {
  display: flex;
  min-width: 0;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 11px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
}

.record-path {
  margin: 6px 0 0;
  overflow: hidden;
  color: var(--text-dim);
  font: 11px/1.45 var(--font-mono);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.record-meta {
  margin: 5px 0 0;
  color: var(--text-muted);
  font-size: 10px;
}

.icon-button,
.remove-button,
.quiet-button,
.primary-button,
.secondary-button {
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
  font: 700 11px/1 var(--font);
}

.icon-button,
.remove-button,
.quiet-button {
  min-height: 30px;
  background: var(--surface);
  color: var(--text-muted);
}

.icon-button {
  width: 30px;
  padding: 0;
  font-size: 15px;
}

.remove-button,
.quiet-button {
  padding: 0 9px;
}

.icon-button:hover:not(:disabled),
.remove-button:hover:not(:disabled),
.quiet-button:hover:not(:disabled) {
  border-color: var(--border-light);
  background: var(--surface);
  color: var(--text);
}

.primary-button,
.secondary-button {
  min-height: 36px;
  padding: 0 12px;
}

.primary-button {
  border-color: var(--text);
  background: var(--text);
  color: var(--surface);
}

.primary-button:hover:not(:disabled) {
  background: var(--text-dim);
}

.secondary-button {
  background: var(--surface);
  color: var(--text-dim);
}

.icon-button:disabled,
.remove-button:disabled,
.quiet-button:disabled,
.primary-button:disabled,
.secondary-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
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
  animation: recent-spin 800ms linear infinite;
}

@keyframes recent-spin {
  to { transform: rotate(360deg); }
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

.optional-label {
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 500;
}

.field input {
  display: block;
  width: 100%;
  min-height: 38px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  outline: 0;
  background: var(--surface);
  color: var(--text);
  font: 12px/1.4 var(--font);
}

.field input:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--accent-soft);
}

.field input:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

.destination-note {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 9px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.45;
}

.destination-badge {
  flex: 0 0 auto;
}

.form-message {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
}

.form-message-error {
  color: var(--danger);
}

.form-actions {
  align-items: center;
  flex-wrap: wrap;
}

.local-list {
  min-width: 0;
}

@media (max-width: 840px) {
  .recent-layout {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 580px) {
  .surface-header,
  .panel-heading {
    align-items: flex-start;
    flex-direction: column;
  }

  .surface-status {
    align-self: flex-start;
  }

  .toolbar-actions {
    width: 100%;
  }

  .toolbar-actions button {
    flex: 1 1 0;
  }

  .record-item {
    flex-direction: column;
  }

  .record-item > .icon-button,
  .record-item > .remove-button {
    align-self: flex-end;
  }
}
</style>
