<template>
  <main class="logging-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">System / Diagnostics</p>
        <h2 class="surface-title">Logging</h2>
        <p class="surface-description">Read the application log stream without inventing entries when the bridge is absent.</p>
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
        Application logs are local desktop records. Open MECH in the Electron desktop app to read them; browser preview
        cannot access the log store.
      </p>
    </div>

    <div v-else-if="!canRead && !canListen" class="surface-notice notice-warning" role="status">
      <strong>Local application bridge unavailable</strong>
      <p>
        The bridge was detected but exposes neither a log reader nor a live log subscription. No log records are shown.
      </p>
    </div>

    <div v-if="loadError" class="surface-notice notice-error" role="alert">
      <strong>Application logs could not be loaded</strong>
      <p>{{ loadError }}</p>
    </div>

    <div v-if="actionMessage" class="surface-notice notice-success" role="status" aria-live="polite">
      <p>{{ actionMessage }}</p>
    </div>

    <div class="logging-layout">
      <section class="surface-panel log-panel" aria-labelledby="application-logs-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Desktop log store</p>
            <h3 id="application-logs-title">Application logs</h3>
            <p class="panel-description">
              {{ logSummary }}
              <span v-if="canListen" class="live-label">{{ liveConnected ? 'Live updates connected' : 'Live subscription available' }}</span>
            </p>
          </div>
          <button
            v-if="canRead"
            class="quiet-button"
            type="button"
            :disabled="isLoading"
            @click="refreshLogs"
          >
            {{ isLoading ? 'Refreshing…' : 'Refresh logs' }}
          </button>
        </header>

        <div class="log-toolbar" role="search" aria-label="Filter application logs">
          <div class="toolbar-field">
            <label for="log-search">Search</label>
            <input
              id="log-search"
              v-model="searchQuery"
              type="search"
              autocomplete="off"
              placeholder="Filter messages or context"
            >
          </div>
          <div class="toolbar-field toolbar-level">
            <label for="log-level">Level</label>
            <select id="log-level" v-model="levelFilter">
              <option value="all">All levels</option>
              <option v-for="level in levelOptions" :key="level" :value="level">{{ titleCase(level) }}</option>
            </select>
          </div>
        </div>

        <div v-if="isLoading" class="panel-message" role="status" aria-live="polite">
          <span class="loading-mark" aria-hidden="true" />
          Loading application logs…
        </div>

        <ol v-else-if="filteredEntries.length" class="log-list" aria-label="Application log entries">
          <li v-for="entry in filteredEntries" :key="entry.id" class="log-entry">
            <div class="log-entry-header">
              <span class="level-badge" :class="`level-${entry.level}`">{{ titleCase(entry.level) }}</span>
              <time v-if="entry.timestamp" :datetime="entry.timestamp">{{ formatTimestamp(entry.timestamp) }}</time>
              <span v-else class="log-time-missing">Time unavailable</span>
            </div>
            <p class="log-message">{{ entry.message }}</p>
            <details v-if="entry.context" class="log-context">
              <summary>Context</summary>
              <pre>{{ entry.context }}</pre>
            </details>
          </li>
        </ol>

        <p v-else-if="loadError" class="panel-message panel-message-error">
          Log records are unavailable for this request.
        </p>
        <p v-else-if="searchQuery || levelFilter !== 'all'" class="panel-message">
          No log entries match the current filters.
        </p>
        <p v-else-if="canRead" class="panel-message">
          No log entries have been returned by the local application bridge.
        </p>
        <p v-else-if="bridgeAvailable" class="panel-message">
          The bridge does not expose a log reader. No log records are available.
        </p>
        <p v-else class="panel-message">
          Application log records require the local application bridge.
        </p>
      </section>

      <aside class="surface-panel info-panel" aria-labelledby="logging-info-title">
        <p class="panel-kicker">Reading notes</p>
        <h3 id="logging-info-title">Local diagnostics</h3>
        <dl class="info-list">
          <div>
            <dt>Reader</dt>
            <dd>{{ canRead ? 'Bridge log reader available' : 'Not exposed' }}</dd>
          </div>
          <div>
            <dt>Live stream</dt>
            <dd>{{ canListen ? (liveConnected ? 'Connected' : 'Ready to connect') : 'Not exposed' }}</dd>
          </div>
          <div>
            <dt>Entries shown</dt>
            <dd>{{ filteredEntries.length }} of {{ logEntries.length }}</dd>
          </div>
        </dl>
        <p class="info-help">
          Logs are shown exactly as returned by the active bridge. If the desktop app is closed or the bridge is unavailable,
          this surface stays empty rather than substituting sample output.
        </p>
      </aside>
    </div>
  </main>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';

type UnknownRecord = Record<string, unknown>;
type Bridge = Record<string, unknown>;
type BridgeMethod = (...args: unknown[]) => unknown;

interface LogEntry {
  id: string;
  timestamp: string;
  level: string;
  message: string;
  context: string;
}

const bridge = ref<Bridge | null>(resolveBridge());
const logEntries = ref<LogEntry[]>([]);
const searchQuery = ref('');
const levelFilter = ref('all');
const isLoading = ref(false);
const liveConnected = ref(false);
const loadError = ref('');
const actionMessage = ref('');
let stopLiveUpdates: (() => void) | null = null;

const bridgeAvailable = computed(() => Boolean(bridge.value));
const canRead = computed(() => hasMethod(bridge.value, 'getAppLogs') || hasMethod(bridge.value, 'listLogs'));
const canListen = computed(() => hasMethod(bridge.value, 'onLogEntry'));
const levelOptions = computed(() => {
  const levels = new Set(logEntries.value.map((entry) => entry.level));
  return Array.from(levels).sort();
});
const filteredEntries = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase();
  return logEntries.value.filter((entry) => {
    const matchesLevel = levelFilter.value === 'all' || entry.level === levelFilter.value;
    if (!matchesLevel) return false;
    if (!query) return true;
    return `${entry.message} ${entry.context}`.toLocaleLowerCase().includes(query);
  });
});
const logSummary = computed(() => {
  if (!logEntries.value.length) return 'No entries loaded yet.';
  return `${logEntries.value.length} ${logEntries.value.length === 1 ? 'entry' : 'entries'} loaded.`;
});

onMounted(() => {
  void loadLogs();
  subscribeToLiveLogs();
});

onBeforeUnmount(() => {
  if (stopLiveUpdates) {
    stopLiveUpdates();
    stopLiveUpdates = null;
  }
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

function normalizeLevel(value: unknown): string {
  const level = stringValue(value).toLocaleLowerCase();
  if (['info', 'warn', 'error', 'debug', 'trace'].includes(level)) return level;
  if (level === 'warning') return 'warn';
  return level || 'info';
}

function contextText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value === undefined || value === null) return '';
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function normalizeEntry(value: unknown, index: number): LogEntry | null {
  if (typeof value === 'string') {
    const message = value.trim();
    return message ? { id: `log-${index}-${message.slice(0, 20)}`, timestamp: '', level: 'info', message, context: '' } : null;
  }
  if (!isRecord(value)) return null;

  const message = stringValue(value.message) || stringValue(value.event) || stringValue(value.line);
  if (!message) return null;

  const timestamp = stringValue(value.timestamp) || stringValue(value.ts) || stringValue(value.time);
  const level = normalizeLevel(value.level || value.severity || value.type);
  const context = contextText(value.context ?? value.meta ?? value.details);
  const id = stringValue(value.id) || `log-${index}-${timestamp}-${message.slice(0, 20)}`;
  return { id, timestamp, level, message, context };
}

function logArray(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  if (isRecord(value)) {
    if (Array.isArray(value.logs)) return value.logs;
    if (Array.isArray(value.entries)) return value.entries;
    if (typeof value.error === 'string') throw new Error(value.error);
  }
  throw new Error('The local application bridge returned no log list.');
}

async function loadLogs() {
  const reader = getMethod(bridge.value, 'getAppLogs') ?? getMethod(bridge.value, 'listLogs');
  if (!reader) return;

  isLoading.value = true;
  loadError.value = '';
  actionMessage.value = '';
  try {
    const result = await reader();
    logEntries.value = logArray(result)
      .map((value, index) => normalizeEntry(value, index))
      .filter((value): value is LogEntry => value !== null);
  } catch (error) {
    logEntries.value = [];
    loadError.value = errorMessage(error);
  } finally {
    isLoading.value = false;
  }
}

async function refreshLogs() {
  if (!canRead.value) return;
  await loadLogs();
}

function subscribeToLiveLogs() {
  const subscribe = getMethod(bridge.value, 'onLogEntry');
  if (!subscribe) return;

  try {
    const result = subscribe((value: unknown) => {
      const entry = normalizeEntry(value, logEntries.value.length);
      if (!entry) return;
      logEntries.value = [entry, ...logEntries.value.filter((item) => item.id !== entry.id)].slice(0, 200);
    });
    if (typeof result === 'function') {
      stopLiveUpdates = result as () => void;
      liveConnected.value = true;
    } else {
      liveConnected.value = true;
    }
  } catch (error) {
    loadError.value = errorMessage(error);
  }
}

function titleCase(value: string): string {
  return value ? value.charAt(0).toLocaleUpperCase() + value.slice(1) : value;
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === 'string' && error.trim()) return error;
  if (isRecord(error) && typeof error.error === 'string' && error.error.trim()) return error.error;
  return 'The local application bridge could not complete that log action.';
}
</script>

<style scoped>
.logging-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.panel-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
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

.logging-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(220px, 0.55fr);
  gap: 14px;
  align-items: start;
}

.surface-panel {
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.log-panel,
.info-panel {
  padding: clamp(16px, 2.2vw, 22px);
}

.panel-heading {
  margin-bottom: 18px;
}

.panel-heading h3,
.info-panel h3 {
  margin: 0;
  color: var(--text);
  font-size: 17px;
  font-weight: 720;
  letter-spacing: -0.02em;
}

.live-label {
  display: inline-flex;
  margin-left: 8px;
  color: var(--text-dim);
  font: 650 10px/1 var(--font-mono);
}

.quiet-button {
  min-height: 30px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text-muted);
  cursor: pointer;
  font: 700 11px/1 var(--font);
}

.quiet-button:hover:not(:disabled) {
  border-color: var(--border-light);
  color: var(--text);
}

.quiet-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.log-toolbar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(130px, 0.35fr);
  gap: 10px;
  margin-bottom: 14px;
  padding: 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
}

.toolbar-field label {
  display: block;
  margin-bottom: 5px;
  color: var(--text-muted);
  font: 700 10px/1.2 var(--font-mono);
  text-transform: uppercase;
}

.toolbar-field input,
.toolbar-field select {
  display: block;
  width: 100%;
  min-height: 34px;
  padding: 0 9px;
  border: 1px solid var(--border);
  border-radius: 5px;
  outline: 0;
  background: var(--surface);
  color: var(--text);
  font: 12px/1.4 var(--font);
}

.toolbar-field input:focus,
.toolbar-field select:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--accent-soft);
}

.toolbar-field input:focus-visible,
.toolbar-field select:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

.log-list {
  display: grid;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.log-entry {
  min-width: 0;
  padding: 11px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
}

.log-entry-header {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-muted);
  font: 10px/1.2 var(--font-mono);
}

.level-badge {
  display: inline-flex;
  min-height: 20px;
  align-items: center;
  padding: 0 7px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface);
  color: var(--text-dim);
  font: 700 9px/1 var(--font-mono);
  text-transform: uppercase;
}

.level-warn {
  border-color: var(--border-light);
}

.level-error {
  border-color: var(--danger);
  color: var(--danger);
}

.level-debug,
.level-trace {
  color: var(--text-muted);
}

.log-time-missing {
  font-style: italic;
}

.log-message {
  margin: 8px 0 0;
  color: var(--text-dim);
  font: 12px/1.55 var(--font-mono);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.log-context {
  margin-top: 8px;
  color: var(--text-muted);
  font-size: 11px;
}

.log-context summary {
  cursor: pointer;
  font-weight: 650;
}

.log-context pre {
  max-height: 180px;
  margin: 7px 0 0;
  padding: 8px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--surface);
  color: var(--text-dim);
  font: 11px/1.45 var(--font-mono);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
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
  animation: logging-spin 800ms linear infinite;
}

@keyframes logging-spin {
  to { transform: rotate(360deg); }
}

.info-panel {
  min-height: 100%;
}

.info-list {
  display: grid;
  gap: 0;
  margin: 20px 0 0;
  border-top: 1px solid var(--border);
}

.info-list div {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
}

.info-list dt,
.info-list dd {
  margin: 0;
  font-size: 11px;
}

.info-list dt {
  color: var(--text-muted);
}

.info-list dd {
  color: var(--text-dim);
  font-weight: 650;
  text-align: right;
}

.info-help {
  margin: 18px 0 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.55;
}

@media (max-width: 780px) {
  .logging-layout {
    grid-template-columns: minmax(0, 1fr);
  }

  .info-panel {
    min-height: 0;
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

  .log-toolbar {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
