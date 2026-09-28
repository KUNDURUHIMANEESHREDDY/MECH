<template>
  <main class="build-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Develop / Local Process</p>
        <h2 class="surface-title">Build Log</h2>
        <p class="surface-description">Run supported local builds and read the output returned by the desktop bridge.</p>
      </div>
      <span
        class="surface-status"
        :class="{ 'is-connected': bridgeAvailable || apiActive }"
        role="status"
        aria-live="polite"
      >
        <span class="status-dot" aria-hidden="true" />
        {{ bridgeAvailable ? 'Bridge detected' : apiActive ? 'Backend API' : 'Bridge unavailable' }}
      </span>
    </header>

    <div v-if="!bridgeAvailable && !apiActive" class="surface-notice notice-warning" role="status">
      <strong>Local application bridge unavailable</strong>
      <p>
        Build controls and output belong to the Electron desktop process. Open MECH in its desktop shell to start a
        build or inspect build output; this browser preview cannot substitute sample logs.
      </p>
    </div>

    <div v-else-if="!canRead && !canStart && !apiActive" class="surface-notice notice-warning" role="status">
      <strong>Local application bridge unavailable</strong>
      <p>The detected bridge does not expose a build reader or a start action. No build state is shown.</p>
    </div>

    <div v-if="errorMessage" class="surface-notice notice-error" role="alert">
      <strong>Build action failed</strong>
      <p>{{ errorMessage }}</p>
    </div>

    <div v-if="actionMessage" class="surface-notice notice-success" role="status" aria-live="polite">
      <p>{{ actionMessage }}</p>
    </div>

    <div class="build-layout">
      <section class="surface-panel output-panel" aria-labelledby="build-output-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Process output</p>
            <div class="title-row">
              <h3 id="build-output-title">Build output</h3>
              <span class="build-status" :class="`status-${buildStatus}`" role="status" aria-live="polite">
                <span class="build-status-dot" aria-hidden="true" />
                {{ statusLabel }}
              </span>
            </div>
            <p class="panel-description">
              {{ buildSummary }}
              <span v-if="canListen" class="live-label">{{ liveConnected ? 'Events connected' : 'Event subscription available' }}</span>
            </p>
          </div>
          <div v-if="canRead || canClear || apiActive" class="toolbar-actions">
            <button
              v-if="canRead || apiActive"
              class="quiet-button"
              type="button"
              :disabled="isLoading"
              @click="refreshLogs"
            >
              {{ isLoading ? 'Refreshing…' : 'Refresh output' }}
            </button>
            <button
              v-if="canClear"
              class="quiet-button"
              type="button"
              :disabled="isClearing || !buildEntries.length"
              @click="clearLogs"
            >
              {{ isClearing ? 'Clearing…' : 'Clear output' }}
            </button>
          </div>
        </header>

        <form v-if="canStart || apiActive" class="start-form" @submit.prevent="startBuild">
          <div class="target-field">
            <label for="build-target">Build target</label>
            <select id="build-target" v-model="buildTarget">
              <option value="renderer">Renderer build</option>
              <option value="python">Python environment check</option>
            </select>
          </div>
          <button class="primary-button" type="submit" :disabled="isStarting || buildStatus === 'running'">
            {{ isStarting ? 'Starting…' : buildStatus === 'running' ? 'Build running' : 'Start build' }}
          </button>
        </form>

        <div v-else class="bridge-note" role="note">
          {{ bridgeAvailable ? 'The active bridge does not expose a build start action. Output remains read-only.' : apiActive ? 'The backend runs renderer builds; Python checks need the desktop bridge.' : 'The local application bridge is unavailable, so no build action can be started.' }}
        </div>

        <div v-if="isLoading" class="panel-message" role="status" aria-live="polite">
          <span class="loading-mark" aria-hidden="true" />
          Loading build output…
        </div>

        <ol v-else-if="buildEntries.length" class="build-list" aria-label="Build output entries">
          <li v-for="entry in buildEntries" :key="entry.id" class="build-entry">
            <div class="build-entry-header">
              <span class="level-badge" :class="`level-${entry.level}`">{{ titleCase(entry.level) }}</span>
              <time v-if="entry.timestamp" :datetime="entry.timestamp">{{ formatTimestamp(entry.timestamp) }}</time>
              <span v-else class="log-time-missing">Time unavailable</span>
            </div>
            <p class="build-message">{{ entry.message }}</p>
            <details v-if="entry.context" class="build-context">
              <summary>Details</summary>
              <pre>{{ entry.context }}</pre>
            </details>
          </li>
        </ol>

        <p v-else-if="errorMessage" class="panel-message panel-message-error">
          Build output is unavailable for this request.
        </p>
        <p v-else-if="canRead || apiActive" class="panel-message">
          No build output has been recorded{{ bridgeAvailable ? ' by the local application bridge' : ' by the backend yet' }}.
        </p>
        <p v-else-if="bridgeAvailable" class="panel-message">
          The active bridge does not expose a build reader. No output records are shown.
        </p>
        <p v-else class="panel-message">
          Build output requires the local application bridge.
        </p>
      </section>

      <aside class="surface-panel info-panel" aria-labelledby="build-info-title">
        <p class="panel-kicker">Build status</p>
        <h3 id="build-info-title">What this surface knows</h3>
        <dl class="info-list">
          <div>
            <dt>Process</dt>
            <dd>{{ buildTarget === 'renderer' ? 'Renderer' : 'Python check' }}</dd>
          </div>
          <div>
            <dt>Reader</dt>
            <dd>{{ canRead ? 'Available' : apiActive ? 'Backend build API' : 'Not exposed' }}</dd>
          </div>
          <div>
            <dt>Events</dt>
            <dd>{{ canListen ? (liveConnected ? 'Connected' : 'Available') : 'Not exposed' }}</dd>
          </div>
          <div>
            <dt>Output lines</dt>
            <dd>{{ buildEntries.length }}</dd>
          </div>
        </dl>
        <p class="info-help">
          Status changes only after {{ bridgeAvailable ? 'the bridge' : 'the backend' }} reports a start, event, or close. If no bridge is present, this surface
          remains unstarted rather than displaying fabricated build history.
        </p>
      </aside>
    </div>
  </main>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { api } from '../services/api';

type UnknownRecord = Record<string, unknown>;
type Bridge = Record<string, unknown>;
type BridgeMethod = (...args: unknown[]) => unknown;
type BuildStatus = 'idle' | 'running' | 'success' | 'error';

interface BuildEntry {
  id: string;
  timestamp: string;
  level: string;
  message: string;
  context: string;
}

const bridge = ref<Bridge | null>(resolveBridge());
const buildEntries = ref<BuildEntry[]>([]);
const buildTarget = ref<'renderer' | 'python'>('renderer');
const buildStatus = ref<BuildStatus>('idle');
const apiActive = ref(false);
let pollTimer: ReturnType<typeof setInterval> | null = null;
const isLoading = ref(false);
const isStarting = ref(false);
const isClearing = ref(false);
const liveConnected = ref(false);
const errorMessage = ref('');
const actionMessage = ref('');
let stopBuildEvents: (() => void) | null = null;

const bridgeAvailable = computed(() => Boolean(bridge.value));
const canRead = computed(() => hasMethod(bridge.value, 'getBuildLogs'));
const canStart = computed(() => hasMethod(bridge.value, 'startBuild'));
const canClear = computed(() => hasMethod(bridge.value, 'clearBuildLogs'));
const canListen = computed(() => hasMethod(bridge.value, 'onBuildEvent'));
const statusLabel = computed(() => {
  if (buildStatus.value === 'running') return 'Build running';
  if (buildStatus.value === 'success') return 'Last build finished';
  if (buildStatus.value === 'error') return 'Last build failed';
  return 'No active build';
});
const buildSummary = computed(() => {
  if (!buildEntries.value.length) return 'No output loaded yet.';
  return `${buildEntries.value.length} ${buildEntries.value.length === 1 ? 'line' : 'lines'} loaded.`;
});

onMounted(() => {
  void loadBuildLogs();
  subscribeToBuildEvents();
});

onBeforeUnmount(() => {
  stopPolling();
  if (stopBuildEvents) {
    stopBuildEvents();
    stopBuildEvents = null;
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

function messageFrom(value: unknown): string {
  if (typeof value === 'string') return value.trim();
  if (!isRecord(value)) return '';
  return stringValue(value.message) || stringValue(value.line) || stringValue(value.event) || contextText(value.data);
}

function normalizeEntry(value: unknown, index: number): BuildEntry | null {
  const message = messageFrom(value);
  if (!message) return null;
  if (isRecord(value)) {
    const timestamp = stringValue(value.ts) || stringValue(value.timestamp) || stringValue(value.time);
    const level = normalizeLevel(value.level || value.severity);
    const context = contextText(value.meta ?? value.details ?? value.context);
    const id = stringValue(value.id) || `build-${index}-${timestamp}-${message.slice(0, 20)}`;
    return { id, timestamp, level, message, context };
  }
  return { id: `build-${index}-${message.slice(0, 20)}`, timestamp: '', level: 'info', message, context: '' };
}

function buildLogArray(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  if (isRecord(value)) {
    if (Array.isArray(value.logs)) return value.logs;
    if (Array.isArray(value.entries)) return value.entries;
    if (typeof value.error === 'string') throw new Error(value.error);
  }
  throw new Error('The local application bridge returned no build output.');
}

async function loadBuildLogs() {
  const reader = getMethod(bridge.value, 'getBuildLogs');
  if (reader) {
    await loadBridgeLogs(reader);
    return;
  }
  await loadApiBuild();
}

async function loadBridgeLogs(reader: BridgeMethod) {
  isLoading.value = true;
  errorMessage.value = '';
  try {
    const result = await reader();
    buildEntries.value = buildLogArray(result)
      .map((value, index) => normalizeEntry(value, index))
      .filter((value): value is BuildEntry => value !== null);
    inferStatusFromEntries();
  } catch (error) {
    buildEntries.value = [];
    errorMessage.value = messageFrom(error) || 'The local application bridge could not read build output.';
  } finally {
    isLoading.value = false;
  }
}

function applyApiBuildState(build: unknown) {
  if (!isRecord(build)) return;
  const status = stringValue(build.status);
  if (status === 'running') buildStatus.value = 'running';
  else if (status === 'completed') buildStatus.value = 'success';
  else if (status === 'failed') buildStatus.value = 'error';
  else if (status === 'idle' && buildStatus.value !== 'running') buildStatus.value = 'idle';
  const tail = stringValue(build.output_tail);
  if (tail) {
    buildEntries.value = tail.split(/\r?\n/).filter((line) => line.trim() !== '').slice(-200).map(
      (line, index) => ({
        id: `backend-build-${index}-${line.slice(0, 20)}`,
        timestamp: stringValue(build.finished_at) || stringValue(build.started_at),
        level: buildStatus.value === 'error' ? 'error' : 'info',
        message: line,
        context: '',
      }),
    );
  }
  if (buildStatus.value === 'running') startPolling();
  else stopPolling();
}

async function loadApiBuild() {
  isLoading.value = true;
  errorMessage.value = '';
  try {
    const result = await api.getBackendBuild();
    if (!isRecord(result) || result.status === 'error' || !isRecord(result.build)) {
      throw new Error(
        isRecord(result) && typeof result.error === 'string'
          ? result.error
          : 'The backend returned no build state.',
      );
    }
    apiActive.value = true;
    applyApiBuildState(result.build);
  } catch (error) {
    apiActive.value = false;
    buildEntries.value = [];
    errorMessage.value = '';
  } finally {
    isLoading.value = false;
  }
}

async function refreshLogs() {
  if (!canRead.value && !apiActive.value) {
    await loadApiBuild();
    return;
  }
  await loadBuildLogs();
}

function startPolling() {
  if (pollTimer !== null) return;
  pollTimer = setInterval(() => {
    void refreshApiBuild();
  }, 5000);
}

function stopPolling() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function refreshApiBuild() {
  try {
    const result = await api.getBackendBuild();
    if (isRecord(result) && isRecord(result.build)) {
      apiActive.value = true;
      applyApiBuildState(result.build);
    }
  } catch {
    // Poll failures must not wipe a running build view; the next tick retries.
  }
}

function inferStatusFromEntries() {
  const lastEntry = buildEntries.value[buildEntries.value.length - 1];
  if (!lastEntry) return;
  if (/build_finished\s+code=0/i.test(lastEntry.message)) buildStatus.value = 'success';
  if (/build_finished\s+code=[1-9]/i.test(lastEntry.message)) buildStatus.value = 'error';
}

function subscribeToBuildEvents() {
  const subscribe = getMethod(bridge.value, 'onBuildEvent');
  if (!subscribe) return;

  try {
    const result = subscribe((value: unknown) => consumeBuildEvent(value));
    if (typeof result === 'function') {
      stopBuildEvents = result as () => void;
      liveConnected.value = true;
    } else {
      liveConnected.value = true;
    }
  } catch (error) {
    errorMessage.value = messageFrom(error) || 'The local application bridge could not subscribe to build events.';
  }
}

function consumeBuildEvent(value: unknown) {
  if (!isRecord(value)) {
    const entry = normalizeEntry(value, buildEntries.value.length);
    if (entry) appendEntry(entry);
    return;
  }

  const type = stringValue(value.type).toLocaleLowerCase();
  const data = value.data;
  if (type === 'close') {
    const code = isRecord(data) ? data.code : undefined;
    buildStatus.value = code === 0 || code === '0' ? 'success' : 'error';
  } else if (type === 'error') {
    buildStatus.value = 'error';
  } else if (type === 'stdout') {
    buildStatus.value = 'running';
  }

  const message = messageFrom(data) || (type ? `[${type}]` : '');
  if (message) {
    const entry = normalizeEntry({
      ts: stringValue(value.ts),
      level: type === 'stderr' ? 'warn' : type === 'error' ? 'error' : 'info',
      message,
    }, buildEntries.value.length);
    if (entry) appendEntry(entry);
  }
}

function appendEntry(entry: BuildEntry) {
  buildEntries.value = [...buildEntries.value, entry].slice(-1000);
}

async function startBuild() {
  if (isStarting.value || buildStatus.value === 'running') return;
  const start = getMethod(bridge.value, 'startBuild');
  if (start) {
    await startBridgeBuild(start);
    return;
  }
  await startApiBuild();
}

async function startBridgeBuild(start: BridgeMethod) {
  isStarting.value = true;
  errorMessage.value = '';
  actionMessage.value = '';
  try {
    const result = await start({ target: buildTarget.value });
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(typeof result.error === 'string' ? result.error : 'The build bridge returned an error.');
    }
    if (isRecord(result) && result.ok === false) {
      throw new Error(stringValue(result.error) || 'The build bridge declined to start a build.');
    }
    if (result === undefined || result === null) {
      throw new Error('The build bridge did not return a start result.');
    }
    buildStatus.value = 'running';
    actionMessage.value = 'Build start accepted by the local application bridge.';
  } catch (error) {
    errorMessage.value = messageFrom(error) || 'The local application bridge could not start the build.';
  } finally {
    isStarting.value = false;
  }
}

async function startApiBuild() {
  if (buildTarget.value !== 'renderer') {
    errorMessage.value = 'Python environment checks require the desktop bridge; the backend runs renderer builds only.';
    return;
  }
  isStarting.value = true;
  errorMessage.value = '';
  actionMessage.value = '';
  try {
    const result = await api.startBackendBuild();
    if (!isRecord(result) || result.status === 'busy') {
      actionMessage.value = 'A build is already running on the backend.';
      await refreshApiBuild();
      return;
    }
    if (!isRecord(result) || result.status === 'error') {
      throw new Error(
        isRecord(result) && typeof result.error === 'string'
          ? result.error
          : 'The backend declined to start a build.',
      );
    }
    apiActive.value = true;
    buildStatus.value = 'running';
    actionMessage.value = 'Renderer build started on the backend; output streams in below.';
    startPolling();
  } catch (error) {
    errorMessage.value = messageFrom(error) || 'The backend could not start the build.';
  } finally {
    isStarting.value = false;
  }
}

async function clearLogs() {
  const clear = getMethod(bridge.value, 'clearBuildLogs');
  if (!clear || isClearing.value) return;

  isClearing.value = true;
  errorMessage.value = '';
  try {
    const result = await clear();
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(typeof result.error === 'string' ? result.error : 'The build bridge returned an error.');
    }
    buildEntries.value = [];
    actionMessage.value = 'Build output cleared through the local application bridge.';
  } catch (error) {
    errorMessage.value = messageFrom(error) || 'The local application bridge could not clear build output.';
  } finally {
    isClearing.value = false;
  }
}

function titleCase(value: string): string {
  return value ? value.charAt(0).toLocaleUpperCase() + value.slice(1) : value;
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}
</script>

<style scoped>
.build-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.panel-heading,
.title-row,
.start-form,
.form-actions {
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

.build-layout {
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

.output-panel,
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

.title-row {
  align-items: center;
  justify-content: flex-start;
  gap: 9px;
}

.build-status {
  display: inline-flex;
  min-height: 21px;
  align-items: center;
  gap: 6px;
  padding: 0 7px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-muted);
  font: 700 9px/1 var(--font-mono);
  white-space: nowrap;
}

.status-running {
  border-color: var(--border-light);
  color: var(--text-dim);
}

.status-success {
  border-color: var(--success);
  color: var(--success);
}

.status-error {
  border-color: var(--danger);
  color: var(--danger);
}

.build-status-dot {
  width: 6px;
  height: 6px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: var(--text-muted);
}

.status-running .build-status-dot {
  background: var(--text);
  animation: build-pulse 1.1s ease-in-out infinite alternate;
}

.status-success .build-status-dot {
  background: var(--success);
}

.status-error .build-status-dot {
  background: var(--danger);
}

@keyframes build-pulse {
  from { opacity: 0.45; }
  to { opacity: 1; }
}

.live-label {
  display: inline-flex;
  margin-left: 8px;
  color: var(--text-dim);
  font: 650 10px/1 var(--font-mono);
}

.toolbar-actions {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 6px;
}

.quiet-button,
.primary-button {
  min-height: 30px;
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
  font: 700 11px/1 var(--font);
}

.quiet-button {
  padding: 0 10px;
  background: var(--surface);
  color: var(--text-muted);
}

.quiet-button:hover:not(:disabled) {
  border-color: var(--border-light);
  color: var(--text);
}

.start-form {
  align-items: flex-end;
  margin-bottom: 14px;
  padding: 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
}

.target-field {
  min-width: 0;
  flex: 1 1 auto;
}

.target-field label {
  display: block;
  margin-bottom: 5px;
  color: var(--text-muted);
  font: 700 10px/1.2 var(--font-mono);
  text-transform: uppercase;
}

.target-field select {
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

.target-field select:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--accent-soft);
}

.target-field select:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

.primary-button {
  min-height: 36px;
  padding: 0 12px;
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

.bridge-note {
  margin-bottom: 14px;
  padding: 10px;
  border: 1px dashed var(--border-light);
  border-radius: 6px;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.5;
}

.build-list {
  display: grid;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.build-entry {
  min-width: 0;
  padding: 11px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
}

.build-entry-header {
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

.build-message {
  margin: 8px 0 0;
  color: var(--text-dim);
  font: 12px/1.55 var(--font-mono);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.build-context {
  margin-top: 8px;
  color: var(--text-muted);
  font-size: 11px;
}

.build-context summary {
  cursor: pointer;
  font-weight: 650;
}

.build-context pre {
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

.log-time-missing {
  font-style: italic;
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
  animation: build-spin 800ms linear infinite;
}

@keyframes build-spin {
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
  .build-layout {
    grid-template-columns: minmax(0, 1fr);
  }

  .info-panel {
    min-height: 0;
  }
}

@media (max-width: 580px) {
  .surface-header,
  .panel-heading,
  .start-form {
    align-items: flex-start;
    flex-direction: column;
  }

  .surface-status {
    align-self: flex-start;
  }

  .toolbar-actions,
  .start-form .primary-button {
    width: 100%;
  }

  .toolbar-actions button,
  .start-form .primary-button {
    flex: 1 1 0;
  }
}
</style>
