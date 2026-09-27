<template>
  <main class="settings-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">System / Local</p>
        <h2 class="surface-title">Settings</h2>
        <p class="surface-description">Configuration for the MECH desktop runtime.</p>
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
        Settings can be read and saved only when MECH is running in its Electron desktop shell.
        Open the packaged desktop app, or reload this window from the Electron launcher, to manage persisted settings.
      </p>
      <p v-if="providedSettings" class="notice-detail">
        The host supplied a settings snapshot. Changes are not saved without the local application bridge.
      </p>
    </div>

    <div v-if="loadError" class="surface-notice notice-error" role="alert">
      <strong>Settings could not be loaded</strong>
      <p>{{ loadError }}</p>
    </div>

    <div v-if="actionMessage" class="surface-notice notice-success" role="status" aria-live="polite">
      <p>{{ actionMessage }}</p>
    </div>

    <div v-if="!valuesLoaded && !isLoading" class="surface-notice notice-neutral" role="status">
      <strong>No persisted settings loaded</strong>
      <p v-if="bridgeAvailable && !canRead">
        The bridge was detected but does not expose a settings reader. No values below are treated as saved data.
      </p>
      <p v-else-if="!bridgeAvailable && !apiActive && !providedSettings">
        The controls are a local preview only. They do not represent records in the application database.
      </p>
    </div>

    <nav class="settings-tabs" role="tablist" aria-label="Settings sections">
      <button
        v-for="(tab, index) in tabs"
        :key="tab.id"
        :id="`settings-tab-${tab.id}`"
        class="settings-tab"
        :class="{ 'is-active': activeTab === tab.id }"
        type="button"
        role="tab"
        :aria-selected="activeTab === tab.id"
        :aria-controls="`settings-panel-${tab.id}`"
        :tabindex="activeTab === tab.id ? 0 : -1"
        :data-testid="`settings-tab-${tab.id}`"
        @click="activeTab = tab.id"
        @keydown="handleTabKeydown($event, index)"
      >
        <span>{{ tab.label }}</span>
        <span class="tab-rule" aria-hidden="true" />
      </button>
    </nav>

    <form class="settings-form" :aria-busy="isSaving" @submit.prevent="saveSettings">
      <section
        id="settings-panel-theme"
        v-show="activeTab === 'theme'"
        class="settings-panel"
        role="tabpanel"
        aria-labelledby="settings-tab-theme"
        tabindex="0"
      >
        <div class="panel-heading">
          <div>
            <p class="panel-kicker">Appearance</p>
            <h3>Theme</h3>
            <p class="panel-description">Choose the preference passed to the desktop application.</p>
          </div>
          <span class="source-badge">{{ valuesLoaded ? (settingsSource === 'api' ? 'BACKEND API' : 'PERSISTED SETTING') : 'LOCAL PREVIEW' }}</span>
        </div>

        <div class="form-grid form-grid-single">
          <div class="field">
            <label for="theme-select">Application theme</label>
            <select
              id="theme-select"
              v-model="form.theme"
              data-testid="theme-select"
              aria-describedby="theme-help"
            >
              <option value="system">System</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
            <p id="theme-help" class="field-help">
              The desktop shell remains a white tool surface; this preference is stored only when the bridge supports it.
            </p>
          </div>
        </div>
      </section>

      <section
        id="settings-panel-gpu"
        v-show="activeTab === 'gpu'"
        class="settings-panel"
        role="tabpanel"
        aria-labelledby="settings-tab-gpu"
        tabindex="0"
      >
        <div class="panel-heading">
          <div>
            <p class="panel-kicker">Compute</p>
            <h3>GPU</h3>
            <p class="panel-description">Control hardware acceleration for local model work.</p>
          </div>
          <span class="source-badge">{{ valuesLoaded ? (settingsSource === 'api' ? 'BACKEND API' : 'PERSISTED SETTING') : 'LOCAL PREVIEW' }}</span>
        </div>

        <fieldset class="settings-fieldset">
          <legend>Acceleration</legend>
          <label class="check-row" for="gpu-enabled">
            <input id="gpu-enabled" v-model="form.gpuEnabled" type="checkbox">
            <span>
              <strong>Enable GPU acceleration</strong>
              <small>Allow supported model operations to use the local accelerator.</small>
            </span>
          </label>
          <div class="field">
            <label for="gpu-acceleration">Acceleration mode</label>
            <select id="gpu-acceleration" v-model="form.acceleration">
              <option value="auto">Automatic</option>
              <option value="on">On</option>
              <option value="off">Off</option>
            </select>
          </div>
        </fieldset>
      </section>

      <section
        id="settings-panel-cache"
        v-show="activeTab === 'cache'"
        class="settings-panel"
        role="tabpanel"
        aria-labelledby="settings-tab-cache"
        tabindex="0"
      >
        <div class="panel-heading">
          <div>
            <p class="panel-kicker">Storage</p>
            <h3>Cache</h3>
            <p class="panel-description">Set the local cache behavior and retention limit.</p>
          </div>
          <span class="source-badge">{{ valuesLoaded ? (settingsSource === 'api' ? 'BACKEND API' : 'PERSISTED SETTING') : 'LOCAL PREVIEW' }}</span>
        </div>

        <fieldset class="settings-fieldset">
          <legend>Cache policy</legend>
          <label class="check-row" for="cache-enabled">
            <input id="cache-enabled" v-model="form.cacheEnabled" type="checkbox">
            <span>
              <strong>Enable local cache</strong>
              <small>Cache generated artifacts and model results when available.</small>
            </span>
          </label>
          <div class="form-grid">
            <div class="field">
              <label for="cache-size">Maximum size (MB)</label>
              <input
                id="cache-size"
                v-model.number="form.cacheMaxSizeMb"
                type="number"
                min="0"
                step="1"
                inputmode="numeric"
              >
            </div>
            <div class="field">
              <label for="cache-location">Cache location</label>
              <input
                id="cache-location"
                v-model="form.cacheLocation"
                type="text"
                autocomplete="off"
                placeholder="Uses the application default when empty"
              >
            </div>
          </div>
        </fieldset>
      </section>

      <section
        id="settings-panel-paths"
        v-show="activeTab === 'paths'"
        class="settings-panel"
        role="tabpanel"
        aria-labelledby="settings-tab-paths"
        tabindex="0"
      >
        <div class="panel-heading">
          <div>
            <p class="panel-kicker">Runtime</p>
            <h3>Paths</h3>
            <p class="panel-description">Point the desktop application at local runtimes and workspaces.</p>
          </div>
          <span class="source-badge">{{ valuesLoaded ? (settingsSource === 'api' ? 'BACKEND API' : 'PERSISTED SETTING') : 'LOCAL PREVIEW' }}</span>
        </div>

        <div class="form-grid form-grid-single">
          <div class="field">
            <label for="python-path">Python executable</label>
            <input id="python-path" v-model="form.pythonPath" type="text" autocomplete="off">
          </div>
          <div class="field">
            <label for="workspace-path">Workspace path</label>
            <input
              id="workspace-path"
              v-model="form.workspacePath"
              type="text"
              autocomplete="off"
              placeholder="Optional"
            >
          </div>
          <div class="field">
            <label for="projects-path">Projects path</label>
            <input
              id="projects-path"
              v-model="form.projectsPath"
              type="text"
              autocomplete="off"
              placeholder="Optional"
            >
          </div>
        </div>
      </section>

      <footer class="form-actions">
        <div class="action-context">
          <span class="save-indicator" :class="{ 'is-ready': canSave }" aria-hidden="true" />
          <span v-if="canSave">Changes can be saved {{ settingsSource === 'api' ? 'to the backend database' : 'to the local application database' }}.</span>
          <span v-else>Saving is disabled until a settings bridge is available.</span>
        </div>
        <div class="action-buttons">
          <button
            class="secondary-button"
            type="button"
            :disabled="!canReset || isResetting"
            @click="resetSettings"
          >
            {{ isResetting ? 'Resetting…' : 'Reset settings' }}
          </button>
          <button class="primary-button" type="submit" :disabled="!canSave || isSaving">
            {{ isSaving ? 'Saving…' : 'Save settings' }}
          </button>
        </div>
      </footer>
    </form>
  </main>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue';
import { api } from '../services/api';

type UnknownRecord = Record<string, unknown>;
type Bridge = Record<string, unknown>;
type BridgeMethod = (...args: unknown[]) => unknown;

type SettingsTab = 'theme' | 'gpu' | 'cache' | 'paths';

const props = defineProps<{
  settings?: unknown;
  onChange?: () => void;
  api?: Bridge;
}>();

const tabs: Array<{ id: SettingsTab; label: string }> = [
  { id: 'theme', label: 'Theme' },
  { id: 'gpu', label: 'GPU' },
  { id: 'cache', label: 'Cache' },
  { id: 'paths', label: 'Paths' },
];

const form = reactive({
  theme: 'system',
  gpuEnabled: true,
  acceleration: 'auto',
  cacheEnabled: true,
  cacheMaxSizeMb: 1024 as number | string,
  cacheLocation: '',
  pythonPath: '',
  workspacePath: '',
  projectsPath: '',
});

const bridge = ref<Bridge | null>(resolveBridge());
const settingsRecord = ref<UnknownRecord | null>(isRecord(props.settings) ? cloneRecord(props.settings) : null);
const settingsSource = ref<'bridge' | 'api' | null>(
  isRecord(props.settings) ? 'bridge' : null,
);
const apiActive = ref(false);
const activeTab = ref<SettingsTab>('theme');
const isLoading = ref(false);
const isSaving = ref(false);
const isResetting = ref(false);
const loadError = ref('');
const actionMessage = ref('');

const providedSettings = computed(() => isRecord(props.settings));
const bridgeAvailable = computed(() => Boolean(bridge.value));
const valuesLoaded = computed(() => settingsRecord.value !== null);
const canRead = computed(() => hasMethod(bridge.value, 'getSettings'));
const canSave = computed(() => valuesLoaded.value && (Boolean(settingsWriter(bridge.value)) || settingsSource.value === 'api'));
const canReset = computed(() => hasMethod(bridge.value, 'resetSettings'));

onMounted(() => {
  void loadSettings();
});

watch(() => props.settings, (value) => {
  if (!bridge.value && isRecord(value)) {
    settingsRecord.value = cloneRecord(value);
    populateForm(value);
  }
});

watch(() => form.theme, (value) => {
  applyTheme(value);
});

function handleTabKeydown(event: KeyboardEvent, index: number) {
  let nextIndex: number | null = null;
  if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
    nextIndex = (index + 1) % tabs.length;
  } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
    nextIndex = (index - 1 + tabs.length) % tabs.length;
  } else if (event.key === 'Home') {
    nextIndex = 0;
  } else if (event.key === 'End') {
    nextIndex = tabs.length - 1;
  }
  if (nextIndex === null) return;

  event.preventDefault();
  const nextTab = tabs[nextIndex];
  activeTab.value = nextTab.id;
  void nextTick(() => {
    document.getElementById(`settings-tab-${nextTab.id}`)?.focus();
  });
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function resolveBridge(): Bridge | null {
  if (typeof window !== 'undefined') {
    const windowApi = (window as Window & { appApi?: unknown }).appApi;
    if (isRecord(windowApi)) return windowApi;
  }
  return isRecord(props.api) ? props.api : null;
}

function hasMethod(value: Bridge | null, name: string): value is Bridge & Record<string, BridgeMethod> {
  return Boolean(value && typeof value[name] === 'function');
}

function settingsWriter(value: Bridge | null): BridgeMethod | null {
  if (!value) return null;
  const setter = value.setSettings ?? value.updateSettings;
  return typeof setter === 'function' ? setter as BridgeMethod : null;
}

function cloneValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(cloneValue);
  if (isRecord(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, cloneValue(item)]));
  }
  return value;
}

function cloneRecord(value: UnknownRecord): UnknownRecord {
  return cloneValue(value) as UnknownRecord;
}

function nestedRecord(value: unknown): UnknownRecord {
  return isRecord(value) ? value : {};
}

function stringValue(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function booleanValue(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function numberValue(value: unknown, fallback: number): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function populateForm(record: UnknownRecord) {
  const gpu = nestedRecord(record.gpu);
  const cache = nestedRecord(record.cache);
  const paths = nestedRecord(record.paths);
  const theme = stringValue(record.theme, 'system');

  form.theme = ['system', 'light', 'dark'].includes(theme) ? theme : 'system';
  form.gpuEnabled = booleanValue(gpu.enabled, true);
  form.acceleration = stringValue(gpu.acceleration, 'auto');
  form.cacheEnabled = booleanValue(cache.enabled, true);
  form.cacheMaxSizeMb = numberValue(cache.maxSizeMb, 1024);
  form.cacheLocation = stringValue(cache.location);
  form.pythonPath = stringValue(paths.python);
  form.workspacePath = stringValue(paths.workspace);
  form.projectsPath = stringValue(paths.projects);
}

function nextSettings(): UnknownRecord {
  const current = settingsRecord.value ? cloneRecord(settingsRecord.value) : {};
  const gpu = cloneRecord(nestedRecord(current.gpu));
  const cache = cloneRecord(nestedRecord(current.cache));
  const paths = cloneRecord(nestedRecord(current.paths));
  const size = numberValue(form.cacheMaxSizeMb, 1024);

  return {
    ...current,
    theme: form.theme,
    gpu: { ...gpu, enabled: form.gpuEnabled, acceleration: form.acceleration },
    cache: {
      ...cache,
      enabled: form.cacheEnabled,
      maxSizeMb: size,
      location: form.cacheLocation.trim(),
    },
    paths: {
      ...paths,
      python: form.pythonPath.trim(),
      workspace: form.workspacePath.trim(),
      projects: form.projectsPath.trim(),
    },
  };
}

async function loadSettings() {
  const activeBridge = bridge.value;
  if (activeBridge) {
    if (!hasMethod(activeBridge, 'getSettings')) {
      loadError.value = 'The local application bridge does not expose getSettings().';
      isLoading.value = false;
      return;
    }

    isLoading.value = true;
    loadError.value = '';
    actionMessage.value = '';
    try {
      const result = await activeBridge.getSettings();
      if (!isRecord(result) || result.error !== undefined) {
        throw new Error(
          isRecord(result) && typeof result.error === 'string'
            ? result.error
            : 'The bridge returned no settings object.',
        );
      }
      settingsRecord.value = cloneRecord(result);
      settingsSource.value = 'bridge';
      populateForm(result);
      applyTheme(form.theme);
      notifyChange();
    } catch (error) {
      loadError.value = errorMessage(error);
    } finally {
      isLoading.value = false;
    }
    return;
  }

  await loadSettingsFromApi();
}

async function loadSettingsFromApi() {
  isLoading.value = true;
  loadError.value = '';
  actionMessage.value = '';
  try {
    const result = await api.getBackendSettings();
    if (!isRecord(result) || result.status === 'error' || !isRecord(result.settings)) {
      throw new Error(
        isRecord(result) && typeof result.error === 'string'
          ? result.error
          : 'The backend returned no settings object.',
      );
    }
    const stored = result.settings as UnknownRecord;
    settingsRecord.value = {
      theme: stored.theme,
      gpu: { enabled: stored.gpuEnabled, acceleration: stored.acceleration },
      cache: {
        enabled: stored.cacheEnabled,
        maxSizeMb: stored.cacheMaxSizeMb,
        location: stored.cachePath,
      },
      paths: {
        python: stored.pythonPath,
        workspace: stored.workspacePath,
        projects: stored.projectsPath,
      },
    };
    settingsSource.value = 'api';
    apiActive.value = true;
    populateForm(settingsRecord.value);
    applyTheme(form.theme);
    notifyChange();
  } catch (error) {
    apiActive.value = false;
    settingsRecord.value = isRecord(props.settings) ? cloneRecord(props.settings) : null;
    settingsSource.value = isRecord(props.settings) ? 'bridge' : null;
    if (!settingsRecord.value) {
      loadError.value = '';
    } else {
      loadError.value = errorMessage(error);
    }
  } finally {
    isLoading.value = false;
  }
}

async function saveSettings() {
  const writer = settingsWriter(bridge.value);
  if (writer && canSave.value) {
    await saveSettingsViaBridge(writer);
    return;
  }
  if (settingsSource.value === 'api') {
    await saveSettingsViaApi();
  }
}

async function saveSettingsViaBridge(writer: BridgeMethod) {
  isSaving.value = true;
  actionMessage.value = '';
  loadError.value = '';
  try {
    const next = nextSettings();
    const result = await writer(next);
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(
        typeof result.error === 'string' ? result.error : 'The settings bridge returned an error.',
      );
    }
    if (isRecord(result) && Object.keys(result).length > 0) {
      settingsRecord.value = cloneRecord(result);
      populateForm(result);
    } else {
      settingsRecord.value = next;
      populateForm(next);
    }
    applyTheme(form.theme);
    actionMessage.value = 'Settings saved through the local application bridge.';
    notifyChange();
  } catch (error) {
    loadError.value = errorMessage(error);
  } finally {
    isSaving.value = false;
  }
}

async function saveSettingsViaApi() {
  isSaving.value = true;
  actionMessage.value = '';
  loadError.value = '';
  try {
    const payload = {
      theme: form.theme,
      gpuEnabled: form.gpuEnabled,
      acceleration: form.acceleration,
      cacheEnabled: form.cacheEnabled,
      cacheMaxSizeMb: numberValue(form.cacheMaxSizeMb, 1024),
      cachePath: form.cacheLocation.trim(),
      workspacePath: form.workspacePath.trim(),
      pythonPath: form.pythonPath.trim(),
      projectsPath: form.projectsPath.trim(),
    };
    const result = await api.updateBackendSettings(payload);
    if (!isRecord(result) || result.status === 'error' || !isRecord(result.settings)) {
      throw new Error(
        isRecord(result) && typeof result.error === 'string'
          ? result.error
          : 'The backend refused the settings update.',
      );
    }
    await loadSettingsFromApi();
    actionMessage.value = 'Settings saved to the backend database.';
    notifyChange();
  } catch (error) {
    loadError.value = errorMessage(error);
  } finally {
    isSaving.value = false;
  }
}

async function resetSettings() {
  if (!hasMethod(bridge.value, 'resetSettings') || isResetting.value) return;

  isResetting.value = true;
  actionMessage.value = '';
  loadError.value = '';
  try {
    const result = await bridge.value?.resetSettings();
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(
        typeof result.error === 'string' ? result.error : 'The settings bridge returned an error.',
      );
    }
    if (isRecord(result) && Object.keys(result).length > 0) {
      settingsRecord.value = cloneRecord(result);
      populateForm(result);
    } else {
      await loadSettings();
    }
    actionMessage.value = 'Settings reset through the local application bridge.';
    notifyChange();
  } catch (error) {
    loadError.value = errorMessage(error);
  } finally {
    isResetting.value = false;
  }
}

function notifyChange() {
  if (typeof props.onChange !== 'function') return;
  try {
    props.onChange();
  } catch {
    // A host callback must not make a successful bridge operation look failed.
  }
}

function applyTheme(value: string) {
  if (typeof document === 'undefined') return;
  if (value === 'system' || value === 'light' || value === 'dark') {
    document.documentElement.dataset.theme = value;
  }
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === 'string' && error.trim()) return error;
  if (isRecord(error) && typeof error.error === 'string' && error.error.trim()) return error.error;
  return 'The local application bridge returned an error while handling settings.';
}
</script>

<style scoped>
.settings-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.panel-heading,
.form-actions {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
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

.surface-status,
.source-badge,
.save-indicator {
  display: inline-flex;
  align-items: center;
  white-space: nowrap;
}

.surface-status {
  gap: 7px;
  min-height: 28px;
  padding: 0 9px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-muted);
  font: 650 10px/1 var(--font-mono);
}

.surface-status.is-connected {
  color: var(--text-dim);
}

.status-dot,
.save-indicator {
  width: 7px;
  height: 7px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: var(--text-muted);
}

.surface-status.is-connected .status-dot,
.save-indicator.is-ready {
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

.notice-neutral {
  border-left-color: var(--border-light);
}

.notice-detail {
  margin-top: 5px !important;
  color: var(--text-muted);
  font-size: 11px;
}

.settings-tabs {
  display: flex;
  max-width: 100%;
  gap: 4px;
  margin-bottom: 16px;
  overflow-x: auto;
  border-bottom: 1px solid var(--border);
  scrollbar-width: thin;
}

.settings-tab {
  position: relative;
  display: inline-flex;
  min-height: 40px;
  align-items: center;
  gap: 8px;
  padding: 0 12px;
  border: 0;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font: 650 12px/1 var(--font);
  white-space: nowrap;
}

.settings-tab:hover {
  color: var(--text);
}

.settings-tab.is-active {
  color: var(--text);
}

.tab-rule {
  display: none;
  width: 100%;
  height: 2px;
  background: var(--text);
}

.settings-tab.is-active .tab-rule {
  display: block;
}

.settings-form {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 14px;
}

.settings-panel,
.form-actions {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.settings-panel {
  min-width: 0;
  padding: clamp(16px, 2.2vw, 22px);
}

.panel-heading {
  align-items: flex-start;
  margin-bottom: 20px;
}

.panel-heading h3 {
  margin: 0;
  color: var(--text);
  font-size: 17px;
  font-weight: 720;
  letter-spacing: -0.02em;
}

.source-badge {
  min-height: 22px;
  padding: 0 8px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-muted);
  font: 700 9px/1 var(--font-mono);
  letter-spacing: 0.04em;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.form-grid-single {
  grid-template-columns: minmax(0, 1fr);
}

.field {
  min-width: 0;
}

.field label,
.settings-fieldset legend {
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
  font: 12px/1.4 var(--font);
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

.field-help {
  margin: 7px 0 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.5;
}

.settings-fieldset {
  min-width: 0;
  margin: 0;
  padding: 0;
  border: 0;
}

.settings-fieldset legend {
  margin-bottom: 12px;
  padding: 0;
}

.check-row {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 16px;
  color: var(--text);
  cursor: pointer;
}

.check-row input {
  width: 16px;
  height: 16px;
  flex: 0 0 auto;
  margin: 2px 0 0;
  accent-color: var(--text);
}

.check-row span {
  display: grid;
  gap: 3px;
}

.check-row strong {
  font-size: 12px;
}

.check-row small {
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.45;
}

.form-actions {
  align-items: center;
  padding: 12px 14px;
  background: var(--surface-2);
}

.action-context {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
  color: var(--text-muted);
  font-size: 11px;
}

.action-buttons {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 7px;
}

.primary-button,
.secondary-button {
  min-height: 34px;
  padding: 0 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
  font: 700 11px/1 var(--font);
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

.secondary-button:hover:not(:disabled) {
  border-color: var(--border-light);
  background: var(--surface-2);
  color: var(--text);
}

.primary-button:disabled,
.secondary-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

@media (max-width: 680px) {
  .surface-header,
  .panel-heading,
  .form-actions {
    align-items: stretch;
    flex-direction: column;
  }

  .surface-status {
    align-self: flex-start;
  }

  .form-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .action-buttons {
    width: 100%;
  }

  .action-buttons button {
    flex: 1 1 0;
  }
}
</style>
