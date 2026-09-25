<template>
  <main class="plugins-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">General / Local bridge</p>
        <h2 class="surface-title">Plugins</h2>
        <p class="surface-description">Inspect the local plugin bridge and catalog contract. This window does not provide a simulated marketplace.</p>
      </div>
      <div class="header-actions">
        <span class="surface-status" :class="{ 'is-connected': bridgeAvailable }" role="status" aria-live="polite">
          <span class="status-dot" aria-hidden="true" />
          {{ bridgeAvailable ? 'Bridge detected' : 'Bridge unavailable' }}
        </span>
        <button v-if="canReadCatalog" class="quiet-button" type="button" :disabled="isLoading" @click="refreshCatalog">
          {{ isLoading ? 'Refreshing…' : 'Refresh catalog' }}
        </button>
      </div>
    </header>

    <div v-if="!bridgeAvailable" class="surface-notice notice-warning" role="status">
      <strong>Local plugin bridge unavailable</strong>
      <p>
        Plugin capabilities belong to the Electron desktop bridge. Browser preview can show the contract boundary, but it cannot
        read an installed plugin catalog or validate marketplace packages.
      </p>
    </div>

    <div v-else-if="!canReadCatalog" class="surface-notice notice-warning" role="status">
      <strong>Catalog reader not exposed</strong>
      <p>
        <code>window.appApi</code> is present, but it does not expose a supported catalog reader. No plugin records are inferred
        from bundled service classes or placeholder data.
      </p>
    </div>

    <div v-if="catalogError" class="surface-notice notice-error" role="alert">
      <strong>Plugin catalog could not be read</strong>
      <p>{{ catalogError }}</p>
    </div>

    <div v-if="actionMessage" class="surface-notice notice-success" role="status" aria-live="polite">
      <p>{{ actionMessage }}</p>
    </div>

    <div class="plugins-layout">
      <section class="surface-panel catalog-panel" aria-labelledby="plugin-catalog-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Installed records</p>
            <h3 id="plugin-catalog-title">Local plugin catalog</h3>
            <p class="panel-description">Only records returned by the active <code>window.appApi</code> reader are shown.</p>
          </div>
          <span class="source-badge">SOURCE: BRIDGE</span>
        </header>

        <div v-if="isLoading" class="panel-message" role="status" aria-live="polite">
          <span class="loading-mark" aria-hidden="true" />
          Reading the local plugin catalog…
        </div>

        <ul v-else-if="catalog.length" class="plugin-list" aria-label="Installed plugin catalog">
          <li v-for="plugin in catalog" :key="plugin.id" class="plugin-item">
            <div class="plugin-heading">
              <span class="plugin-mark" aria-hidden="true">P</span>
              <div class="plugin-heading-copy">
                <h4>{{ plugin.name }}</h4>
                <p>{{ plugin.description || 'No description supplied by the bridge.' }}</p>
              </div>
              <span class="record-badge">{{ plugin.enabled ? 'ENABLED' : 'RECORDED' }}</span>
            </div>
            <dl class="plugin-meta">
              <div><dt>ID</dt><dd class="mono">{{ plugin.id }}</dd></div>
              <div><dt>Version</dt><dd>{{ plugin.version || 'Not supplied' }}</dd></div>
              <div><dt>Author</dt><dd>{{ plugin.author || 'Not supplied' }}</dd></div>
            </dl>
            <details v-if="plugin.raw" class="raw-details">
              <summary>Raw bridge record</summary>
              <pre>{{ formatJson(plugin.raw) }}</pre>
            </details>
          </li>
        </ul>

        <p v-else-if="catalogError" class="panel-message panel-message-error">
          Plugin catalog records are unavailable for this request.
        </p>
        <p v-else-if="canReadCatalog && hasLoaded" class="panel-message">
          The local bridge returned an empty plugin catalog. No marketplace entries are substituted.
        </p>
        <p v-else-if="bridgeAvailable" class="panel-message">
          The detected bridge does not expose a catalog reader, so no plugin records are shown.
        </p>
        <p v-else class="panel-message">
          An installed plugin catalog requires the local application bridge.
        </p>
      </section>

      <aside class="surface-panel capability-panel" aria-labelledby="plugin-capabilities-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Bridge inspection</p>
            <h3 id="plugin-capabilities-title">Exposed capabilities</h3>
            <p class="panel-description">Method names are detected without invoking them.</p>
          </div>
        </header>
        <ul class="capability-list" aria-label="Detected local bridge capabilities">
          <li v-for="capability in capabilities" :key="capability.name" class="capability-row">
            <span class="capability-name">{{ capability.name }}</span>
            <span class="capability-state" :class="{ 'is-present': capability.present }">
              {{ capability.present ? 'EXPOSED' : 'NOT EXPOSED' }}
            </span>
          </li>
        </ul>
        <div class="bridge-summary">
          <div><span>Bridge object</span><strong>{{ bridgeAvailable ? 'window.appApi detected' : 'Not detected' }}</strong></div>
          <div><span>Catalog source</span><strong>{{ catalogSourceLabel }}</strong></div>
          <div><span>Marketplace</span><strong>Not offered by this surface</strong></div>
        </div>
      </aside>
    </div>

    <section class="surface-panel contract-panel" aria-labelledby="plugin-contract-title">
      <p class="panel-kicker">Integration boundary</p>
      <h3 id="plugin-contract-title">What this window does not claim</h3>
      <div class="contract-columns">
        <p>No plugin is installed, enabled, or removed from this screen. No package is downloaded, signed, or executed.</p>
        <p>A future catalog reader can be added to <code>window.appApi</code>; until then, the absence of a reader remains visible.</p>
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';

type UnknownRecord = Record<string, unknown>;
type Bridge = Record<string, unknown>;
type BridgeMethod = (...args: unknown[]) => unknown;

interface PluginRecord {
  id: string;
  name: string;
  version: string;
  author: string;
  description: string;
  enabled: boolean;
  raw: unknown;
}

const bridge = ref<Bridge | null>(resolveBridge());
const catalog = ref<PluginRecord[]>([]);
const isLoading = ref(false);
const hasLoaded = ref(false);
const catalogError = ref('');
const actionMessage = ref('');

const bridgeAvailable = computed(() => Boolean(bridge.value));
const canReadCatalog = computed(() => Boolean(readerMethod()));
const catalogSourceLabel = computed(() => bridgeAvailable.value ? 'window.appApi' : 'Unavailable');
const capabilities = computed(() => {
  const activeBridge = bridge.value;
  return [
    { name: 'listPlugins', present: hasMethod(activeBridge, 'listPlugins') },
    { name: 'getPlugins', present: hasMethod(activeBridge, 'getPlugins') },
    { name: 'listPluginCatalog', present: hasMethod(activeBridge, 'listPluginCatalog') },
    { name: 'getPluginStatus', present: hasMethod(activeBridge, 'getPluginStatus') },
    { name: 'installPlugin', present: hasMethod(activeBridge, 'installPlugin') },
    { name: 'uninstallPlugin', present: hasMethod(activeBridge, 'uninstallPlugin') },
  ];
});

onMounted(() => {
  if (canReadCatalog.value) void refreshCatalog();
  else hasLoaded.value = true;
});

function readerMethod(): BridgeMethod | null {
  const activeBridge = bridge.value;
  if (!activeBridge) return null;
  for (const name of ['listPlugins', 'getPlugins', 'listPluginCatalog']) {
    const method = getMethod(activeBridge, name);
    if (method) return method;
  }
  return null;
}

async function refreshCatalog() {
  const reader = readerMethod();
  if (!reader) {
    hasLoaded.value = true;
    return;
  }

  isLoading.value = true;
  catalogError.value = '';
  actionMessage.value = '';
  try {
    const result = await reader();
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(stringValue(result.error) || 'The local plugin bridge returned an error.');
    }
    catalog.value = extractCatalog(result)
      .map((value, index) => normalizePlugin(value, index))
      .filter((value): value is PluginRecord => value !== null);
    hasLoaded.value = true;
    actionMessage.value = 'Plugin catalog refreshed from the local bridge.';
  } catch (error) {
    catalog.value = [];
    hasLoaded.value = true;
    catalogError.value = errorMessage(error);
  } finally {
    isLoading.value = false;
  }
}

function extractCatalog(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  if (!isRecord(value)) throw new Error('The local plugin bridge returned no catalog list.');
  for (const key of ['plugins', 'catalog', 'extensions', 'registry', 'items']) {
    if (Array.isArray(value[key])) return value[key] as unknown[];
  }
  if (value.result !== undefined) return extractCatalog(value.result);
  throw new Error('The local plugin bridge returned no catalog list.');
}

function normalizePlugin(value: unknown, index: number): PluginRecord | null {
  if (typeof value === 'string') {
    const name = value.trim();
    return name ? { id: `plugin-${index}-${name}`, name, version: '', author: '', description: '', enabled: false, raw: value } : null;
  }
  if (!isRecord(value)) return null;
  const name = stringValue(value.name) || stringValue(value.label) || stringValue(value.title);
  const id = stringValue(value.id) || stringValue(value.plugin_id);
  if (!name && !id) return null;
  return {
    id: id || `plugin-${index}`,
    name: name || id || `Plugin ${index + 1}`,
    version: stringValue(value.version),
    author: stringValue(value.author),
    description: stringValue(value.description) || stringValue(value.summary),
    enabled: value.enabled === true || value.active === true || value.installed === true,
    raw: value,
  };
}

function resolveBridge(): Bridge | null {
  if (typeof window === 'undefined') return null;
  const appApi = (window as Window & { appApi?: unknown }).appApi;
  return isRecord(appApi) ? appApi : null;
}

function hasMethod(value: Bridge | null, name: string): value is Bridge & Record<string, BridgeMethod> {
  return Boolean(value && typeof value[name] === 'function');
}

function getMethod(value: Bridge | null, name: string): BridgeMethod | null {
  return hasMethod(value, name) ? value[name] : null;
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

function formatJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === 'string' && error.trim()) return error;
  if (isRecord(error) && stringValue(error.error)) return stringValue(error.error);
  return 'The local application bridge could not read the plugin catalog.';
}
</script>

<style scoped>
.plugins-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.header-actions,
.panel-heading,
.plugin-heading,
.capability-row,
.bridge-summary div,
.contract-columns {
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
.plugin-heading-copy {
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

.plugins-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(250px, 0.65fr);
  gap: 14px;
  align-items: start;
}

.surface-panel {
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.catalog-panel,
.capability-panel,
.contract-panel {
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

.source-badge,
.record-badge,
.capability-state {
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

.plugin-list {
  display: grid;
  gap: 9px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.plugin-item {
  min-width: 0;
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
}

.plugin-heading {
  align-items: flex-start;
  gap: 9px;
}

.plugin-mark {
  display: inline-grid;
  width: 27px;
  height: 27px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 5px;
  background: var(--surface);
  color: var(--primary);
  font: 750 10px/1 var(--font-mono);
}

.plugin-heading h4 {
  margin: 0;
  overflow: hidden;
  color: var(--text);
  font-size: 13px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.plugin-heading p {
  margin: 3px 0 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.45;
}

.plugin-meta {
  display: grid;
  gap: 0;
  margin: 13px 0 0;
  border-top: 1px solid var(--border);
}

.plugin-meta div {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  padding: 7px 0;
  border-bottom: 1px solid var(--border);
}

.plugin-meta dt,
.plugin-meta dd {
  margin: 0;
  font-size: 10px;
}

.plugin-meta dt {
  color: var(--text-muted);
}

.plugin-meta dd {
  color: var(--text-dim);
  font-weight: 650;
  text-align: right;
  overflow-wrap: anywhere;
}

.mono {
  font-family: var(--font-mono);
}

.raw-details {
  margin-top: 12px;
  color: var(--text-muted);
  font-size: 11px;
}

.raw-details summary {
  cursor: pointer;
  font-weight: 700;
}

.raw-details pre {
  max-height: 220px;
  margin: 8px 0 0;
  padding: 9px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 5px;
  background: var(--surface);
  color: var(--text-dim);
  font: 10px/1.5 var(--font-mono);
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
  animation: plugins-spin 800ms linear infinite;
}

@keyframes plugins-spin {
  to { transform: rotate(360deg); }
}

.capability-list {
  display: grid;
  gap: 0;
  margin: 0;
  padding: 0;
  border-top: 1px solid var(--border);
  list-style: none;
}

.capability-row {
  align-items: center;
  padding: 9px 0;
  border-bottom: 1px solid var(--border);
}

.capability-name {
  color: var(--text-dim);
  font: 11px var(--font-mono);
}

.capability-state {
  color: var(--text-muted);
}

.capability-state.is-present {
  border-color: var(--border-light);
  color: var(--text-dim);
}

.bridge-summary {
  display: grid;
  gap: 0;
  margin-top: 18px;
  border-top: 1px solid var(--border);
}

.bridge-summary div {
  align-items: baseline;
  padding: 9px 0;
  border-bottom: 1px solid var(--border);
}

.bridge-summary span {
  color: var(--text-muted);
  font-size: 11px;
}

.bridge-summary strong {
  color: var(--text-dim);
  font-size: 11px;
  text-align: right;
}

.contract-panel {
  margin-top: 14px;
}

.contract-columns {
  align-items: stretch;
  margin-top: 14px;
}

.contract-columns p {
  flex: 1 1 0;
  margin: 0;
  padding: 11px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.55;
}

@media (max-width: 820px) {
  .plugins-layout {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 580px) {
  .surface-header,
  .header-actions,
  .panel-heading,
  .contract-columns {
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

  .contract-columns p {
    width: 100%;
  }
}

@media (prefers-reduced-motion: reduce) {
  .loading-mark {
    animation: none;
  }
}
</style>
