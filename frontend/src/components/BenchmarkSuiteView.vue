<template>
  <section class="suite-tool" aria-labelledby="benchmark-suite-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">Develop / Evaluation</p>
        <h1 id="benchmark-suite-title">Benchmark suite</h1>
        <p class="tool-intro">
          Assemble a session suite from the backend benchmark catalog. The current runtime exposes an individual run
          endpoint, not a dedicated suite endpoint, so this tool reports that boundary instead of presenting a
          fabricated suite score.
        </p>
      </div>
      <div class="tool-header__actions">
        <span class="source-pill" :class="`source-pill--${sourceTone}`" role="status">
          <span class="source-dot" aria-hidden="true" />
          {{ sourceLabel }}
        </span>
        <button class="tool-button" type="button" :disabled="loadingCatalog" @click="loadCatalog">
          <span aria-hidden="true">↻</span>
          Refresh catalog
        </button>
      </div>
    </header>

    <div class="source-strip" role="status" aria-live="polite">
      <span class="source-strip__label">Execution contract</span>
      <code>GET /api/benchmarks</code>
      <code>POST /api/benchmarks/run</code>
      <span class="source-strip__detail">Selected benchmarks run sequentially through the individual endpoint.</span>
    </div>

    <div v-if="errorMessage" class="tool-notice" :class="`tool-notice--${offline ? 'warning' : 'error'}`" role="alert">
      <strong>{{ offline ? 'Benchmark service offline' : 'Suite request failed' }}</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <section class="metric-strip" aria-label="Suite session observations">
      <div class="metric-tile">
        <span class="metric-tile__value">{{ catalog.length }}</span>
        <span class="metric-tile__label">Catalog benchmarks</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ selectedBenchmarks.length }}</span>
        <span class="metric-tile__label">Selected for session</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ results.length }}</span>
        <span class="metric-tile__label">Responses received</span>
      </div>
      <div class="metric-tile metric-tile--note">
        <span class="metric-tile__label">Suite boundary</span>
        <span class="metric-tile__caption">No aggregate pass rate or suite ranking is calculated by this view.</span>
      </div>
    </section>

    <div class="suite-layout">
      <section class="tool-panel selection-panel" aria-labelledby="suite-selection-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Backend catalog</p>
            <h2 id="suite-selection-title">Select benchmarks</h2>
          </div>
          <label class="select-all"><input v-model="selectAll" type="checkbox" :disabled="!catalog.length" /> Select all</label>
        </header>
        <div v-if="loadingCatalog" class="state-block state-block--loading" role="status" aria-live="polite">
          <span class="spinner" aria-hidden="true" />
          <span>Reading the benchmark catalog…</span>
        </div>
        <div v-else-if="offline" class="state-block" role="alert">
          <span class="state-mark" aria-hidden="true">!</span>
          <strong>Catalog unavailable</strong>
          <span>Reconnect the runtime to assemble a suite.</span>
        </div>
        <div v-else-if="!catalog.length" class="state-block">
          <span class="state-mark" aria-hidden="true">∅</span>
          <strong>No benchmarks returned</strong>
          <span>No individual benchmark names are available to select.</span>
        </div>
        <fieldset v-else class="benchmark-options">
          <legend class="desktop-sr-only">Benchmarks to run in this session</legend>
          <label v-for="name in catalog" :key="name" class="benchmark-option" :class="{ 'benchmark-option--selected': selectedBenchmarks.includes(name) }">
            <input v-model="selectedBenchmarks" type="checkbox" :value="name" :disabled="running" />
            <span><strong>{{ name }}</strong><small>Catalog entry returned by backend</small></span>
          </label>
        </fieldset>
        <div class="selection-actions">
          <button class="tool-button tool-button--primary" type="button" :disabled="!selectedBenchmarks.length || running || offline" @click="runSuite">
            <span v-if="running" class="spinner" aria-hidden="true" />
            <span v-else aria-hidden="true">▶</span>
            {{ running ? `Running ${currentName || 'suite'}…` : 'Run selected benchmarks' }}
          </button>
          <button v-if="results.length" class="tool-button" type="button" :disabled="running" @click="clearResults">Clear session results</button>
        </div>
      </section>

      <section class="tool-panel contract-panel" aria-labelledby="suite-contract-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">What this tool can promise</p>
            <h2 id="suite-contract-title">Execution contract</h2>
          </div>
        </header>
        <ol class="contract-list">
          <li><span>01</span><div><strong>Catalog</strong><p>Names come from <code>GET /api/benchmarks</code>.</p></div></li>
          <li><span>02</span><div><strong>Execution</strong><p>Each selected name is sent to <code>POST /api/benchmarks/run</code>.</p></div></li>
          <li><span>03</span><div><strong>Interpretation</strong><p>Only returned fields are displayed; absent metrics remain unavailable.</p></div></li>
        </ol>
        <div class="contract-note"><strong>Not a suite API</strong><span>A backend-native suite endpoint is not currently mounted. This session runner is an explicit composition of individual calls.</span></div>
      </section>
    </div>

    <section class="tool-panel results-panel" aria-labelledby="suite-results-title">
      <header class="panel-header">
        <div>
          <p class="section-kicker">Session activity · not persisted</p>
          <h2 id="suite-results-title">Run responses</h2>
        </div>
        <span class="count-label">{{ results.length }} responses</span>
      </header>
      <div v-if="results.length" class="results-list">
        <article v-for="result in results" :key="result.key" class="result-row" :class="`result-row--${result.tone}`">
          <div class="result-row__header"><strong>{{ result.name }}</strong><span>{{ result.status }}</span></div>
          <dl class="result-row__metrics">
            <div><dt>Score</dt><dd>{{ formatValue(result.score) }}</dd></div>
            <div><dt>Pass rate</dt><dd>{{ formatValue(result.passRate) }}</dd></div>
            <div><dt>Provenance</dt><dd>{{ result.provenance || 'Unavailable' }}</dd></div>
            <div><dt>Field provenance</dt><dd>{{ formatFieldProvenance(result.fieldProvenance) }}</dd></div>
          </dl>
          <p v-if="result.error || result.reason" class="result-row__error">{{ result.error || result.reason }}</p>
        </article>
      </div>
      <div v-else class="inline-state">No suite responses have been received in this window.</div>
    </section>

    <footer class="tool-footer">
      <span>Related tool</span>
      <a href="#benchmark">Open single-benchmark dashboard <span aria-hidden="true">↗</span></a>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { api } from '../services/api';

type SourceTone = 'online' | 'offline' | 'loading' | 'partial';
type ResultTone = 'success' | 'error' | 'neutral';
type JsonRecord = Record<string, unknown>;
type SuiteResult = { key: string; name: string; status: string; score: unknown; passRate: unknown; provenance: string; fieldProvenance: Record<string, string>; error: string; reason: string; tone: ResultTone };

const catalog = ref<string[]>([]);
const selectedBenchmarks = ref<string[]>([]);
const selectAll = ref(false);
const results = ref<SuiteResult[]>([]);
const loadingCatalog = ref(true);
const running = ref(false);
const currentName = ref('');
const offline = ref(false);
const errorMessage = ref('');

function isRecord(value: unknown): value is JsonRecord { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function text(value: unknown, fallback = ''): string { return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? String(value) : fallback; }
function messageOf(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function isOffline(message: string): boolean { return /offline|failed to fetch|network|load|connection|timeout/i.test(message); }
function formatValue(value: unknown): string { return value === undefined || value === null || value === '' ? 'Unavailable' : typeof value === 'number' ? String(value) : text(value); }
function fieldProvenanceOf(value: unknown): Record<string, string> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, text(item, 'unavailable')]));
}
function formatFieldProvenance(value: Record<string, string>): string {
  const entries = Object.entries(value);
  return entries.length ? entries.map(([key, item]) => `${key}: ${item}`).join(' · ') : 'Unavailable';
}
function resultTone(status: string, error: string): ResultTone { const normalized = status.toLowerCase(); if (normalized === 'unavailable') return 'neutral'; if (['completed', 'complete', 'ok', 'success', 'passed'].includes(normalized)) return 'success'; if (['error', 'failed', 'failure'].includes(normalized) || error) return 'error'; return 'neutral'; }

const sourceTone = computed<SourceTone>(() => loadingCatalog.value ? 'loading' : offline.value ? 'offline' : errorMessage.value ? 'partial' : 'online');
const sourceLabel = computed(() => sourceTone.value === 'loading' ? 'Loading catalog' : sourceTone.value === 'offline' ? 'Offline / unavailable' : sourceTone.value === 'partial' ? 'Partially available' : 'Benchmark API reachable');

watch(selectedBenchmarks, (value) => {
  selectAll.value = catalog.value.length > 0 && value.length === catalog.value.length;
}, { deep: true });
watch(selectAll, (value) => {
  if (value) selectedBenchmarks.value = [...catalog.value];
  else if (!selectedBenchmarks.value.some((item) => !catalog.value.includes(item))) selectedBenchmarks.value = [];
});

async function loadCatalog(): Promise<void> {
  loadingCatalog.value = true;
  offline.value = false;
  errorMessage.value = '';
  try {
    catalog.value = await api.listBenchmarks();
    selectedBenchmarks.value = selectedBenchmarks.value.filter((name) => catalog.value.includes(name));
    if (!selectedBenchmarks.value.length && catalog.value.length) selectedBenchmarks.value = [catalog.value[0]];
  } catch (error) {
    catalog.value = [];
    selectedBenchmarks.value = [];
    errorMessage.value = messageOf(error);
    offline.value = isOffline(errorMessage.value);
  } finally {
    loadingCatalog.value = false;
  }
}

async function runSuite(): Promise<void> {
  if (running.value || !selectedBenchmarks.value.length) return;
  running.value = true;
  errorMessage.value = '';
  for (const name of selectedBenchmarks.value) {
    currentName.value = name;
    try {
      const raw = await api.runBenchmark(name);
      const record = isRecord(raw) ? raw : {};
      const status = text(record.status, 'response received');
      const reason = text(record.reason);
      const error = text(record.error);
      const provenance = text(record.provenance, 'unavailable');
      const fieldProvenance = fieldProvenanceOf(record.field_provenance);
      results.value.push({ key: `${name}-${Date.now()}-${Math.random()}`, name, status, score: record.score, passRate: record.pass_rate, provenance, fieldProvenance, error, reason, tone: resultTone(status, error || reason) });
    } catch (error) {
      const message = messageOf(error);
      offline.value = isOffline(message);
      results.value.push({ key: `${name}-${Date.now()}-${Math.random()}`, name, status: 'request failed', score: undefined, passRate: undefined, provenance: 'unavailable', fieldProvenance: {}, error: message, reason: '', tone: 'error' });
      errorMessage.value = `${name}: ${message}`;
    }
  }
  currentName.value = '';
  running.value = false;
}

function clearResults(): void {
  results.value = [];
  errorMessage.value = '';
}

onMounted(() => { void loadCatalog(); });
</script>

<style scoped>
.suite-tool { display: flex; width: 100%; min-height: 100%; flex-direction: column; gap: 14px; padding: 22px; color: var(--text); font-size: 13px; }
.tool-header, .tool-header__actions, .panel-header, .tool-footer, .result-row__header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.tool-header { align-items: flex-start; }
.tool-header__copy { min-width: 0; }
.tool-eyebrow, .section-kicker, .source-strip__label, .metric-tile__label, .count-label, .tool-footer > span:first-child, .select-all, .result-row__header span { color: var(--text-muted); font: 700 10px/1.2 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; }
.tool-eyebrow, .section-kicker { margin: 0 0 5px; }
.tool-header h1 { margin: 0; color: var(--text); font-size: clamp(22px, 3vw, 30px); font-weight: 720; letter-spacing: -.035em; line-height: 1.1; }
.tool-intro { max-width: 760px; margin: 8px 0 0; color: var(--text-muted); line-height: 1.55; }
.tool-header__actions { flex: 0 0 auto; flex-wrap: wrap; justify-content: flex-end; }
.source-pill { display: inline-flex; min-height: 25px; align-items: center; gap: 6px; border: 1px solid var(--border); border-radius: 999px; background: var(--surface-2); padding: 0 9px; white-space: nowrap; }
.source-pill--online { border-color: #b9dfd3; background: var(--success-soft); color: var(--success); }
.source-pill--partial { border-color: #ecd79c; background: var(--warning-soft); color: var(--warning); }
.source-pill--offline { border-color: #efc2c7; background: var(--danger-soft); color: var(--danger); }
.source-pill--loading { border-color: #cbd8ef; background: var(--accent-soft); color: var(--primary-focus); }
.source-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.source-pill--loading .source-dot { animation: pulse 1.2s ease-in-out infinite; }
.tool-button { display: inline-flex; min-height: 34px; align-items: center; justify-content: center; gap: 7px; border: 1px solid var(--border-light); border-radius: 6px; background: var(--surface); color: var(--text); cursor: pointer; font: 650 12px/1.2 var(--font); padding: 0 12px; white-space: nowrap; }
.tool-button:hover:not(:disabled) { border-color: var(--text); background: var(--surface-2); }
.tool-button:disabled { cursor: not-allowed; opacity: .45; }
.tool-button--primary { border-color: var(--primary); background: var(--primary); color: #fff; }
.tool-button--primary:hover:not(:disabled) { border-color: var(--primary-focus); background: var(--primary-focus); }
.source-strip { display: flex; min-height: 34px; align-items: center; flex-wrap: wrap; gap: 8px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 7px 10px; color: var(--text-muted); font-size: 11px; }
.source-strip code, .tool-footer code, .contract-list code, .result-row__metrics code { font-family: var(--font-mono); }
.source-strip__detail { margin-left: auto; }
.tool-notice { display: flex; flex-direction: column; gap: 3px; border: 1px solid var(--border); border-left: 3px solid var(--danger); border-radius: 6px; background: var(--danger-soft); padding: 10px 12px; color: var(--danger); line-height: 1.45; }
.tool-notice--warning { border-left-color: var(--warning); background: var(--warning-soft); color: var(--warning); }
.metric-strip { display: grid; grid-template-columns: repeat(3, minmax(100px, 1fr)) minmax(220px, 1.5fr); overflow: hidden; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); }
.metric-tile { min-height: 66px; border-right: 1px solid var(--border); padding: 12px 14px; }
.metric-tile:last-child { border-right: 0; }
.metric-tile__value { display: block; margin-bottom: 3px; color: var(--text); font: 700 21px/1 var(--font-mono); }
.metric-tile__label { display: block; color: var(--text-muted); font-size: 9px; }
.metric-tile__caption { display: block; margin-top: 4px; color: var(--text-muted); font-size: 10px; }
.metric-tile--note { display: flex; flex-direction: column; justify-content: center; background: var(--surface-2); }
.suite-layout { display: grid; grid-template-columns: minmax(350px, 1.05fr) minmax(320px, .95fr); gap: 14px; min-height: 390px; }
.tool-panel { min-width: 0; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); padding: 14px; }
.panel-header { min-height: 34px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.panel-header h2 { margin: 0; color: var(--text); font-size: 14px; font-weight: 680; }
.count-label { color: var(--text-muted); font-size: 9px; white-space: nowrap; }
.select-all { display: inline-flex; align-items: center; gap: 6px; font-size: 9px; white-space: nowrap; }
.select-all input, .benchmark-option input { width: 16px; height: 16px; accent-color: var(--primary); }
.state-block { display: flex; min-height: 210px; flex-direction: column; align-items: center; justify-content: center; gap: 7px; padding: 22px; color: var(--text-muted); text-align: center; line-height: 1.5; }
.state-block--loading { min-height: 180px; }
.state-block strong { color: var(--text); font-size: 13px; }
.state-mark { display: grid; width: 32px; height: 32px; place-items: center; border: 1px solid var(--border-light); border-radius: 50%; color: var(--text-muted); font: 700 15px/1 var(--font-mono); }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border-light); border-top-color: var(--primary); border-radius: 50%; animation: spin .75s linear infinite; }
.benchmark-options { display: flex; flex-direction: column; gap: 6px; margin: 0; border: 0; padding: 0; }
.benchmark-option { display: flex; align-items: center; gap: 9px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); cursor: pointer; padding: 9px 10px; }
.benchmark-option:hover { border-color: var(--border-light); }
.benchmark-option--selected { border-color: var(--primary); background: var(--accent-soft); }
.benchmark-option > span { display: flex; min-width: 0; flex-direction: column; gap: 3px; }
.benchmark-option strong { font-size: 12px; }
.benchmark-option small { color: var(--text-muted); font-size: 9px; }
.selection-actions { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.contract-list { display: flex; flex-direction: column; gap: 12px; margin: 0; padding: 0; list-style: none; }
.contract-list li { display: grid; grid-template-columns: 30px minmax(0, 1fr); gap: 9px; border-bottom: 1px solid var(--border); padding: 0 0 12px; }
.contract-list li:last-child { border-bottom: 0; }
.contract-list li > span { color: var(--text-muted); font: 10px/1.4 var(--font-mono); }
.contract-list strong { font-size: 12px; }
.contract-list p { margin: 3px 0 0; color: var(--text-muted); font-size: 10px; line-height: 1.45; }
.contract-note { display: flex; flex-direction: column; gap: 3px; margin-top: 14px; border-left: 3px solid var(--warning); background: var(--warning-soft); padding: 10px; color: var(--warning); font-size: 10px; line-height: 1.45; }
.contract-note strong { color: var(--text); font-size: 11px; }
.results-panel { display: flex; flex-direction: column; gap: 2px; }
.results-list { display: flex; flex-direction: column; gap: 7px; }
.result-row { border: 1px solid var(--border); border-left: 3px solid var(--primary); border-radius: 6px; background: var(--surface-2); padding: 9px 10px; }
.result-row--success { border-left-color: var(--success); }
.result-row--error { border-left-color: var(--danger); }
.result-row__header strong { font-size: 12px; }
.result-row__header span { font-size: 9px; }
.result-row__metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; margin: 8px 0 0; }
.result-row__metrics > div { border: 1px solid var(--border); border-radius: 5px; background: var(--surface); padding: 6px 7px; }
.result-row__metrics dt { color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); text-transform: uppercase; }
.result-row__metrics dd { margin: 3px 0 0; color: var(--text); font: 10px/1.3 var(--font-mono); }
.result-row__error { margin: 7px 0 0; color: var(--danger); font-size: 10px; line-height: 1.4; overflow-wrap: anywhere; }
.inline-state { color: var(--text-muted); font-size: 11px; line-height: 1.5; padding: 10px 0; }
.tool-footer { justify-content: flex-start; flex-wrap: wrap; color: var(--text-muted); font-size: 10px; }
.tool-footer a { color: var(--text-dim); font-weight: 650; text-decoration: none; }
.tool-footer a:hover { text-decoration: underline; text-underline-offset: 3px; }
.desktop-sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; margin: -1px; padding: 0; border: 0; clip: rect(0, 0, 0, 0); white-space: nowrap; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 50% { opacity: .35; } }
@media (max-width: 850px) { .suite-layout { grid-template-columns: 1fr; } }
@media (max-width: 620px) { .suite-tool { padding: 14px; } .tool-header { flex-direction: column; } .tool-header__actions { width: 100%; justify-content: flex-start; } .source-strip__detail { width: 100%; margin-left: 0; } .metric-strip { grid-template-columns: repeat(3, 1fr); } .metric-tile--note { grid-column: 1 / -1; border-top: 1px solid var(--border); } .selection-actions { align-items: stretch; flex-direction: column; } .selection-actions .tool-button { width: 100%; } .result-row__metrics { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { .spinner, .source-pill--loading .source-dot { animation: none; } }
</style>
