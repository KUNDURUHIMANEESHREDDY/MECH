<template>
  <section class="benchmark-tool" aria-labelledby="benchmark-dashboard-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">Develop / Evaluation</p>
        <h1 id="benchmark-dashboard-title">Benchmark dashboard</h1>
        <p class="tool-intro">
          Run a benchmark returned by the backend catalog and inspect the response contract. Dashboard counters are
          session observations only; no historical performance is implied when the API has no result history.
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
      <span class="source-strip__label">Contracts</span>
      <code>GET /api/benchmarks</code>
      <code>POST /api/benchmarks/run</code>
      <span class="source-strip__detail">Results are retained only in this open window.</span>
    </div>

    <div v-if="errorMessage" class="tool-notice" :class="`tool-notice--${offline ? 'warning' : 'error'}`" role="alert">
      <strong>{{ offline ? 'Benchmark service offline' : 'Benchmark request failed' }}</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <section class="metric-strip" aria-label="Benchmark session observations">
      <div class="metric-tile">
        <span class="metric-tile__value">{{ catalog.length }}</span>
        <span class="metric-tile__label">Catalog benchmarks</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ completedSessionRuns }}</span>
        <span class="metric-tile__label">Completed this session</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ failedSessionRuns }}</span>
        <span class="metric-tile__label">Failed this session</span>
      </div>
      <div class="metric-tile metric-tile--note">
        <span class="metric-tile__label">History boundary</span>
        <span class="metric-tile__caption">The current API does not expose a persistent benchmark result list.</span>
      </div>
    </section>

    <div class="benchmark-grid">
      <section class="tool-panel catalog-panel" aria-labelledby="benchmark-catalog-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Backend catalog</p>
            <h2 id="benchmark-catalog-title">Available benchmarks</h2>
          </div>
          <span class="count-label">{{ catalog.length }} returned</span>
        </header>
        <div v-if="loadingCatalog" class="state-block state-block--loading" role="status" aria-live="polite">
          <span class="spinner" aria-hidden="true" />
          <span>Reading the benchmark catalog…</span>
        </div>
        <div v-else-if="offline" class="state-block" role="alert">
          <span class="state-mark" aria-hidden="true">!</span>
          <strong>Benchmark catalog offline</strong>
          <span>Start the runtime to load available benchmark names.</span>
        </div>
        <div v-else-if="!catalog.length" class="state-block">
          <span class="state-mark" aria-hidden="true">∅</span>
          <strong>No benchmark catalog returned</strong>
          <span>The endpoint responded without benchmark names.</span>
        </div>
        <ul v-else class="catalog-list" aria-label="Available benchmarks">
          <li v-for="name in catalog" :key="name">
            <button class="catalog-row" :class="{ 'catalog-row--selected': name === selectedBenchmark }" type="button" :aria-pressed="name === selectedBenchmark" @click="selectedBenchmark = name">
              <span class="catalog-row__mark" aria-hidden="true">›</span>
              <span><strong>{{ name }}</strong><small>Backend catalog entry</small></span>
              <span class="catalog-row__arrow" aria-hidden="true">→</span>
            </button>
          </li>
        </ul>
      </section>

      <section class="tool-panel run-panel" aria-labelledby="benchmark-run-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Single-run contract</p>
            <h2 id="benchmark-run-title">Run benchmark</h2>
          </div>
          <span class="api-badge">POST /api/benchmarks/run</span>
        </header>
        <form class="run-form" @submit.prevent="runSelectedBenchmark">
          <div class="field">
            <label for="benchmark-select">Benchmark</label>
            <select id="benchmark-select" v-model="selectedBenchmark" :disabled="!catalog.length || running">
              <option value="" disabled>{{ catalog.length ? 'Select a benchmark' : 'Catalog unavailable' }}</option>
              <option v-for="name in catalog" :key="name" :value="name">{{ name }}</option>
            </select>
          </div>
          <button class="tool-button tool-button--primary" type="submit" :disabled="!selectedBenchmark || running || offline">
            <span v-if="running" class="spinner" aria-hidden="true" />
            <span v-else aria-hidden="true">▶</span>
            {{ running ? 'Running…' : 'Run selected benchmark' }}
          </button>
        </form>
        <p class="run-note">The endpoint accepts a benchmark name. It does not accept a model or sample configuration in the current contract.</p>
        <div v-if="lastResult" class="result-card" :class="`result-card--${resultTone}`" role="status" aria-live="polite">
          <div class="result-card__heading"><strong>Last backend response</strong><span>{{ lastResult.status || 'status unavailable' }}</span></div>
          <dl class="result-grid">
            <div><dt>Benchmark</dt><dd>{{ lastResult.benchmarkName || selectedBenchmark || 'Unavailable' }}</dd></div>
            <div><dt>Score</dt><dd>{{ formatValue(lastResult.score) }}</dd></div>
            <div><dt>Pass rate</dt><dd>{{ formatValue(lastResult.passRate) }}</dd></div>
            <div><dt>Provenance</dt><dd>{{ lastResult.provenance || 'Not supplied' }}</dd></div>
          </dl>
          <p v-if="lastResult.error || lastResult.reason" class="result-error">{{ lastResult.error || lastResult.reason }}</p>
        </div>
        <div v-else class="result-empty">
          <span class="state-mark" aria-hidden="true">—</span>
          <strong>No result in this session</strong>
          <span>Run a catalog benchmark to see the backend response here.</span>
        </div>
      </section>
    </div>

    <section class="tool-panel activity-panel" aria-labelledby="session-activity-title">
      <header class="panel-header">
        <div>
          <p class="section-kicker">Not persisted by this view</p>
          <h2 id="session-activity-title">Session activity</h2>
        </div>
        <span class="count-label">{{ sessionRuns.length }} requests</span>
      </header>
      <div v-if="sessionRuns.length" class="activity-list">
        <div v-for="item in sessionRuns" :key="item.key" class="activity-row">
          <span class="activity-row__status" :class="`activity-row__status--${item.tone}`" aria-hidden="true">{{ item.tone === 'success' ? '✓' : item.tone === 'error' ? '!' : '…' }}</span>
          <span class="activity-row__body"><strong>{{ item.name }}</strong><span>{{ item.detail }}</span></span>
          <time>{{ item.time }}</time>
        </div>
      </div>
      <div v-else class="inline-state">No benchmark requests have been made in this window.</div>
    </section>

    <footer class="tool-footer">
      <span>Interpretation</span>
      <span>Scores and pass rates appear only after a backend response; absent values remain “Unavailable”.</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { api } from '../services/api';

type SourceTone = 'online' | 'offline' | 'loading' | 'partial';
type ResultTone = 'success' | 'error' | 'neutral';
type JsonRecord = Record<string, unknown>;
type SessionRun = { key: string; name: string; detail: string; time: string; tone: ResultTone };
type NormalizedResult = { status: string; benchmarkName: string; score: unknown; passRate: unknown; provenance: string; error: string; reason: string };

const catalog = ref<string[]>([]);
const selectedBenchmark = ref('');
const lastResult = ref<NormalizedResult | null>(null);
const sessionRuns = ref<SessionRun[]>([]);
const loadingCatalog = ref(true);
const running = ref(false);
const offline = ref(false);
const errorMessage = ref('');

function isRecord(value: unknown): value is JsonRecord { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function text(value: unknown, fallback = ''): string { return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? String(value) : fallback; }
function messageOf(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function isOffline(message: string): boolean { return /offline|failed to fetch|network|load|connection|timeout/i.test(message); }
function formatValue(value: unknown): string { return value === undefined || value === null || value === '' ? 'Unavailable' : typeof value === 'number' ? String(value) : text(value); }
function nowLabel(): string { return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }); }

function normalizeResult(value: unknown, benchmark: string): NormalizedResult {
  const record = isRecord(value) ? value : {};
  const status = text(record.status, 'response received');
  const reason = text(record.reason);
  return {
    status,
    benchmarkName: text(record.benchmark_name, benchmark),
    score: record.score,
    passRate: record.pass_rate,
    provenance: text(record.provenance, 'unavailable'),
    error: text(record.error),
    reason,
  };
}
function resultToneFor(status: string, error: string): ResultTone {
  const normalized = status.toLowerCase();
  if (['completed', 'complete', 'ok', 'success', 'passed'].includes(normalized)) return 'success';
  if (['error', 'failed', 'failure'].includes(normalized) || error) return 'error';
  return 'neutral';
}

const completedSessionRuns = computed(() => sessionRuns.value.filter((item) => item.tone === 'success').length);
const failedSessionRuns = computed(() => sessionRuns.value.filter((item) => item.tone === 'error').length);
const resultTone = computed<ResultTone>(() => resultToneFor(lastResult.value?.status || '', lastResult.value?.error || ''));
const sourceTone = computed<SourceTone>(() => loadingCatalog.value ? 'loading' : offline.value ? 'offline' : errorMessage.value ? 'partial' : 'online');
const sourceLabel = computed(() => sourceTone.value === 'loading' ? 'Loading catalog' : sourceTone.value === 'offline' ? 'Offline / unavailable' : sourceTone.value === 'partial' ? 'Partially available' : 'Benchmark API reachable');

async function loadCatalog(): Promise<void> {
  loadingCatalog.value = true;
  offline.value = false;
  errorMessage.value = '';
  try {
    catalog.value = await api.listBenchmarks();
    if (!catalog.value.includes(selectedBenchmark.value)) selectedBenchmark.value = catalog.value[0] ?? '';
  } catch (error) {
    catalog.value = [];
    errorMessage.value = messageOf(error);
    offline.value = isOffline(errorMessage.value);
  } finally {
    loadingCatalog.value = false;
  }
}

async function runSelectedBenchmark(): Promise<void> {
  const name = selectedBenchmark.value;
  if (!name || running.value) return;
  running.value = true;
  errorMessage.value = '';
  try {
    const result = normalizeResult(await api.runBenchmark(name), name);
    lastResult.value = result;
    const tone = resultToneFor(result.status, result.error);
    sessionRuns.value.unshift({ key: `${name}-${Date.now()}`, name, detail: result.error || result.reason || `${result.status} · score ${formatValue(result.score)}`, time: nowLabel(), tone });
  } catch (error) {
    errorMessage.value = messageOf(error);
    offline.value = isOffline(errorMessage.value);
    sessionRuns.value.unshift({ key: `${name}-${Date.now()}`, name, detail: errorMessage.value, time: nowLabel(), tone: 'error' });
  } finally {
    running.value = false;
  }
}

onMounted(() => { void loadCatalog(); });
</script>

<style scoped>
.benchmark-tool { display: flex; width: 100%; min-height: 100%; flex-direction: column; gap: 14px; padding: 22px; color: var(--text); font-size: 13px; }
.tool-header, .tool-header__actions, .panel-header, .tool-footer, .result-card__heading, .activity-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.tool-header { align-items: flex-start; }
.tool-header__copy { min-width: 0; }
.tool-eyebrow, .section-kicker, .source-strip__label, .metric-tile__label, .count-label, .api-badge, .tool-footer > span:first-child { color: var(--text-muted); font: 700 10px/1.2 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; }
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
.source-strip code, .tool-footer code, .result-grid code, .activity-row code { font-family: var(--font-mono); }
.source-strip__detail { margin-left: auto; }
.tool-notice { display: flex; flex-direction: column; gap: 3px; border: 1px solid var(--border); border-left: 3px solid var(--danger); border-radius: 6px; background: var(--danger-soft); padding: 10px 12px; color: var(--danger); line-height: 1.45; }
.tool-notice--warning { border-left-color: var(--warning); background: var(--warning-soft); color: var(--warning); }
.metric-strip { display: grid; grid-template-columns: repeat(3, minmax(100px, 1fr)) minmax(210px, 1.5fr); overflow: hidden; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); }
.metric-tile { min-height: 66px; border-right: 1px solid var(--border); padding: 12px 14px; }
.metric-tile:last-child { border-right: 0; }
.metric-tile__value { display: block; margin-bottom: 3px; color: var(--text); font: 700 21px/1 var(--font-mono); }
.metric-tile__label { display: block; color: var(--text-muted); font-size: 9px; }
.metric-tile__caption { display: block; margin-top: 4px; color: var(--text-muted); font-size: 10px; }
.metric-tile--note { display: flex; flex-direction: column; justify-content: center; background: var(--surface-2); }
.benchmark-grid { display: grid; grid-template-columns: minmax(280px, .8fr) minmax(420px, 1.2fr); gap: 14px; min-height: 420px; }
.tool-panel { min-width: 0; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); padding: 14px; }
.panel-header { min-height: 34px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.panel-header h2 { margin: 0; color: var(--text); font-size: 14px; font-weight: 680; }
.count-label { color: var(--text-muted); font-size: 9px; white-space: nowrap; }
.state-block, .result-empty { display: flex; min-height: 220px; flex-direction: column; align-items: center; justify-content: center; gap: 7px; padding: 22px; color: var(--text-muted); text-align: center; line-height: 1.5; }
.state-block--loading, .result-empty { min-height: 180px; }
.state-block strong, .result-empty strong { color: var(--text); font-size: 13px; }
.state-mark { display: grid; width: 32px; height: 32px; place-items: center; border: 1px solid var(--border-light); border-radius: 50%; color: var(--text-muted); font: 700 15px/1 var(--font-mono); }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border-light); border-top-color: var(--primary); border-radius: 50%; animation: spin .75s linear infinite; }
.catalog-list { display: flex; flex-direction: column; gap: 5px; margin: 0; padding: 0; list-style: none; }
.catalog-row { display: grid; width: 100%; grid-template-columns: 22px minmax(0, 1fr) 18px; align-items: center; gap: 8px; border: 1px solid transparent; border-radius: 6px; background: transparent; color: var(--text); cursor: pointer; padding: 9px 8px; text-align: left; }
.catalog-row:hover { border-color: var(--border); background: var(--surface-2); }
.catalog-row--selected { border-color: var(--primary); background: var(--accent-soft); }
.catalog-row__mark { color: var(--text-muted); font: 18px/1 var(--font-mono); }
.catalog-row strong { display: block; font-size: 12px; }
.catalog-row small { display: block; margin-top: 3px; color: var(--text-muted); font-size: 9px; }
.catalog-row__arrow { color: var(--text-muted); }
.run-form { display: flex; align-items: end; gap: 10px; }
.field { display: flex; flex: 1; flex-direction: column; gap: 5px; }
.field label { color: var(--text-dim); font: 700 10px/1.2 var(--font-mono); letter-spacing: .06em; text-transform: uppercase; }
.field select { width: 100%; min-height: 36px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--text); padding: 0 9px; font: inherit; font-size: 11px; }
.field select:focus { border-color: var(--primary); box-shadow: 0 0 0 2px var(--accent-soft); outline: none; }
.run-note { margin: 8px 0 13px; color: var(--text-muted); font-size: 10px; line-height: 1.5; }
.result-card { display: flex; flex-direction: column; gap: 10px; border: 1px solid var(--border); border-left: 3px solid var(--primary); border-radius: 6px; background: var(--surface-2); padding: 11px; }
.result-card--success { border-left-color: var(--success); }
.result-card--error { border-left-color: var(--danger); }
.result-card__heading strong { font-size: 12px; }
.result-card__heading span { color: var(--text-muted); font: 9px/1.3 var(--font-mono); text-transform: uppercase; }
.result-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; margin: 0; }
.result-grid > div { border: 1px solid var(--border); border-radius: 5px; background: var(--surface); padding: 7px 8px; }
.result-grid dt { color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); letter-spacing: .05em; text-transform: uppercase; }
.result-grid dd { margin: 3px 0 0; color: var(--text); font: 11px/1.3 var(--font-mono); }
.result-error { margin: 0; color: var(--danger); font-size: 10px; line-height: 1.45; overflow-wrap: anywhere; }
.activity-panel { display: flex; flex-direction: column; gap: 2px; }
.activity-list { display: flex; flex-direction: column; }
.activity-row { display: grid; grid-template-columns: 22px minmax(0, 1fr) auto; align-items: center; gap: 8px; border-bottom: 1px solid var(--border); padding: 8px 0; }
.activity-row:last-child { border-bottom: 0; }
.activity-row__status { display: grid; width: 20px; height: 20px; place-items: center; border: 1px solid currentColor; border-radius: 50%; font: 700 9px/1 var(--font-mono); }
.activity-row__status--success { color: var(--success); }
.activity-row__status--error { color: var(--danger); }
.activity-row__status--neutral { color: var(--text-muted); }
.activity-row__body { display: flex; min-width: 0; flex-direction: column; gap: 2px; }
.activity-row__body strong { font-size: 11px; }
.activity-row__body span { color: var(--text-muted); font-size: 10px; }
.activity-row time { color: var(--text-muted); font: 9px/1.3 var(--font-mono); white-space: nowrap; }
.inline-state { color: var(--text-muted); font-size: 11px; line-height: 1.5; padding: 10px 0; }
.tool-footer { justify-content: flex-start; flex-wrap: wrap; color: var(--text-muted); font-size: 10px; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 50% { opacity: .35; } }
@media (max-width: 900px) { .benchmark-grid { grid-template-columns: 1fr; } .run-panel { min-height: 340px; } }
@media (max-width: 620px) { .benchmark-tool { padding: 14px; } .tool-header { flex-direction: column; } .tool-header__actions { width: 100%; justify-content: flex-start; } .source-strip__detail { width: 100%; margin-left: 0; } .metric-strip { grid-template-columns: repeat(3, 1fr); } .metric-tile--note { grid-column: 1 / -1; border-top: 1px solid var(--border); } .run-form { align-items: stretch; flex-direction: column; } .run-form .tool-button { width: 100%; } .result-grid { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { .spinner, .source-pill--loading .source-dot { animation: none; } }
</style>
