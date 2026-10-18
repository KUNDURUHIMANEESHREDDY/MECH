<template>
  <section class="benchmark-tool" aria-labelledby="benchmark-dashboard-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">Develop / Evaluation</p>
        <h1 id="benchmark-dashboard-title">Benchmark dashboard</h1>
        <p class="tool-intro">
          Run a benchmark returned by the backend catalog and inspect the response contract. Completed live runs are
          recorded, and this view charts their recorded score over time. The session counters cover requests made in this
          window; the persisted history panel reads stored runs.
        </p>
      </div>
      <div class="tool-header__actions">
        <span class="source-pill" :class="`source-pill--${sourceTone}`" role="status">
          <span class="source-dot" aria-hidden="true" />
          {{ sourceLabel }}
        </span>
        <button class="tool-button" type="button" :disabled="loadingCatalog" @click="loadCatalog">
          <span aria-hidden="true">&#8635;</span>
          Refresh catalog
        </button>
      </div>
    </header>

    <div class="source-strip" role="status" aria-live="polite" data-testid="benchmark-contracts">
      <span class="source-strip__label">Contracts</span>
      <code>GET /api/benchmarks</code>
      <code>POST /api/benchmarks/run</code>
      <code>GET /api/benchmarks/history</code>
      <code>GET /api/benchmarks/compare</code>
      <span class="source-strip__detail">Completed live runs are persisted and read back from storage.</span>
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
      <div class="metric-tile">
        <span class="metric-tile__value" data-testid="persisted-run-count">{{ historyCount }}</span>
        <span class="metric-tile__label">Persisted runs</span>
      </div>
      <div class="metric-tile metric-tile--note">
        <span class="metric-tile__label">Storage</span>
        <span class="metric-tile__caption">{{ historyStateCaption }}</span>
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
          <span>Reading the benchmark catalog&#8230;</span>
        </div>
        <div v-else-if="offline" class="state-block" role="alert">
          <span class="state-mark" aria-hidden="true">!</span>
          <strong>Benchmark catalog offline</strong>
          <span>Start the runtime to load available benchmark names.</span>
        </div>
        <div v-else-if="!catalog.length" class="state-block">
          <span class="state-mark" aria-hidden="true">&#8709;</span>
          <strong>No benchmark catalog returned</strong>
          <span>The endpoint responded without benchmark names.</span>
        </div>
        <ul v-else class="catalog-list" aria-label="Available benchmarks">
          <li v-for="name in catalog" :key="name">
            <button class="catalog-row" :class="{ 'catalog-row--selected': name === selectedBenchmark }" type="button" :aria-pressed="name === selectedBenchmark" @click="selectedBenchmark = name">
              <span class="catalog-row__mark" aria-hidden="true">&#8250;</span>
              <span><strong>{{ name }}</strong><small>Backend catalog entry</small></span>
              <span class="catalog-row__arrow" aria-hidden="true">&#8594;</span>
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
            <select id="benchmark-select" v-model="selectedBenchmark" data-testid="benchmark-select" :disabled="!catalog.length || running">
              <option value="" disabled>{{ catalog.length ? 'Select a benchmark' : 'Catalog unavailable' }}</option>
              <option v-for="name in catalog" :key="name" :value="name">{{ name }}</option>
            </select>
          </div>
          <button class="tool-button tool-button--primary" type="submit" data-testid="benchmark-run" :disabled="!selectedBenchmark || running || offline">
            <span v-if="running" class="spinner" aria-hidden="true" />
            <span v-else aria-hidden="true">&#9654;</span>
            {{ running ? 'Running&#8230;' : 'Run selected benchmark' }}
          </button>
        </form>
        <p class="run-note">The endpoint accepts a benchmark name. It does not accept a model or sample configuration in the current contract. A completed live run is stored with its measured score and run id.</p>
        <div v-if="lastResult" class="result-card" :class="`result-card--${resultTone}`" role="status" aria-live="polite" data-testid="benchmark-result">
          <div class="result-card__heading"><strong>Last backend response</strong><span>{{ lastResult.status || 'status unavailable' }}</span></div>
          <dl class="result-grid">
            <div><dt>Benchmark</dt><dd>{{ lastResult.benchmarkName || selectedBenchmark || 'Unavailable' }}</dd></div>
            <div><dt>Score</dt><dd>{{ formatValue(lastResult.score) }}</dd></div>
            <div><dt>Pass rate</dt><dd>{{ formatValue(lastResult.passRate) }}</dd></div>
            <div><dt>Provenance</dt><dd>{{ lastResult.provenance || 'Unavailable' }}</dd></div>
            <div><dt>Field provenance</dt><dd>{{ formatFieldProvenance(lastResult.fieldProvenance) }}</dd></div>
            <div><dt>Persisted run id</dt><dd data-testid="benchmark-run-id">{{ lastResult.runId || 'Not recorded' }}</dd></div>
          </dl>
          <p v-if="lastResult.recordNote" class="result-error">{{ lastResult.recordNote }}</p>
          <p v-if="lastResult.error || lastResult.reason" class="result-error">{{ lastResult.error || lastResult.reason }}</p>
        </div>
        <div v-else class="result-empty">
          <span class="state-mark" aria-hidden="true">&#8212;</span>
          <strong>No result in this session</strong>
          <span>Run a catalog benchmark to see the backend response here.</span>
        </div>
      </section>
    </div>

    <section class="tool-panel history-panel" aria-labelledby="benchmark-history-title">
      <header class="panel-header">
        <div>
          <p class="section-kicker">Persisted runs</p>
          <h2 id="benchmark-history-title">Benchmark history</h2>
        </div>
        <div class="history-panel__actions">
          <label class="field field--inline" for="history-benchmark-filter">
            <span>Benchmark</span>
            <select id="history-benchmark-filter" v-model="historyFilter" data-testid="history-filter" :disabled="loadingHistory">
              <option value="">All benchmarks</option>
              <option v-for="name in catalog" :key="name" :value="name">{{ name }}</option>
            </select>
          </label>
          <button class="tool-button" type="button" data-testid="history-refresh" :disabled="loadingHistory" @click="loadHistory">
            <span aria-hidden="true">&#8635;</span>
            Refresh history
          </button>
        </div>
      </header>

      <div class="history-status" role="status" aria-live="polite" data-testid="history-status">{{ historyStatusText }}</div>

      <div v-if="loadingHistory" class="state-block state-block--loading" role="status" aria-live="polite">
        <span class="spinner" aria-hidden="true" />
        <span>Reading persisted run history&#8230;</span>
      </div>
      <div v-else-if="historyError" class="state-block" role="alert" data-testid="history-error">
        <span class="state-mark" aria-hidden="true">!</span>
        <strong>Persisted history unavailable</strong>
        <span>{{ historyError }}</span>
      </div>
      <div v-else-if="!historyRuns.length" class="state-block" data-testid="history-empty">
        <span class="state-mark" aria-hidden="true">&#8709;</span>
        <strong>No persisted history</strong>
        <span>{{ historyFilter ? `No stored run was recorded for ${historyFilter}.` : 'No completed live run has been recorded yet. A refused or unavailable run is never stored.' }}</span>
      </div>
      <template v-else>
        <div v-if="regression" class="regression-strip" :class="`regression-strip--${regression.tone}`" role="status" aria-live="polite" data-testid="regression-summary">
          <span class="regression-strip__delta">{{ regression.delta }}</span>
          <span class="regression-strip__body">
            <strong>{{ regression.label }}</strong>
            <span>{{ regression.detail }}</span>
          </span>
        </div>
        <div v-else class="inline-state" data-testid="regression-none">
          {{ regressionStatus || 'No two stored runs share a benchmark and a measured score, so no change is reported.' }}
        </div>

        <div v-if="regressionChart" class="chart-block">
          <p class="chart-caption">{{ regressedBenchmark }} &#183; score per stored run ({{ regressionChart.points.length }} plotted)</p>
          <svg class="regression-chart" :viewBox="`0 0 ${regressionChart.width} ${regressionChart.height}`" role="img" :aria-label="`${regressedBenchmark} score per stored run, oldest to newest`" data-testid="regression-chart">
            <line v-for="line in regressionChart.gridLines" :key="line.y" class="chart-grid" :x1="regressionChart.left" :x2="regressionChart.right" :y1="line.y" :y2="line.y" />
            <text v-for="line in regressionChart.gridLines" :key="`label-${line.y}`" class="chart-axis-text" :x="regressionChart.left - 7" :y="line.y + 4" text-anchor="end">{{ line.value }}</text>
            <line class="chart-axis" :x1="regressionChart.left" :x2="regressionChart.left" :y1="regressionChart.top" :y2="regressionChart.bottom" />
            <line class="chart-axis" :x1="regressionChart.left" :x2="regressionChart.right" :y1="regressionChart.bottom" :y2="regressionChart.bottom" />
            <polyline class="chart-line" :points="regressionChart.line" />
            <g v-for="point in regressionChart.points" :key="point.runId">
              <circle class="chart-point" :class="{ 'chart-point--latest': point.latest }" :cx="point.x" :cy="point.y" r="4" />
              <text class="chart-value" :x="point.x" :y="point.y - 9" text-anchor="middle">{{ point.value }}</text>
              <title>{{ point.title }}</title>
            </g>
            <text class="chart-axis-text" :x="regressionChart.left" :y="regressionChart.height - 4" text-anchor="start">oldest</text>
            <text class="chart-axis-text" :x="regressionChart.right" :y="regressionChart.height - 4" text-anchor="end">newest</text>
          </svg>
        </div>

        <div class="history-list" data-testid="history-runs">
          <div v-for="run in historyRuns" :key="run.runId" class="history-row">
            <span class="history-row__score">{{ formatValue(run.score) }}</span>
            <span class="history-row__body">
              <strong>{{ run.benchmark || 'benchmark unavailable' }}</strong>
              <span>
                {{ run.status || 'status unavailable' }} &#183; {{ run.provenance || 'provenance unavailable' }}
                <template v-if="run.runId"> &#183; <code>{{ run.runId }}</code></template>
                <template v-if="run.createdAt"> &#183; {{ run.createdAt }}</template>
              </span>
            </span>
          </div>
        </div>
      </template>
    </section>

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
          <span class="activity-row__status" :class="`activity-row__status--${item.tone}`" aria-hidden="true">{{ item.tone === 'success' ? '&#10003;' : item.tone === 'error' ? '!' : '&#8230;' }}</span>
          <span class="activity-row__body"><strong>{{ item.name }}</strong><span>{{ item.detail }}</span></span>
          <time>{{ item.time }}</time>
        </div>
      </div>
      <div v-else class="inline-state">No benchmark requests have been made in this window.</div>
    </section>

    <footer class="tool-footer">
      <span>Interpretation</span>
      <span>Scores and pass rates appear only after a backend response; a run with no measured score stays &#8220;Unavailable&#8221; and is never plotted as 0. Absent values remain &#8220;Unavailable&#8221;.</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { api } from '../services/api';

type SourceTone = 'online' | 'offline' | 'loading' | 'partial';
type ResultTone = 'success' | 'error' | 'neutral';
type JsonRecord = Record<string, unknown>;
type SessionRun = { key: string; name: string; detail: string; time: string; tone: ResultTone };
type NormalizedResult = { status: string; benchmarkName: string; score: unknown; passRate: unknown; provenance: string; fieldProvenance: Record<string, string>; error: string; reason: string; runId: string; recordNote: string };
type HistoryRun = { runId: string; benchmark: string; score: number | null; status: string; provenance: string; createdAt: string };
type ChartPoint = { runId: string; x: number; y: number; value: string; title: string; latest: boolean };
type ChartGridLine = { y: number; value: string };
type RegressionChart = { width: number; height: number; left: number; right: number; top: number; bottom: number; points: ChartPoint[]; line: string; gridLines: ChartGridLine[] };
type Regression = { delta: string; tone: 'up' | 'down' | 'flat'; label: string; detail: string };

const chartWidth = 720;
const chartHeight = 240;
const chartPadding = { left: 54, right: 18, top: 24, bottom: 30 };
const HISTORY_LIMIT = 50;

const catalog = ref<string[]>([]);
const selectedBenchmark = ref('');
const lastResult = ref<NormalizedResult | null>(null);
const sessionRuns = ref<SessionRun[]>([]);
const loadingCatalog = ref(true);
const running = ref(false);
const offline = ref(false);
const errorMessage = ref('');
const historyRuns = ref<HistoryRun[]>([]);
const loadingHistory = ref(true);
const historyError = ref('');
const historyFilter = ref('');
const historyFieldProvenance = ref<Record<string, string>>({});
const comparison = ref<Record<string, unknown> | null>(null);
const comparisonError = ref('');
const comparisonFailed = ref(false);
const comparisonLoading = ref(false);

function isRecord(value: unknown): value is JsonRecord { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function text(value: unknown, fallback = ''): string { return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? String(value) : fallback; }
function messageOf(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function isOffline(message: string): boolean { return /offline|failed to fetch|network|load|connection|timeout/i.test(message); }
function formatValue(value: unknown): string { return value === undefined || value === null || value === '' ? 'Unavailable' : typeof value === 'number' ? String(value) : text(value); }
function nowLabel(): string { return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }); }

/**
 * A stored score is plotted only when it is a real finite number. SQL NULL
 * arrives as null/undefined and is kept as "no measurement" - it is never
 * coerced to 0, because a fabricated zero would render as a real data point.
 */
function finiteScore(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}
function formatScore(value: number | null): string { return value === null ? 'Unavailable' : String(value); }
function formatDelta(value: number | null): string {
  if (value === null) return 'no measured change';
  return `${value > 0 ? '+' : ''}${value.toFixed(4)}`;
}

function historyRunOf(value: unknown): HistoryRun | null {
  if (!isRecord(value)) return null;
  const runId = text(value.run_id);
  if (!runId) return null;
  return {
    runId,
    benchmark: text(value.benchmark),
    score: finiteScore(value.primary_score),
    status: text(value.status),
    provenance: text(value.provenance),
    createdAt: text(value.created_at),
  };
}

function fieldProvenanceOf(value: unknown): Record<string, string> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, text(item, 'unavailable')]));
}

function formatFieldProvenance(value: Record<string, string>): string {
  const entries = Object.entries(value);
  return entries.length ? entries.map(([key, item]) => `${key}: ${item}`).join(' \u00b7 ') : 'Unavailable';
}

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
    fieldProvenance: fieldProvenanceOf(record.field_provenance),
    error: text(record.error),
    reason,
    runId: text(record.run_id),
    // A completed live run whose id is missing was not stored. Said out loud
    // rather than left as a blank cell that reads like a successful record.
    recordNote: status.toLowerCase() === 'completed' && !text(record.run_id)
      ? 'Not stored: the response carried no run id, so no history row was written.'
      : '',
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

const historyCount = computed(() => loadingHistory.value ? '\u2014' : String(historyRuns.value.length));
const historyStateCaption = computed(() => loadingHistory.value
  ? 'Reading persisted runs\u2026'
  : historyError.value
    ? 'Persisted history could not be read.'
    : historyRuns.value.length
      ? `${historyRuns.value.length} run(s) read from storage.`
      : 'No persisted run has been recorded yet.');
const historyStatusText = computed(() => loadingHistory.value
  ? 'Reading persisted benchmark runs\u2026'
  : historyError.value
    ? `Persisted history unavailable: ${historyError.value}`
    : historyRuns.value.length
      ? `${historyRuns.value.length} persisted run(s) read${historyFilter.value ? ` for ${historyFilter.value}` : ''}.`
      : 'No persisted history: no completed live run has been recorded.');

/** Stored runs that carry a real score, one benchmark at a time, oldest first. */
function scoreSeries(): HistoryRun[] {
  return historyRuns.value
    .filter((run) => run.benchmark && run.score !== null)
    .slice()
    .reverse();
}

/** The benchmark whose stored runs can actually be compared. */
function comparableBenchmark(): string {
  const byBenchmark = new Map<string, HistoryRun[]>();
  for (const run of scoreSeries()) {
    if (!run.benchmark) continue;
    const list = byBenchmark.get(run.benchmark) ?? [];
    list.push(run);
    byBenchmark.set(run.benchmark, list);
  }
  let best = '';
  let bestCount = 0;
  for (const [benchmark, runs] of byBenchmark) {
    if (runs.length > bestCount) {
      bestCount = runs.length;
      best = benchmark;
    }
  }
  return best;
}

const regressedBenchmark = computed(() => comparableBenchmark());

const regressionChart = computed<RegressionChart | null>(() => {
  const benchmark = regressedBenchmark.value;
  if (!benchmark) return null;
  const runs = scoreSeries().filter((run) => run.benchmark === benchmark);
  if (!runs.length) return null;

  const values = runs.map((run) => run.score as number);
  const left = chartPadding.left;
  const right = chartWidth - chartPadding.right;
  const top = chartPadding.top;
  const bottom = chartHeight - chartPadding.bottom;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min;
  const plotHeight = bottom - top;
  const xFor = (index: number): number => runs.length === 1
    ? (left + right) / 2
    : left + (index * (right - left)) / (runs.length - 1);
  const yFor = (score: number): number => span === 0
    ? (top + bottom) / 2
    : bottom - ((score - min) / span) * plotHeight;

  const points: ChartPoint[] = runs.map((run, index) => ({
    runId: run.runId,
    x: Number(xFor(index).toFixed(2)),
    y: Number(yFor(run.score as number).toFixed(2)),
    value: formatScore(run.score),
    title: `${run.benchmark} \u00b7 ${formatScore(run.score)} \u00b7 ${run.createdAt || run.runId} \u00b7 ${run.provenance || 'provenance unavailable'}`,
    latest: index === runs.length - 1,
  }));

  const gridLines: ChartGridLine[] = [
    { y: Number(top.toFixed(2)), value: String(Number(max.toFixed(4))) },
    { y: Number(((top + bottom) / 2).toFixed(2)), value: String(Number(((min + max) / 2).toFixed(4))) },
    { y: Number(bottom.toFixed(2)), value: String(Number(min.toFixed(4))) },
  ];
  return {
    width: chartWidth,
    height: chartHeight,
    left,
    right,
    top,
    bottom,
    points,
    line: points.map((point) => `${point.x},${point.y}`).join(' '),
    gridLines,
  };
});

const notComparable = computed(() => Array.isArray(comparison.value?.not_comparable)
  && comparison.value.not_comparable.length > 0);

/** The store's own explanation for a refused delta, never a substitute for one. */
function reasonOf(payload: Record<string, unknown> | null): string {
  if (!payload) return '';
  const explicit = text(payload.score_delta_reason);
  if (explicit) return explicit;
  const reasons = Array.isArray(payload.not_comparable) ? payload.not_comparable : [];
  return reasons.map((item) => text(item)).filter(Boolean).join(' ');
}

const regressionStatus = computed(() => {
  if (loadingHistory.value) return '';
  if (comparisonLoading.value) return 'Comparing the latest two stored runs\u2026';
  if (comparisonFailed.value) return `The store could not compare the latest two runs: ${comparisonError.value}`;
  if (regression.value) return '';
  if (notComparable.value) {
    return reasonOf(comparison.value)
      || 'The store reports these runs as not comparable, so no change is claimed.';
  }
  if (historyRuns.value.some((run) => run.score === null)) {
    return 'A stored run has no measured score, so no change is reported for it.';
  }
  return 'No two stored runs share a benchmark, so no change is reported.';
});

const regression = computed<Regression | null>(() => {
  const payload = comparison.value;
  if (!payload) return null;
  const delta = finiteScore(payload.score_delta);
  // "not comparable" on the delta is the backend's own verdict, carried
  // through verbatim: no delta is invented where the store refused one. The
  // reasons travel with it into the panel.
  if (notComparable.value || delta === null) return null;
  const runA = isRecord(payload.run_a) ? payload.run_a : null;
  const runB = isRecord(payload.run_b) ? payload.run_b : null;
  const scoreA = runA ? finiteScore(runA.primary_score) : null;
  const scoreB = runB ? finiteScore(runB.primary_score) : null;
  const metric = text(payload.primary_metric) || 'primary metric';
  const reason = reasonOf(payload);
  return {
    delta: formatDelta(delta),
    tone: delta > 0 ? 'up' : delta < 0 ? 'down' : 'flat',
    label: `${regressedBenchmark.value || 'Benchmark'} \u00b7 ${metric} change ${formatDelta(delta)}`,
    detail: `from ${formatScore(scoreA)} to ${formatScore(scoreB)} across the latest two stored runs.${reason ? ` ${reason}` : ''}`,
  };
});

watch(regressionChart, (chart) => {
  // Capture the refusal reason BEFORE clearing the payload it came from, so
  // "not comparable" stays visible instead of collapsing to a blank panel.
  comparisonError.value = reasonOf(comparison.value);
  comparisonFailed.value = false;
  comparison.value = null;
  const points = chart?.points ?? [];
  if (points.length < 2) return;
  void compareLatest(points[points.length - 2].runId, points[points.length - 1].runId);
}, { immediate: true });

async function compareLatest(runA: string, runB: string): Promise<void> {
  comparisonLoading.value = true;
  try {
    comparison.value = await api.compareBenchmarkRuns(runA, runB);
  } catch (error) {
    comparison.value = null;
    comparisonFailed.value = true;
    comparisonError.value = messageOf(error);
  } finally {
    comparisonLoading.value = false;
  }
}

async function loadHistory(): Promise<void> {
  loadingHistory.value = true;
  historyError.value = '';
  try {
    const payload = await api.benchmarkHistory(historyFilter.value || undefined, HISTORY_LIMIT);
    const runs = Array.isArray(payload.runs) ? payload.runs : [];
    historyRuns.value = runs
      .map(historyRunOf)
      .filter((run): run is HistoryRun => run !== null);
    historyFieldProvenance.value = fieldProvenanceOf(payload.field_provenance);
  } catch (error) {
    historyRuns.value = [];
    historyFieldProvenance.value = {};
    historyError.value = messageOf(error);
  } finally {
    loadingHistory.value = false;
  }
}

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
  await loadHistory();
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
    sessionRuns.value.unshift({ key: `${name}-${Date.now()}`, name, detail: result.error || result.reason || `${result.status} \u00b7 score ${formatValue(result.score)}`, time: nowLabel(), tone });
  } catch (error) {
    errorMessage.value = messageOf(error);
    offline.value = isOffline(errorMessage.value);
    sessionRuns.value.unshift({ key: `${name}-${Date.now()}`, name, detail: errorMessage.value, time: nowLabel(), tone: 'error' });
  } finally {
    running.value = false;
  }
  // Read the store back after the run: the panel shows what was persisted,
  // not what the response claimed.
  await loadHistory();
}

onMounted(() => { void loadCatalog(); });
watch(historyFilter, () => { void loadHistory(); });
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
.source-strip code, .tool-footer code, .result-grid code, .activity-row code, .history-row code { font-family: var(--font-mono); }
.source-strip__detail { margin-left: auto; }
.tool-notice { display: flex; flex-direction: column; gap: 3px; border: 1px solid var(--border); border-left: 3px solid var(--danger); border-radius: 6px; background: var(--danger-soft); padding: 10px 12px; color: var(--danger); line-height: 1.45; }
.tool-notice--warning { border-left-color: var(--warning); background: var(--warning-soft); color: var(--warning); }
.metric-strip { display: grid; grid-template-columns: repeat(3, minmax(100px, 1fr)) minmax(120px, .8fr) minmax(200px, 1.4fr); overflow: hidden; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); }
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
.result-grid dd { margin: 3px 0 0; color: var(--text); font: 11px/1.3 var(--font-mono); overflow-wrap: anywhere; }
.result-error { margin: 0; color: var(--danger); font-size: 10px; line-height: 1.45; overflow-wrap: anywhere; }
.history-panel { display: flex; flex-direction: column; gap: 10px; }
.history-panel__actions { display: flex; align-items: end; gap: 8px; }
.field--inline { flex: 0 0 auto; min-width: 150px; }
.history-status { color: var(--text-muted); font-size: 10px; line-height: 1.45; }
.regression-strip { display: flex; align-items: center; gap: 11px; border: 1px solid var(--border); border-left: 3px solid var(--text-muted); border-radius: 6px; background: var(--surface-2); padding: 9px 11px; }
.regression-strip--up { border-left-color: var(--success); }
.regression-strip--down { border-left-color: var(--danger); }
.regression-strip__delta { color: var(--text); font: 700 18px/1 var(--font-mono); }
.regression-strip__body { display: flex; min-width: 0; flex-direction: column; gap: 3px; }
.regression-strip__body strong { font-size: 11px; }
.regression-strip__body span { color: var(--text-muted); font-size: 10px; line-height: 1.45; }
.chart-block { border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 9px 11px 5px; }
.chart-caption { margin: 0 0 5px; color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); letter-spacing: .05em; text-transform: uppercase; }
.regression-chart { display: block; width: 100%; height: auto; }
.chart-grid { stroke: var(--border); stroke-width: 1; stroke-dasharray: 3 4; }
.chart-axis { stroke: var(--border-light); stroke-width: 1; }
.chart-axis-text { fill: var(--text-muted); font: 9px var(--font-mono); }
.chart-line { fill: none; stroke: var(--primary); stroke-width: 2; stroke-linejoin: round; }
.chart-point { fill: var(--primary); }
.chart-point--latest { fill: var(--accent); stroke: var(--primary); stroke-width: 2; }
.chart-value { fill: var(--text); font: 9px var(--font-mono); }
.history-list { display: flex; flex-direction: column; }
.history-row { display: grid; grid-template-columns: 86px minmax(0, 1fr); align-items: center; gap: 10px; border-bottom: 1px solid var(--border); padding: 7px 0; }
.history-row:last-child { border-bottom: 0; }
.history-row__score { color: var(--text); font: 700 12px/1.2 var(--font-mono); }
.history-row__body { display: flex; min-width: 0; flex-direction: column; gap: 2px; }
.history-row__body strong { font-size: 11px; }
.history-row__body span { color: var(--text-muted); font-size: 10px; line-height: 1.4; overflow-wrap: anywhere; }
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
@media (max-width: 900px) { .benchmark-grid { grid-template-columns: 1fr; } .run-panel { min-height: 340px; } .panel-header { flex-wrap: wrap; } .history-panel__actions { width: 100%; flex-wrap: wrap; } }
@media (max-width: 620px) { .benchmark-tool { padding: 14px; } .tool-header { flex-direction: column; } .tool-header__actions { width: 100%; justify-content: flex-start; } .source-strip__detail { width: 100%; margin-left: 0; } .metric-strip { grid-template-columns: repeat(3, 1fr); } .metric-tile--note { grid-column: 1 / -1; border-top: 1px solid var(--border); } .run-form { align-items: stretch; flex-direction: column; } .run-form .tool-button { width: 100%; } .result-grid { grid-template-columns: 1fr; } .history-row { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { .spinner, .source-pill--loading .source-dot { animation: none; } }
</style>