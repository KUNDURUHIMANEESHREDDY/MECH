<template>
  <section class="reports-tool" aria-labelledby="reports-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">Research / Deliverables</p>
        <h1 id="reports-title">Reports</h1>
        <p class="tool-intro">
          Inspect report payloads returned by completed Society runs. This view does not manufacture report summaries
          or treat a catalog entry as a generated result.
        </p>
      </div>
      <div class="tool-header__actions">
        <span class="source-pill" :class="`source-pill--${sourceTone}`" role="status">
          <span class="source-dot" aria-hidden="true" />
          {{ sourceLabel }}
        </span>
        <button class="tool-button" type="button" :disabled="loading" @click="loadReports">
          <span aria-hidden="true">↻</span>
          Refresh reports
        </button>
      </div>
    </header>

    <div class="source-strip" role="status" aria-live="polite">
      <span class="source-strip__label">Sources</span>
      <code>Society run results</code>
      <code>GET /api/research_catalog?item_type=reports</code>
      <span class="source-strip__detail">{{ sourceDetail }}</span>
    </div>

    <div v-if="runError || catalogError" class="tool-notice" :class="`tool-notice--${runError ? 'error' : 'warning'}`" role="alert">
      <strong>{{ runError ? 'Run report source unavailable' : 'Reference catalog unavailable' }}</strong>
      <span>{{ runError || catalogError }}</span>
    </div>

    <section class="metric-strip" aria-label="Report source observations">
      <div class="metric-tile">
        <span class="metric-tile__value">{{ liveReports.length }}</span>
        <span class="metric-tile__label">Live report payloads</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ runCount }}</span>
        <span class="metric-tile__label">Runs inspected</span>
      </div>
      <div class="metric-tile">
        <span class="metric-tile__value">{{ catalogEntries.length }}</span>
        <span class="metric-tile__label">Reference entries</span>
      </div>
      <div class="metric-tile metric-tile--note">
        <span class="metric-tile__label">Provenance rule</span>
        <span class="metric-tile__caption">Only returned run payloads appear as reports. Catalog rows are labeled separately.</span>
      </div>
    </section>

    <div class="reports-grid">
      <section class="tool-panel report-list-panel" aria-labelledby="live-report-list-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Returned by backend</p>
            <h2 id="live-report-list-title">Live report payloads</h2>
          </div>
          <span class="count-label">{{ liveReports.length }} {{ liveReports.length === 1 ? 'report' : 'reports' }}</span>
        </header>

        <div v-if="loading" class="state-block state-block--loading" role="status" aria-live="polite">
          <span class="spinner" aria-hidden="true" />
          <span>Reading report payloads from the run ledger…</span>
        </div>
        <div v-else-if="offline" class="state-block" role="alert">
          <span class="state-mark" aria-hidden="true">!</span>
          <strong>Backend offline</strong>
          <span>Report payloads are unavailable until the MECH runtime is reachable.</span>
        </div>
        <div v-else-if="liveReports.length === 0" class="state-block">
          <span class="state-mark" aria-hidden="true">∅</span>
          <strong>No reports generated yet</strong>
          <span>The run ledger is reachable, but no completed run returned a report payload.</span>
        </div>
        <ul v-else class="report-list" aria-label="Live report payloads">
          <li v-for="report in liveReports" :key="report.id">
            <button
              class="report-row"
              :class="{ 'report-row--selected': report.id === selectedReportId }"
              type="button"
              :aria-pressed="report.id === selectedReportId"
              @click="selectedReportId = report.id"
            >
              <span class="report-row__type">RUN</span>
              <span class="report-row__body">
                <strong>{{ report.title }}</strong>
                <span>{{ report.goal || 'Goal not returned' }}</span>
                <span class="report-row__meta"><code>{{ report.runId }}</code> · {{ report.status || 'status unavailable' }}</span>
              </span>
              <time v-if="report.generatedAt" :datetime="report.generatedAt">{{ formatDate(report.generatedAt) }}</time>
            </button>
          </li>
        </ul>
      </section>

      <section class="tool-panel report-detail-panel" aria-labelledby="report-detail-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Read-only preview</p>
            <h2 id="report-detail-title">{{ selectedReport ? selectedReport.title : 'Report detail' }}</h2>
          </div>
          <button v-if="selectedReport" class="tool-button tool-button--small" type="button" @click="copySelectedReport">
            {{ copyLabel }}
          </button>
        </header>

        <div v-if="!selectedReport" class="state-block">
          <span class="state-mark" aria-hidden="true">↖</span>
          <strong>Select a report</strong>
          <span>Choose a returned report to inspect its Markdown payload and run provenance.</span>
        </div>
        <div v-else class="report-detail">
          <dl class="record-grid">
            <div>
              <dt>Source run</dt>
              <dd><code>{{ selectedReport.runId }}</code></dd>
            </div>
            <div>
              <dt>Run status</dt>
              <dd>{{ selectedReport.status || 'Unavailable' }}</dd>
            </div>
            <div>
              <dt>Generated</dt>
              <dd>{{ formatDate(selectedReport.generatedAt) }}</dd>
            </div>
            <div>
              <dt>Payload</dt>
              <dd>Backend-returned Markdown</dd>
            </div>
          </dl>
          <div class="provenance-note">
            <strong>Provenance</strong>
            <span>This text was returned inside the selected Society run record. It is not regenerated or scored by this view.</span>
          </div>
          <pre class="report-markdown">{{ selectedReport.markdown }}</pre>
        </div>
      </section>
    </div>

    <section class="tool-panel catalog-panel" aria-labelledby="reference-catalog-title">
      <header class="panel-header">
        <div>
          <p class="section-kicker">Illustrative index only</p>
          <h2 id="reference-catalog-title">Reference catalog</h2>
        </div>
        <span class="reference-badge">Reference catalog</span>
      </header>
      <p class="catalog-help">
        The backend exposes a research catalog index, but not a persisted report listing endpoint. These entries are
        navigation references only; select the live payload list above for actual generated output.
      </p>
      <div v-if="loading" class="inline-state">Loading catalog…</div>
      <div v-else-if="catalogEntries.length" class="catalog-list">
        <div v-for="entry in catalogEntries" :key="entry.id" class="catalog-row">
          <span class="catalog-row__id"><code>{{ entry.id }}</code></span>
          <span>{{ entry.title || 'Untitled reference entry' }}</span>
          <span class="catalog-row__tag">Reference catalog</span>
        </div>
      </div>
      <div v-else class="inline-state">No reference entries returned. This does not imply that live reports exist or do not exist.</div>
    </section>

    <footer class="tool-footer">
      <span>Contract note</span>
      <span>No dedicated <code>GET /api/reports</code> route is exposed by the current runtime.</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { apiUrl } from '../services/api';

type SourceTone = 'online' | 'offline' | 'loading' | 'partial';

type JsonRecord = Record<string, unknown>;

interface ReportEntry {
  id: string;
  runId: string;
  title: string;
  goal: string;
  status: string;
  generatedAt: string;
  markdown: string;
}

interface CatalogEntry {
  id: string;
  title: string;
}

const API_BASE = apiUrl('/api');
const liveReports = ref<ReportEntry[]>([]);
const catalogEntries = ref<CatalogEntry[]>([]);
const runCount = ref(0);
const selectedReportId = ref('');
const loading = ref(true);
const offline = ref(false);
const runError = ref('');
const catalogError = ref('');
const copyLabel = ref('Copy Markdown');

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function text(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return fallback;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function looksOffline(message: string): boolean {
  return /offline|failed to fetch|network|load|connection|timeout/i.test(message);
}

async function request<T>(path: string): Promise<T> {
  if (typeof navigator !== 'undefined' && navigator.onLine === false) throw new Error('Browser is offline');
  const response = await fetch(`${API_BASE}${path}`, { headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error(`GET ${path} returned ${response.status}`);
  return response.json() as Promise<T>;
}

function reportPayload(value: unknown): { markdown: string; title: string; generatedAt: string } {
  if (typeof value === 'string') return { markdown: value, title: '', generatedAt: '' };
  if (!isRecord(value)) return { markdown: '', title: '', generatedAt: '' };
  return {
    markdown: text(value.markdown) || text(value.report),
    title: text(value.title) || text(value.report_title),
    generatedAt: text(value.generated_at) || text(value.published_at),
  };
}

function normalizeRun(value: unknown): { id: string; goal: string; status: string; created: string } | null {
  if (!isRecord(value)) return null;
  const id = text(value.run_id) || text(value.id);
  if (!id) return null;
  return { id, goal: text(value.goal), status: text(value.status), created: text(value.created) || text(value.published_at) };
}

function normalizeReport(run: { id: string; goal: string; status: string; created: string }, raw: unknown): ReportEntry | null {
  if (!isRecord(raw)) return null;
  const result = isRecord(raw.result) ? raw.result : raw;
  const publication = isRecord(result.publication) ? result.publication : isRecord(raw.publication) ? raw.publication : null;
  const payload = reportPayload(publication?.report ?? result.report ?? raw.report);
  if (!payload.markdown.trim()) return null;
  return {
    id: `${run.id}:report`,
    runId: run.id,
    title: payload.title || `Mechanistic report · ${run.goal || run.id}`,
    goal: text(result.goal, run.goal),
    status: text(raw.status, text(result.status, run.status)),
    generatedAt: payload.generatedAt || run.created,
    markdown: payload.markdown,
  };
}

function normalizeCatalog(value: unknown): CatalogEntry[] {
  const values = Array.isArray(value) ? value : isRecord(value) && Array.isArray(value.catalog) ? value.catalog : [];
  return values.flatMap((item) => {
    if (!isRecord(item)) return [];
    const id = text(item.id) || text(item.report_id);
    if (!id) return [];
    return [{ id, title: text(item.title) }];
  });
}

async function loadReports(): Promise<void> {
  loading.value = true;
  offline.value = false;
  runError.value = '';
  catalogError.value = '';
  const [runResult, catalogResult] = await Promise.allSettled([
    request<unknown>('/society/runs'),
    request<unknown>('/research_catalog?item_type=reports'),
  ]);

  const runs = runResult.status === 'fulfilled'
    ? (Array.isArray(runResult.value) ? runResult.value : isRecord(runResult.value) && Array.isArray(runResult.value.runs) ? runResult.value.runs : [])
    : [];
  const normalizedRuns = runs.map(normalizeRun).filter((run): run is { id: string; goal: string; status: string; created: string } => Boolean(run));
  runCount.value = normalizedRuns.length;

  if (runResult.status === 'rejected') {
    runError.value = `The run report source could not be read: ${errorMessage(runResult.reason)}`;
    if (looksOffline(errorMessage(runResult.reason))) offline.value = true;
  } else {
    const details = await Promise.allSettled(normalizedRuns.slice(0, 50).map((run) => request<unknown>(`/society/runs/${encodeURIComponent(run.id)}`)));
    liveReports.value = details.flatMap((detail, index) => {
      if (detail.status !== 'fulfilled') return [];
      const report = normalizeReport(normalizedRuns[index], detail.value);
      return report ? [report] : [];
    });
  }

  if (catalogResult.status === 'fulfilled') {
    catalogEntries.value = normalizeCatalog(catalogResult.value);
  } else {
    catalogError.value = `The reference catalog could not be read: ${errorMessage(catalogResult.reason)}`;
  }

  if (!liveReports.value.some((report) => report.id === selectedReportId.value)) {
    selectedReportId.value = liveReports.value[0]?.id ?? '';
  }
  if (runResult.status === 'rejected' && catalogResult.status === 'rejected' && offline.value) {
    liveReports.value = [];
    catalogEntries.value = [];
  }
  loading.value = false;
}

const selectedReport = computed(() => liveReports.value.find((report) => report.id === selectedReportId.value) ?? null);
const sourceTone = computed<SourceTone>(() => {
  if (loading.value) return 'loading';
  if (offline.value) return 'offline';
  if (runError.value || catalogError.value) return 'partial';
  return 'online';
});
const sourceLabel = computed(() => {
  if (sourceTone.value === 'loading') return 'Loading sources';
  if (sourceTone.value === 'offline') return 'Offline / unavailable';
  if (sourceTone.value === 'partial') return 'Partially available';
  return 'Backend sources reachable';
});
const sourceDetail = computed(() => {
  if (sourceTone.value === 'loading') return 'Waiting for report and catalog responses.';
  if (offline.value) return 'Neither live payloads nor catalog entries can be read.';
  if (sourceTone.value === 'partial') return 'At least one source returned an error; available data remains labeled.';
  return 'Live payloads are derived from Society run records; catalog entries are reference-only.';
});

function formatDate(value: string): string {
  if (!value) return 'Unavailable';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
}

async function copySelectedReport(): Promise<void> {
  if (!selectedReport.value || typeof navigator === 'undefined' || !navigator.clipboard) {
    copyLabel.value = 'Clipboard unavailable';
    return;
  }
  try {
    await navigator.clipboard.writeText(selectedReport.value.markdown);
    copyLabel.value = 'Copied';
    window.setTimeout(() => { copyLabel.value = 'Copy Markdown'; }, 1600);
  } catch {
    copyLabel.value = 'Copy failed';
  }
}

onMounted(() => {
  void loadReports();
});
</script>

<style scoped>
.reports-tool { display: flex; width: 100%; min-height: 100%; flex-direction: column; gap: 14px; padding: 22px; color: var(--text); font-size: 13px; }
.tool-header, .tool-header__actions, .panel-header, .tool-footer { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.tool-header { align-items: flex-start; }
.tool-header__copy { min-width: 0; }
.tool-eyebrow, .section-kicker, .source-strip__label, .metric-tile__label, .count-label, .reference-badge, .tool-footer > span:first-child { color: var(--text-muted); font: 700 10px/1.2 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; }
.tool-eyebrow, .section-kicker { margin: 0 0 5px; }
.tool-header h1 { margin: 0; color: var(--text); font-size: clamp(22px, 3vw, 30px); font-weight: 720; letter-spacing: -.035em; line-height: 1.1; }
.tool-intro { max-width: 760px; margin: 8px 0 0; color: var(--text-muted); line-height: 1.55; }
.tool-header__actions { flex: 0 0 auto; flex-wrap: wrap; justify-content: flex-end; }
.source-pill, .reference-badge { display: inline-flex; min-height: 25px; align-items: center; gap: 6px; border: 1px solid var(--border); border-radius: 999px; background: var(--surface-2); padding: 0 9px; white-space: nowrap; }
.source-pill--online { border-color: #b9dfd3; background: var(--success-soft); color: var(--success); }
.source-pill--partial { border-color: #ecd79c; background: var(--warning-soft); color: var(--warning); }
.source-pill--offline { border-color: #efc2c7; background: var(--danger-soft); color: var(--danger); }
.source-pill--loading { border-color: #cbd8ef; background: var(--accent-soft); color: var(--primary-focus); }
.source-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.source-pill--loading .source-dot { animation: pulse 1.2s ease-in-out infinite; }
.tool-button { display: inline-flex; min-height: 34px; align-items: center; justify-content: center; gap: 7px; border: 1px solid var(--border-light); border-radius: 6px; background: var(--surface); color: var(--text); cursor: pointer; font: 650 12px/1.2 var(--font); padding: 0 12px; white-space: nowrap; }
.tool-button:hover:not(:disabled) { border-color: var(--text); background: var(--surface-2); }
.tool-button:disabled { cursor: not-allowed; opacity: .45; }
.tool-button--small { min-height: 30px; padding: 0 9px; font-size: 11px; }
.source-strip { display: flex; min-height: 34px; align-items: center; flex-wrap: wrap; gap: 8px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 7px 10px; color: var(--text-muted); font-size: 11px; }
.source-strip code, .tool-footer code, .report-row code, .record-grid code, .catalog-row code { font-family: var(--font-mono); }
.source-strip__detail { margin-left: auto; }
.tool-notice { display: flex; flex-direction: column; gap: 3px; border: 1px solid var(--border); border-left: 3px solid var(--danger); border-radius: 6px; background: var(--danger-soft); padding: 10px 12px; color: var(--danger); line-height: 1.45; }
.metric-strip { display: grid; grid-template-columns: repeat(3, minmax(100px, 1fr)) minmax(200px, 1.5fr); overflow: hidden; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); }
.metric-tile { min-height: 66px; border-right: 1px solid var(--border); padding: 12px 14px; }
.metric-tile:last-child { border-right: 0; }
.metric-tile__value { display: block; margin-bottom: 3px; color: var(--text); font: 700 21px/1 var(--font-mono); }
.metric-tile__label { display: block; color: var(--text-muted); font-size: 9px; }
.metric-tile__caption { display: block; margin-top: 4px; color: var(--text-muted); font-size: 10px; }
.metric-tile--note { display: flex; flex-direction: column; justify-content: center; background: var(--surface-2); }
.reports-grid { display: grid; grid-template-columns: minmax(280px, .82fr) minmax(420px, 1.45fr); gap: 14px; min-height: 420px; }
.tool-panel { min-width: 0; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); padding: 14px; }
.panel-header { min-height: 34px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.panel-header h2 { margin: 0; color: var(--text); font-size: 14px; font-weight: 680; }
.count-label { color: var(--text-muted); font-size: 9px; white-space: nowrap; }
.state-block { display: flex; min-height: 220px; flex-direction: column; align-items: center; justify-content: center; gap: 7px; padding: 22px; color: var(--text-muted); text-align: center; line-height: 1.5; }
.state-block strong { color: var(--text); font-size: 13px; }
.state-block--loading { min-height: 180px; }
.state-mark { display: grid; width: 32px; height: 32px; place-items: center; border: 1px solid var(--border-light); border-radius: 50%; color: var(--text-muted); font: 700 15px/1 var(--font-mono); }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border-light); border-top-color: var(--primary); border-radius: 50%; animation: spin .75s linear infinite; }
.report-list { display: flex; max-height: 500px; flex-direction: column; gap: 6px; margin: 0; overflow-y: auto; padding: 0; list-style: none; }
.report-row { display: grid; width: 100%; grid-template-columns: 35px minmax(0, 1fr) auto; align-items: start; gap: 8px; border: 1px solid transparent; border-radius: 6px; background: transparent; color: var(--text); cursor: pointer; padding: 9px 8px; text-align: left; }
.report-row:hover { border-color: var(--border); background: var(--surface-2); }
.report-row--selected { border-color: var(--primary); background: var(--accent-soft); }
.report-row__type { color: var(--text-muted); font: 700 9px/1.4 var(--font-mono); letter-spacing: .06em; }
.report-row__body { display: flex; min-width: 0; flex-direction: column; gap: 3px; }
.report-row__body strong { overflow: hidden; font-size: 12px; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
.report-row__body > span:not(.report-row__meta) { overflow: hidden; color: var(--text-muted); font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
.report-row__meta { color: var(--text-muted); font-size: 9px; }
.report-row time { color: var(--text-muted); font: 9px/1.3 var(--font-mono); white-space: nowrap; }
.report-detail { display: flex; min-width: 0; flex-direction: column; gap: 12px; }
.record-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 0; }
.record-grid > div { min-width: 0; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 8px 10px; }
.record-grid dt { color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); letter-spacing: .06em; text-transform: uppercase; }
.record-grid dd { margin: 4px 0 0; overflow-wrap: anywhere; color: var(--text); font-size: 11px; line-height: 1.4; }
.provenance-note { display: flex; flex-direction: column; gap: 3px; border-left: 3px solid var(--primary); background: var(--accent-soft); padding: 9px 10px; color: var(--text-dim); font-size: 11px; line-height: 1.45; }
.provenance-note strong { color: var(--text); font-size: 10px; letter-spacing: .04em; text-transform: uppercase; }
.report-markdown { max-height: 440px; margin: 0; overflow: auto; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 12px; color: var(--text-dim); font: 10px/1.6 var(--font-mono); white-space: pre-wrap; overflow-wrap: anywhere; }
.catalog-panel { display: flex; flex-direction: column; gap: 9px; }
.reference-badge { color: var(--warning); }
.catalog-help { margin: 0; color: var(--text-muted); font-size: 11px; line-height: 1.5; }
.catalog-list { display: flex; flex-direction: column; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.catalog-row { display: grid; grid-template-columns: 130px minmax(0, 1fr) auto; gap: 10px; align-items: center; border-bottom: 1px solid var(--border); padding: 8px 10px; font-size: 11px; }
.catalog-row:last-child { border-bottom: 0; }
.catalog-row__id { color: var(--text-muted); }
.catalog-row__tag { color: var(--warning); font: 9px/1.2 var(--font-mono); text-transform: uppercase; }
.inline-state { color: var(--text-muted); font-size: 11px; line-height: 1.5; padding: 9px 0; }
.tool-footer { justify-content: flex-start; flex-wrap: wrap; color: var(--text-muted); font-size: 10px; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 50% { opacity: .35; } }
@media (max-width: 920px) { .reports-grid { grid-template-columns: 1fr; } .report-detail-panel { min-height: 360px; } }
@media (max-width: 620px) { .reports-tool { padding: 14px; } .tool-header { flex-direction: column; } .tool-header__actions { width: 100%; justify-content: flex-start; } .source-strip__detail { width: 100%; margin-left: 0; } .metric-strip { grid-template-columns: repeat(3, 1fr); } .metric-tile--note { grid-column: 1 / -1; border-top: 1px solid var(--border); } .record-grid { grid-template-columns: 1fr; } .report-row { grid-template-columns: 32px minmax(0, 1fr); } .report-row time { grid-column: 2; } .catalog-row { grid-template-columns: 1fr; gap: 3px; } .catalog-row__tag { justify-self: start; } }
@media (prefers-reduced-motion: reduce) { .spinner, .source-pill--loading .source-dot { animation: none; } }
</style>
