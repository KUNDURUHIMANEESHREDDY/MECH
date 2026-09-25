<template>
  <section class="reproduction-tool" aria-labelledby="paper-reproduction-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">General / Reproducibility</p>
        <h1 id="paper-reproduction-title">Paper reproduction</h1>
        <p class="tool-intro">
          Inspect reproduction artifacts attached to Society publications and the backend paper reference catalog.
          The active runtime does not expose a standalone paper reproduction endpoint, so missing evidence is called
          out instead of simulated.
        </p>
      </div>
      <div class="tool-header__actions">
        <span class="source-pill" :class="`source-pill--${sourceTone}`" role="status">
          <span class="source-dot" aria-hidden="true" />
          {{ sourceLabel }}
        </span>
        <button class="tool-button" type="button" :disabled="loading" @click="loadReproduction">
          <span aria-hidden="true">↻</span>
          Refresh reproduction data
        </button>
      </div>
    </header>

    <div class="source-strip" role="status" aria-live="polite">
      <span class="source-strip__label">Available sources</span>
      <code>publication.reproducibility</code>
      <code>GET /api/research_catalog?item_type=papers</code>
      <span class="source-strip__detail">The paper catalog is reference-only; run artifacts are live backend payloads.</span>
    </div>

    <div v-if="errorMessage" class="tool-notice" :class="`tool-notice--${offline ? 'warning' : 'error'}`" role="alert">
      <strong>{{ offline ? 'Reproduction sources offline' : 'Reproduction source unavailable' }}</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <section class="repro-status-grid" aria-label="Reproduction source status">
      <article class="status-card status-card--unavailable">
        <div class="status-card__top"><span class="status-card__icon" aria-hidden="true">—</span><span>Unavailable</span></div>
        <h2>Paper reproduction API</h2>
        <p>No standalone <code>/api/v2/science/reproduce</code> contract is mounted in the current runtime.</p>
        <code>Use a validation-capable backend for new runs.</code>
      </article>
      <article class="status-card" :class="artifacts.length ? 'status-card--online' : 'status-card--empty'">
        <div class="status-card__top"><span class="status-card__icon" aria-hidden="true">{{ artifacts.length ? '✓' : '—' }}</span><span>{{ artifacts.length ? 'Available' : 'No artifacts' }}</span></div>
        <h2>Run reproduction artifacts</h2>
        <p>{{ artifacts.length ? `${artifacts.length} Society publication${artifacts.length === 1 ? '' : 's'} returned reproducibility data.` : 'No selected run returned a reproducibility payload.' }}</p>
        <code>result.publication.reproducibility</code>
      </article>
      <article class="status-card status-card--reference">
        <div class="status-card__top"><span class="status-card__icon" aria-hidden="true">≡</span><span>Reference catalog</span></div>
        <h2>Paper index</h2>
        <p>{{ catalog.length }} catalog {{ catalog.length === 1 ? 'entry' : 'entries' }} returned for navigation.</p>
        <code>GET /api/research_catalog?item_type=papers</code>
      </article>
    </section>

    <div class="repro-layout">
      <section class="tool-panel artifact-panel" aria-labelledby="artifact-list-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Backend-returned payloads</p>
            <h2 id="artifact-list-title">Reproduction artifacts</h2>
          </div>
          <span class="count-label">{{ artifacts.length }} returned</span>
        </header>
        <div v-if="loading" class="state-block state-block--loading" role="status" aria-live="polite">
          <span class="spinner" aria-hidden="true" />
          <span>Reading Society publication records…</span>
        </div>
        <div v-else-if="offline" class="state-block" role="alert">
          <span class="state-mark" aria-hidden="true">!</span>
          <strong>Artifacts unavailable offline</strong>
          <span>Reconnect the runtime to inspect reproducibility payloads.</span>
        </div>
        <div v-else-if="!artifacts.length" class="state-block">
          <span class="state-mark" aria-hidden="true">∅</span>
          <strong>No reproduction artifacts</strong>
          <span>The run ledger returned no publication with a readable <code>reproducibility</code> object.</span>
        </div>
        <ul v-else class="artifact-list" aria-label="Reproduction artifacts">
          <li v-for="artifact in artifacts" :key="artifact.id">
            <button class="artifact-row" :class="{ 'artifact-row--selected': artifact.id === selectedArtifactId }" type="button" :aria-pressed="artifact.id === selectedArtifactId" @click="selectedArtifactId = artifact.id">
              <span class="artifact-row__mark" aria-hidden="true">↗</span>
              <span class="artifact-row__body"><strong>{{ artifact.title }}</strong><span>{{ artifact.goal || 'Goal unavailable' }}</span><code>{{ artifact.runId }}</code></span>
              <span class="artifact-row__status">{{ artifact.status || 'status unavailable' }}</span>
            </button>
          </li>
        </ul>
      </section>

      <section class="tool-panel detail-panel" aria-labelledby="artifact-detail-title">
        <header class="panel-header">
          <div>
            <p class="section-kicker">Read-only artifact</p>
            <h2 id="artifact-detail-title">{{ selectedArtifact ? selectedArtifact.title : 'Artifact detail' }}</h2>
          </div>
        </header>
        <div v-if="!selectedArtifact" class="state-block state-block--compact">
          <span class="state-mark" aria-hidden="true">↖</span>
          <strong>Select an artifact</strong>
          <span>Choose a returned payload to inspect its fields without interpretation.</span>
        </div>
        <div v-else class="artifact-detail">
          <dl class="record-grid">
            <div><dt>Source run</dt><dd><code>{{ selectedArtifact.runId }}</code></dd></div>
            <div><dt>Run status</dt><dd>{{ selectedArtifact.status || 'Unavailable' }}</dd></div>
            <div><dt>Paper ID</dt><dd>{{ selectedArtifact.paperId || 'Not returned' }}</dd></div>
            <div><dt>Generated</dt><dd>{{ formatDate(selectedArtifact.created) }}</dd></div>
          </dl>
          <div class="detail-note"><strong>Provenance</strong><span>These fields were returned inside the selected Society publication. No reproduction tier or fidelity verdict is calculated by this view.</span></div>
          <div class="payload-block">
            <div class="payload-block__heading"><h3>Returned payload</h3><span>JSON</span></div>
            <pre>{{ pretty(selectedArtifact.payload) }}</pre>
          </div>
        </div>
      </section>
    </div>

    <section class="tool-panel catalog-panel" aria-labelledby="paper-catalog-title">
      <header class="panel-header">
        <div>
          <p class="section-kicker">Illustrative index only</p>
          <h2 id="paper-catalog-title">Reference catalog</h2>
        </div>
        <span class="reference-badge">Reference catalog</span>
      </header>
      <p class="catalog-help">These entries are returned by the backend catalog contract. They identify paper records; they are not proof that a reproduction ran.</p>
      <div v-if="loading" class="inline-state">Loading reference catalog…</div>
      <div v-else-if="catalog.length" class="catalog-list">
        <div v-for="entry in catalog" :key="entry.id" class="catalog-row">
          <code>{{ entry.id }}</code>
          <span>{{ entry.title || 'Untitled paper entry' }}</span>
          <span class="catalog-row__tag">Reference catalog</span>
        </div>
      </div>
      <div v-else class="inline-state">No paper reference entries returned.</div>
    </section>

    <section class="next-step" aria-labelledby="reproduction-next-step-title">
      <div class="next-step__icon" aria-hidden="true">→</div>
      <div>
        <p class="section-kicker">Next available action</p>
        <h2 id="reproduction-next-step-title">Use a backend validation surface</h2>
        <p>Run a Society workflow or use the benchmark tools for executable comparisons. This view remains a read-only evidence surface until a dedicated reproduction contract is mounted.</p>
      </div>
      <a class="tool-button" href="#benchmark">Open benchmark tools</a>
    </section>

    <footer class="tool-footer">
      <span>Data policy</span>
      <span>Reference catalog entries and live reproduction payloads are kept visually and semantically separate.</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { apiUrl } from '../services/api';

type JsonRecord = Record<string, unknown>;
type SourceTone = 'online' | 'offline' | 'loading' | 'partial';
type Artifact = { id: string; runId: string; title: string; goal: string; status: string; paperId: string; created: string; payload: JsonRecord };
type CatalogEntry = { id: string; title: string };

const API_BASE = apiUrl('/api');
const artifacts = ref<Artifact[]>([]);
const catalog = ref<CatalogEntry[]>([]);
const selectedArtifactId = ref('');
const loading = ref(true);
const offline = ref(false);
const errorMessage = ref('');

function isRecord(value: unknown): value is JsonRecord { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function text(value: unknown, fallback = ''): string { return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? String(value) : fallback; }
function messageOf(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function isOffline(message: string): boolean { return /offline|failed to fetch|network|load|connection|timeout/i.test(message); }
async function request<T>(path: string): Promise<T> {
  if (typeof navigator !== 'undefined' && navigator.onLine === false) throw new Error('Browser is offline');
  const response = await fetch(`${API_BASE}${path}`, { headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error(`GET ${path} returned ${response.status}`);
  return response.json() as Promise<T>;
}
function normalizeRun(value: unknown): { id: string; goal: string; status: string; created: string } | null {
  if (!isRecord(value)) return null;
  const id = text(value.run_id) || text(value.id);
  if (!id) return null;
  return { id, goal: text(value.goal), status: text(value.status), created: text(value.created) || text(value.published_at) };
}
function extractArtifact(run: { id: string; goal: string; status: string; created: string }, raw: unknown): Artifact | null {
  if (!isRecord(raw)) return null;
  const result = isRecord(raw.result) ? raw.result : raw;
  const publication = isRecord(result.publication) ? result.publication : isRecord(raw.publication) ? raw.publication : null;
  const payload = publication?.reproducibility ?? result.reproducibility ?? raw.reproducibility;
  if (!isRecord(payload) || Object.keys(payload).length === 0) return null;
  return { id: `${run.id}:reproducibility`, runId: run.id, title: text(payload.paper_title) || text(payload.paper_id) || `Reproduction artifact · ${run.goal || run.id}`, goal: text(result.goal, run.goal), status: text(raw.status, text(result.status, run.status)), paperId: text(payload.paper_id) || text(payload.paper), created: text(payload.generated_at) || text(publication?.published_at, run.created), payload };
}
function normalizeCatalog(value: unknown): CatalogEntry[] {
  const values = Array.isArray(value) ? value : isRecord(value) && Array.isArray(value.catalog) ? value.catalog : [];
  return values.flatMap((item) => { if (!isRecord(item)) return []; const id = text(item.id) || text(item.paper_id); if (!id) return []; return [{ id, title: text(item.title) }]; });
}
function formatDate(value: string): string { if (!value) return 'Unavailable'; const date = new Date(value); return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }); }
function pretty(value: JsonRecord): string { try { return JSON.stringify(value, null, 2); } catch { return 'Payload could not be serialized.'; } }

const selectedArtifact = computed(() => artifacts.value.find((artifact) => artifact.id === selectedArtifactId.value) ?? null);
const sourceTone = computed<SourceTone>(() => loading.value ? 'loading' : offline.value ? 'offline' : errorMessage.value ? 'partial' : 'online');
const sourceLabel = computed(() => sourceTone.value === 'loading' ? 'Loading sources' : sourceTone.value === 'offline' ? 'Offline / unavailable' : sourceTone.value === 'partial' ? 'Partially available' : artifacts.value.length ? 'Run artifacts available' : 'Reference source reachable');

async function loadReproduction(): Promise<void> {
  loading.value = true;
  offline.value = false;
  errorMessage.value = '';
  const [runResult, catalogResult] = await Promise.allSettled([request<unknown>('/society/runs'), request<unknown>('/research_catalog?item_type=papers')]);
  const normalizedRuns = runResult.status === 'fulfilled' ? (Array.isArray(runResult.value) ? runResult.value : isRecord(runResult.value) && Array.isArray(runResult.value.runs) ? runResult.value.runs : []).map(normalizeRun).filter((run): run is { id: string; goal: string; status: string; created: string } => Boolean(run)) : [];
  if (runResult.status === 'fulfilled') {
    const details = await Promise.allSettled(normalizedRuns.slice(0, 50).map((run) => request<unknown>(`/society/runs/${encodeURIComponent(run.id)}`)));
    artifacts.value = details.flatMap((detail, index) => { const artifact = detail.status === 'fulfilled' ? extractArtifact(normalizedRuns[index], detail.value) : null; return artifact ? [artifact] : []; });
  } else {
    artifacts.value = [];
    errorMessage.value = `Society run source unavailable: ${messageOf(runResult.reason)}`;
    offline.value = isOffline(errorMessage.value);
  }
  if (catalogResult.status === 'fulfilled') catalog.value = normalizeCatalog(catalogResult.value);
  else {
    catalog.value = [];
    errorMessage.value = [errorMessage.value, `Paper catalog unavailable: ${messageOf(catalogResult.reason)}`].filter(Boolean).join(' · ');
    offline.value = offline.value || isOffline(messageOf(catalogResult.reason));
  }
  if (!artifacts.value.some((artifact) => artifact.id === selectedArtifactId.value)) selectedArtifactId.value = artifacts.value[0]?.id ?? '';
  loading.value = false;
}

onMounted(() => { void loadReproduction(); });
</script>

<style scoped>
.reproduction-tool { display: flex; width: 100%; min-height: 100%; flex-direction: column; gap: 14px; padding: 22px; color: var(--text); font-size: 13px; }
.tool-header, .tool-header__actions, .panel-header, .tool-footer, .next-step, .payload-block__heading, .status-card__top { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.tool-header { align-items: flex-start; }
.tool-header__copy { min-width: 0; }
.tool-eyebrow, .section-kicker, .source-strip__label, .count-label, .reference-badge, .tool-footer > span:first-child, .status-card__top > span:last-child, .catalog-row__tag { color: var(--text-muted); font: 700 10px/1.2 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; }
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
.tool-button { display: inline-flex; min-height: 34px; align-items: center; justify-content: center; gap: 7px; border: 1px solid var(--border-light); border-radius: 6px; background: var(--surface); color: var(--text); cursor: pointer; font: 650 12px/1.2 var(--font); padding: 0 12px; text-decoration: none; white-space: nowrap; }
.tool-button:hover:not(:disabled) { border-color: var(--text); background: var(--surface-2); }
.source-strip { display: flex; min-height: 34px; align-items: center; flex-wrap: wrap; gap: 8px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 7px 10px; color: var(--text-muted); font-size: 11px; }
.source-strip code, .tool-footer code, .next-step code, .status-card code, .artifact-row code, .record-grid code, .payload-block pre { font-family: var(--font-mono); }
.source-strip__detail { margin-left: auto; }
.tool-notice { display: flex; flex-direction: column; gap: 3px; border: 1px solid var(--border); border-left: 3px solid var(--danger); border-radius: 6px; background: var(--danger-soft); padding: 10px 12px; color: var(--danger); line-height: 1.45; }
.tool-notice--warning { border-left-color: var(--warning); background: var(--warning-soft); color: var(--warning); }
.repro-status-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.status-card { display: flex; min-height: 150px; flex-direction: column; gap: 7px; border: 1px solid var(--border); border-top: 3px solid var(--border-light); border-radius: 8px; background: var(--surface); padding: 12px; box-shadow: var(--shadow-sm); }
.status-card--online { border-top-color: var(--success); }
.status-card--unavailable, .status-card--reference { border-top-color: var(--warning); }
.status-card--empty { border-top-color: var(--border-light); }
.status-card__top { color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); text-transform: uppercase; }
.status-card__icon { color: var(--text-dim); font: 16px/1 var(--font-mono); }
.status-card--online .status-card__icon { color: var(--success); }
.status-card--unavailable .status-card__icon { color: var(--warning); }
.status-card h2 { margin: 0; color: var(--text); font-size: 13px; font-weight: 680; }
.status-card p { flex: 1; margin: 0; color: var(--text-muted); font-size: 10px; line-height: 1.5; }
.status-card code { color: var(--text-muted); font-size: 9px; }
.repro-layout { display: grid; grid-template-columns: minmax(350px, .9fr) minmax(420px, 1.1fr); gap: 14px; min-height: 400px; }
.tool-panel { min-width: 0; border: 1px solid var(--border); border-radius: 9px; background: var(--surface); box-shadow: var(--shadow-sm); padding: 14px; }
.panel-header { min-height: 34px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.panel-header h2 { margin: 0; color: var(--text); font-size: 14px; font-weight: 680; }
.count-label { font-size: 9px; white-space: nowrap; }
.state-block { display: flex; min-height: 220px; flex-direction: column; align-items: center; justify-content: center; gap: 7px; padding: 22px; color: var(--text-muted); text-align: center; line-height: 1.5; }
.state-block--compact { min-height: 180px; }
.state-block--loading { min-height: 180px; }
.state-block strong { color: var(--text); font-size: 13px; }
.state-block code { font-family: var(--font-mono); font-size: 10px; }
.state-mark { display: grid; width: 32px; height: 32px; place-items: center; border: 1px solid var(--border-light); border-radius: 50%; color: var(--text-muted); font: 700 15px/1 var(--font-mono); }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border-light); border-top-color: var(--primary); border-radius: 50%; animation: spin .75s linear infinite; }
.artifact-list { display: flex; max-height: 420px; flex-direction: column; gap: 6px; margin: 0; overflow-y: auto; padding: 0; list-style: none; }
.artifact-row { display: grid; width: 100%; grid-template-columns: 20px minmax(0, 1fr) auto; align-items: start; gap: 8px; border: 1px solid transparent; border-radius: 6px; background: transparent; color: var(--text); cursor: pointer; padding: 9px 8px; text-align: left; }
.artifact-row:hover { border-color: var(--border); background: var(--surface-2); }
.artifact-row--selected { border-color: var(--primary); background: var(--accent-soft); }
.artifact-row__mark { color: var(--text-muted); font: 16px/1.2 var(--font-mono); }
.artifact-row__body { display: flex; min-width: 0; flex-direction: column; gap: 3px; }
.artifact-row__body strong { overflow: hidden; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.artifact-row__body > span { overflow: hidden; color: var(--text-muted); font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
.artifact-row__body code { color: var(--text-muted); font-size: 9px; }
.artifact-row__status { color: var(--text-muted); font: 9px/1.3 var(--font-mono); white-space: nowrap; }
.artifact-detail { display: flex; flex-direction: column; gap: 12px; }
.record-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 0; }
.record-grid > div { border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 8px 10px; }
.record-grid dt { color: var(--text-muted); font: 700 9px/1.2 var(--font-mono); letter-spacing: .06em; text-transform: uppercase; }
.record-grid dd { margin: 4px 0 0; overflow-wrap: anywhere; color: var(--text); font-size: 11px; line-height: 1.4; }
.detail-note { display: flex; flex-direction: column; gap: 3px; border-left: 3px solid var(--primary); background: var(--accent-soft); padding: 9px 10px; color: var(--text-dim); font-size: 10px; line-height: 1.45; }
.detail-note strong { color: var(--text); font-size: 10px; text-transform: uppercase; }
.payload-block { display: flex; flex-direction: column; gap: 7px; }
.payload-block__heading { border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.payload-block h3 { margin: 0; font-size: 12px; }
.payload-block__heading span { color: var(--text-muted); font: 9px/1.2 var(--font-mono); }
.payload-block pre { max-height: 320px; margin: 0; overflow: auto; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); padding: 10px; color: var(--text-dim); font-size: 9px; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; }
.catalog-panel { display: flex; flex-direction: column; gap: 8px; }
.reference-badge { color: var(--warning); }
.catalog-help { margin: 0; color: var(--text-muted); font-size: 10px; line-height: 1.5; }
.catalog-list { display: flex; flex-direction: column; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.catalog-row { display: grid; grid-template-columns: 120px minmax(0, 1fr) auto; gap: 8px; align-items: center; border-bottom: 1px solid var(--border); padding: 8px 10px; font-size: 11px; }
.catalog-row:last-child { border-bottom: 0; }
.catalog-row code { color: var(--text-muted); font: 9px/1.3 var(--font-mono); }
.catalog-row__tag { color: var(--warning); font-size: 9px; }
.inline-state { color: var(--text-muted); font-size: 11px; line-height: 1.5; padding: 10px 0; }
.next-step { align-items: center; border: 1px dashed var(--border-light); border-radius: 8px; background: var(--surface-2); padding: 13px; }
.next-step__icon { display: grid; width: 30px; height: 30px; flex: 0 0 auto; place-items: center; border: 1px solid var(--border-light); border-radius: 50%; color: var(--primary); font: 700 15px/1 var(--font-mono); }
.next-step > div:nth-child(2) { flex: 1; }
.next-step h2 { margin: 0; font-size: 13px; }
.next-step p:last-child { max-width: 700px; margin: 5px 0 0; color: var(--text-muted); font-size: 10px; line-height: 1.5; }
.tool-footer { justify-content: flex-start; flex-wrap: wrap; color: var(--text-muted); font-size: 10px; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 50% { opacity: .35; } }
@media (max-width: 900px) { .repro-layout { grid-template-columns: 1fr; } }
@media (max-width: 620px) { .reproduction-tool { padding: 14px; } .tool-header { flex-direction: column; } .tool-header__actions { width: 100%; justify-content: flex-start; } .source-strip__detail { width: 100%; margin-left: 0; } .repro-status-grid { grid-template-columns: 1fr; } .record-grid { grid-template-columns: 1fr; } .next-step { align-items: flex-start; flex-wrap: wrap; } .next-step .tool-button { width: 100%; } .catalog-row { grid-template-columns: 1fr; gap: 3px; } .catalog-row__tag { justify-self: start; } }
@media (prefers-reduced-motion: reduce) { .spinner, .source-pill--loading .source-dot { animation: none; } }
</style>
