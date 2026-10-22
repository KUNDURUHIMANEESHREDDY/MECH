<template>
  <section class="datasets-surface" data-testid="datasets">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Research / On-disk data</p>
        <h2 class="surface-title">Datasets</h2>
        <p class="surface-description">
          Lists the datasets that exist on disk (<code>GET /api/datasets</code>) and previews bounded
          rows from a real file (<code>POST /api/datasets/preview</code>). Digests and counts are the
          backend's; nothing is inferred locally.
        </p>
      </div>
      <div class="header-actions">
        <button class="quiet-button" type="button" data-testid="datasets-refresh" :disabled="isLoading" @click="loadDatasets">
          {{ isLoading ? 'Loading…' : 'Refresh datasets' }}
        </button>
      </div>
    </header>

    <p v-if="stalePreview" class="notice notice-warn" role="status" data-testid="datasets-stale">
      <strong>Stale preview.</strong>
      The preview below was read for {{ preview?.label }}; the selection has since changed. Preview
      again before treating those rows as the selected dataset's.
    </p>

    <div
      v-if="listFailure"
      class="notice"
      :class="listFailure.kind === 'unsupported' ? 'notice-unsupported' : 'notice-error'"
      role="alert"
      data-testid="datasets-failure"
    >
      <strong>{{ listFailure.kind === 'unsupported' ? 'NOT AVAILABLE' : 'Datasets could not be listed' }}</strong>
      <p>{{ listFailure.message }}</p>
      <p v-if="listFailure.kind === 'unsupported'" class="notice-foot">
        This deployment does not serve <code>GET /api/datasets</code>. No sample dataset is
        substituted.
      </p>
    </div>

    <p v-if="isLoading" class="notice" role="status" aria-live="polite" data-testid="datasets-loading">
      <strong>Reading the dataset directory…</strong>
      <p>The backend reads each <code>dataset.json</code>; this view waits for that answer.</p>
    </p>

    <section v-if="listState === 'ok'" class="panel" aria-labelledby="datasets-list-title">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">On-disk catalogue</p>
          <h3 id="datasets-list-title">{{ datasets.length }} dataset(s) listed</h3>
          <p class="panel-description">
            Source returned by the backend: <span class="mono">{{ sourcePath || 'not returned' }}</span>
          </p>
        </div>
        <span class="source-badge">{{ listProvenance || 'PROVENANCE NOT RETURNED' }}</span>
      </header>

      <ul class="dataset-list" data-testid="datasets-list" aria-label="Datasets returned by the backend">
        <li
          v-for="dataset in datasets"
          :key="datasetId(dataset)"
          class="dataset-card"
          :class="{ 'is-selected': selectedId === datasetId(dataset) }"
        >
          <div class="dataset-head">
            <button
              type="button"
              class="dataset-select"
              :aria-current="selectedId === datasetId(dataset) ? 'true' : undefined"
              :aria-label="`Select dataset ${datasetId(dataset)}`"
              :data-testid="`datasets-select-${datasetId(dataset)}`"
              @click="selectDataset(datasetId(dataset))"
            >
              <span class="dataset-id mono">{{ datasetId(dataset) }}</span>
              <span class="dataset-status" :class="`status-${entryStatus(dataset)}`">
                {{ entryStatus(dataset).toUpperCase() }}
              </span>
            </button>
            <span v-if="promptCountLabel(dataset)" class="dataset-count">
              {{ promptCountLabel(dataset) }} prompts
            </span>
          </div>

          <dl class="dataset-facts">
            <div>
              <dt>dataset.json</dt>
              <dd class="mono break">{{ asString(dataset.path) || 'not returned' }}</dd>
            </div>
            <div>
              <dt>sha256 of the bytes read</dt>
              <dd class="mono break">{{ provenanceDigest(dataset) || 'not returned' }}</dd>
            </div>
            <div>
              <dt>Columns</dt>
              <dd>{{ columnsLabel(dataset) }}</dd>
            </div>
            <div>
              <dt>Integrity</dt>
              <dd data-testid="datasets-integrity">
                {{ integrityLabel(dataset) }}
              </dd>
            </div>
          </dl>

          <p v-if="integrityNote(dataset)" class="dataset-note">{{ integrityNote(dataset) }}</p>
          <p v-if="entryReason(dataset)" class="dataset-note dataset-note-warn">
            Backend reason: {{ entryReason(dataset) }}
          </p>
          <p v-if="entryNote(dataset)" class="dataset-note">{{ entryNote(dataset) }}</p>

          <div class="dataset-actions">
            <button
              class="quiet-button"
              type="button"
              :disabled="previewingId === datasetId(dataset)"
              :data-testid="`datasets-preview-${datasetId(dataset)}`"
              @click="previewDatasetById(dataset)"
            >
              {{ previewingId === datasetId(dataset) ? 'Previewing…' : 'Preview rows' }}
            </button>
          </div>
        </li>
      </ul>

      <footer v-if="listNote" class="panel-footnote">{{ listNote }}</footer>
    </section>

    <section v-else-if="listState === 'empty'" class="panel" aria-labelledby="datasets-empty-title" data-testid="datasets-empty">
      <p class="panel-kicker">On-disk catalogue</p>
      <h3 id="datasets-empty-title">No datasets were listed</h3>
      <p class="empty-copy">{{ emptyReason }}</p>
      <p class="empty-copy">
        An empty catalogue is not an empty dataset: no rows, columns or metrics are shown, because
        none were returned.
      </p>
      <p class="empty-copy mono">Directory: {{ sourcePath || 'not returned' }}</p>
    </section>

    <section class="panel" aria-labelledby="datasets-preview-title">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">Bounded read</p>
          <h3 id="datasets-preview-title">Preview</h3>
          <p class="panel-description">
            Rows come from the file's own bytes. A file above the backend's byte limit is refused
            rather than truncated.
          </p>
        </div>
        <div class="limit-form">
          <label class="field-label" for="datasets-limit">Rows to return</label>
          <input
            id="datasets-limit"
            v-model.number="limit"
            class="text-input tiny-input"
            type="number"
            min="1"
            max="20"
            step="1"
            data-testid="datasets-limit"
          />
        </div>
      </header>

      <form class="path-form" @submit.prevent="previewByPath">
        <label class="field-label" for="datasets-path">Workspace file path</label>
        <input
          id="datasets-path"
          v-model="pathInput"
          class="text-input"
          type="text"
          autocomplete="off"
          spellcheck="false"
          placeholder="relative to the workspace root"
          data-testid="datasets-path"
        />
        <button class="quiet-button" type="submit" data-testid="datasets-preview-path" :disabled="previewingPath || !pathInput.trim()">
          {{ previewingPath ? 'Reading…' : 'Preview path' }}
        </button>
      </form>

      <div
        v-if="previewFailure"
        class="notice"
        :class="previewFailure.kind === 'unsupported' ? 'notice-unsupported' : 'notice-error'"
        role="alert"
        data-testid="datasets-preview-error"
      >
        <strong>{{ previewFailure.kind === 'unsupported' ? 'NOT AVAILABLE' : 'Preview refused' }}</strong>
        <p>{{ previewFailure.message }}</p>
        <p v-if="previewFailure.kind === 'error'" class="notice-foot">
          The backend's own validation message is shown verbatim; the request was refused, so no
          rows are displayed.
        </p>
      </div>

      <p v-if="!preview && !previewFailure && !isPreviewing" class="panel-message" data-testid="datasets-preview-idle">
        No preview yet. Select a dataset above, or name a workspace file.
      </p>

      <p v-if="isPreviewing" class="panel-message" role="status" aria-live="polite" data-testid="datasets-preview-loading">
        Reading bounded rows…
      </p>

      <template v-if="preview">
        <dl class="preview-facts" data-testid="datasets-preview-facts">
          <div><dt>Source</dt><dd class="mono break">{{ preview.label }}</dd></div>
          <div><dt>Format</dt><dd>{{ previewFormat }}</dd></div>
          <div><dt>Rows returned / total</dt><dd class="mono">{{ previewRowCounts }}</dd></div>
          <div><dt>Bytes read</dt><dd class="mono">{{ previewBytes }}</dd></div>
          <div><dt>sha256 of the bytes read</dt><dd class="mono break">{{ previewDigest }}</dd></div>
        </dl>

        <p v-if="previewTruncated" class="notice notice-warn" role="status" data-testid="datasets-preview-partial">
          <strong>Partial preview.</strong>
          {{ previewRowCounts }} rows were returned; the backend's limit for this request is
          {{ previewLimits }}. This is a bounded read, not the whole file.
        </p>

        <table v-if="previewRows.length" class="preview-table" data-testid="datasets-preview-table">
          <caption>
            Rows returned by <code>POST /api/datasets/preview</code>. A cell the response did not
            carry reads “—” rather than zero.
          </caption>
          <thead>
            <tr>
              <th scope="col">#</th>
              <th v-for="column in previewColumns" :key="column" scope="col">{{ column }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, rowIndex) in previewRows" :key="`row-${rowIndex}`">
              <th scope="row">{{ rowIndex + 1 }}</th>
              <td v-for="column in previewColumns" :key="`${rowIndex}-${column}`" class="mono">
                {{ cellText(row, column) }}
              </td>
            </tr>
          </tbody>
        </table>
        <p v-else class="panel-message" data-testid="datasets-preview-empty">
          The backend returned no rows for this file. Zero rows is not a zero value.
        </p>

        <p v-if="previewNote" class="panel-footnote">{{ previewNote }}</p>
      </template>
    </section>

    <section class="panel" aria-labelledby="datasets-split-title" data-testid="datasets-split-unavailable">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">Evaluation setup</p>
          <h3 id="datasets-split-title">Split &amp; metric configuration</h3>
        </div>
      </header>
      <div class="unavailable-block" role="note">
        <strong>NOT AVAILABLE — splits and metrics</strong>
        <p>
          <code>GET /api/datasets</code> returns each dataset's path, digest, prompt count, columns
          and integrity state; <code>POST /api/datasets/preview</code> returns bounded rows. Neither
          response carries a split definition, a metric set or an evaluation configuration, so no
          split or metric controls are rendered here.
        </p>
        <p>Controls that cannot be backed by the API are not invented.</p>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import {
  asArray,
  asNumber,
  asString,
  describeError,
  displayValue,
  isRecord,
  isUnsupportedError,
  listDatasets,
  previewDataset,
  provenanceOf,
  statusOf,
  type JsonRecord,
} from '../services/interpretabilityClient';

interface Failure {
  kind: 'unsupported' | 'error';
  message: string;
}

interface Preview {
  label: string;
  /** Dataset id this preview was read for; null for an explicit path preview. */
  datasetId: string | null;
  record: JsonRecord;
}

type ListState = 'idle' | 'ok' | 'empty';

const datasets = ref<JsonRecord[]>([]);
const listState = ref<ListState>('idle');
const listFailure = ref<Failure | null>(null);
const isLoading = ref(false);
const listNote = ref('');
const listProvenance = ref('');
const sourcePath = ref('');
const emptyReason = ref('');

const selectedId = ref('');
const limit = ref(10);

const pathInput = ref('');
const preview = ref<Preview | null>(null);
const previewFailure = ref<Failure | null>(null);
const isPreviewing = ref(false);
const previewingId = ref('');
const previewingPath = ref(false);

/** Increments per preview request; a slower older response cannot overwrite a newer one. */
let previewSequence = 0;

onMounted(() => {
  void loadDatasets();
});

const stalePreview = computed(() => {
  const shown = preview.value;
  if (!shown || shown.datasetId === null) return false;
  return Boolean(selectedId.value) && shown.datasetId !== selectedId.value;
});

const previewColumns = computed<string[]>(() =>
  asArray(preview.value?.record.columns).map((column) => (typeof column === 'string' ? column : String(column))),
);

const previewRows = computed<JsonRecord[]>(() => asArray(preview.value?.record.rows).filter(isRecord));

const previewFormat = computed(() => {
  const source = preview.value?.record.source;
  return isRecord(source) ? asString(source.format) || 'not returned' : 'not returned';
});

const previewRowCounts = computed(() => {
  const record = preview.value?.record;
  if (!record) return '—';
  const returned = asNumber(record.returned_rows);
  const total = asNumber(record.total_rows);
  return `${returned === null ? '—' : returned} / ${total === null ? '—' : total}`;
});

const previewBytes = computed(() => {
  const source = preview.value?.record.source;
  const bytes = isRecord(source) ? asNumber(source.bytes) : null;
  return bytes === null ? '—' : String(bytes);
});

const previewDigest = computed(() => {
  const source = preview.value?.record.source;
  const digest = isRecord(source) ? asString(source.sha256) : '';
  return digest || 'not returned';
});

const previewTruncated = computed(() => preview.value?.record.truncated === true);

const previewLimits = computed(() => {
  const limits = preview.value?.record.limits;
  if (!isRecord(limits)) return 'not returned';
  const maxRows = asNumber(limits.max_rows);
  const maxBytes = asNumber(limits.max_bytes);
  return `max_rows ${maxRows === null ? '—' : maxRows}, max_bytes ${maxBytes === null ? '—' : maxBytes}`;
});

const previewNote = computed(() => {
  const record = preview.value?.record;
  if (!record) return '';
  const provenanceNote = isRecord(record.provenance) ? asString(record.provenance.note) : '';
  return provenanceNote;
});

async function loadDatasets() {
  isLoading.value = true;
  listFailure.value = null;
  try {
    const record = await listDatasets();
    const state = statusOf(record);
    const entries = asArray(record.datasets).filter(isRecord);
    datasets.value = entries;
    sourcePath.value = asString(record.source);
    listNote.value = asString(record.note);
    listProvenance.value = provenanceOf(record);

    if (state === 'empty') {
      listState.value = 'empty';
      emptyReason.value = asString(record.reason) || 'The backend returned an empty catalogue.';
      return;
    }
    if (state === 'ok' || entries.length > 0) {
      listState.value = entries.length > 0 ? 'ok' : 'empty';
      emptyReason.value =
        asString(record.reason) ||
        'The backend listed zero datasets. Nothing was listed and nothing was invented.';
      return;
    }
    listState.value = 'idle';
    listFailure.value = {
      kind: state === 'unavailable' ? 'unsupported' : 'error',
      message:
        asString(record.error) ||
        asString(record.reason) ||
        `The backend returned status “${state || 'none'}” without a reason.`,
    };
  } catch (error) {
    datasets.value = [];
    listState.value = 'idle';
    listFailure.value = {
      kind: isUnsupportedError(error) ? 'unsupported' : 'error',
      message: describeError(error),
    };
  } finally {
    isLoading.value = false;
  }
}

function selectDataset(id: string) {
  // The previous preview is kept on screen but flagged stale below: dropping it
  // silently would leave the reader unsure whether rows had ever been read.
  selectedId.value = id;
  previewFailure.value = null;
}

async function previewDatasetById(dataset: JsonRecord) {
  const id = datasetId(dataset);
  const sequence = (previewSequence += 1);
  isPreviewing.value = true;
  previewingId.value = id;
  previewFailure.value = null;
  preview.value = null;
  try {
    const record = await previewDataset({ dataset_id: id, limit: clampLimit() });
    if (sequence !== previewSequence) return;
    preview.value = { label: `dataset_id: ${id}`, datasetId: id, record };
  } catch (error) {
    if (sequence !== previewSequence) return;
    previewFailure.value = {
      kind: isUnsupportedError(error) ? 'unsupported' : 'error',
      message: describeError(error),
    };
  } finally {
    if (sequence === previewSequence) {
      isPreviewing.value = false;
      previewingId.value = '';
    }
  }
}

async function previewByPath() {
  const path = pathInput.value.trim();
  if (!path) return;
  const sequence = (previewSequence += 1);
  isPreviewing.value = true;
  previewingPath.value = true;
  previewFailure.value = null;
  preview.value = null;
  try {
    const record = await previewDataset({ path, limit: clampLimit() });
    if (sequence !== previewSequence) return;
    preview.value = { label: `path: ${path}`, datasetId: null, record };
  } catch (error) {
    if (sequence !== previewSequence) return;
    previewFailure.value = {
      kind: isUnsupportedError(error) ? 'unsupported' : 'error',
      message: describeError(error),
    };
  } finally {
    if (sequence === previewSequence) {
      isPreviewing.value = false;
      previewingPath.value = false;
    }
  }
}

function clampLimit(): number {
  const value = typeof limit.value === 'number' && Number.isFinite(limit.value) ? Math.trunc(limit.value) : 10;
  return Math.min(20, Math.max(1, value));
}

function datasetId(dataset: JsonRecord): string {
  return asString(dataset.id);
}

function entryStatus(dataset: JsonRecord): string {
  return asString(dataset.status) || 'unknown';
}

function promptCountLabel(dataset: JsonRecord): string {
  const count = asNumber(dataset.prompt_count);
  return count === null ? '' : String(count);
}

function provenanceDigest(dataset: JsonRecord): string {
  const provenance = dataset.provenance;
  return isRecord(provenance) ? asString(provenance.sha256) : '';
}

function columnsLabel(dataset: JsonRecord): string {
  const columns = asArray(dataset.columns).map((column) => (typeof column === 'string' ? column : String(column)));
  if (!columns.length) return 'not returned';
  const shown = columns.slice(0, 6).join(', ');
  return columns.length > 6 ? `${shown} (+${columns.length - 6} more)` : shown;
}

function integrityOf(dataset: JsonRecord): JsonRecord {
  return isRecord(dataset.integrity) ? dataset.integrity : {};
}

function integrityLabel(dataset: JsonRecord): string {
  const integrity = integrityOf(dataset);
  const registered = integrity.registered;
  const verified = integrity.integrity_verified;
  if (typeof registered !== 'boolean') return 'registered: not determined · verified: not determined';
  if (!registered) return 'registered: no (not in golden_manifest.json) · verified: no';
  if (verified === true) return 'registered: yes · verified: yes';
  if (verified === false) return 'registered: yes · verified: no';
  return 'registered: yes · verified: not determined';
}

function integrityNote(dataset: JsonRecord): string {
  const integrity = integrityOf(dataset);
  const note = asString(integrity.note);
  if (note) return note;
  const bypassed = integrity.bypassed === true;
  return bypassed ? 'Integrity checks were bypassed for this load.' : '';
}

function entryReason(dataset: JsonRecord): string {
  return asString(dataset.reason);
}

function entryNote(dataset: JsonRecord): string {
  return asString(dataset.note);
}

function cellText(row: JsonRecord, column: string): string {
  if (!(column in row)) return '—';
  return displayValue(row[column]);
}
</script>

<style scoped>
.datasets-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.panel-heading,
.header-actions,
.dataset-head,
.dataset-actions,
.path-form,
.limit-form {
  display: flex;
  gap: 10px;
}

.surface-header,
.panel-heading {
  align-items: flex-start;
  justify-content: space-between;
  flex-wrap: wrap;
}

.surface-header {
  margin-bottom: 18px;
}

.header-copy {
  flex: 1 1 340px;
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
  font-size: clamp(22px, 3vw, 30px);
  font-weight: 750;
  letter-spacing: -0.035em;
}

.surface-description,
.panel-description,
.empty-copy {
  margin: 7px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.55;
}

.header-actions {
  align-items: center;
  flex-wrap: wrap;
}

.quiet-button {
  min-height: 32px;
  padding: 0 11px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text-muted);
  cursor: pointer;
  font: 700 11px/1 var(--font);
}

.quiet-button:hover:not(:disabled) {
  color: var(--text);
  border-color: var(--border-light);
}

.quiet-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.notice {
  margin: 0 0 14px;
  padding: 11px 13px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--border-light);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text-dim);
  font-size: 12px;
  line-height: 1.5;
}

.notice strong {
  display: block;
  margin-bottom: 3px;
  color: var(--text);
}

.notice p {
  margin: 0;
}

.notice-warn {
  border-left-color: var(--warning);
}

.notice-error {
  border-left-color: var(--danger);
}

.notice-unsupported {
  border-left-color: var(--text-muted);
  border-style: dashed;
}

.notice-foot {
  margin-top: 6px !important;
  color: var(--text-muted);
  font-size: 11px;
}

.panel {
  margin-bottom: 14px;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.panel-heading {
  margin-bottom: 12px;
}

.panel-heading h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 700;
}

.source-badge {
  padding: 3px 6px;
  border: 1px solid var(--border-light);
  border-radius: 999px;
  color: var(--text-muted);
  font: 700 9px/1 var(--font-mono);
  letter-spacing: 0.06em;
  white-space: nowrap;
}

.dataset-list {
  display: grid;
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.dataset-card {
  padding: 11px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
}

.dataset-card.is-selected {
  border-color: var(--primary);
  box-shadow: inset 0 0 0 1px var(--primary);
}

.dataset-head {
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
}

.dataset-select {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 6px;
  border: 1px solid transparent;
  border-radius: 5px;
  background: transparent;
  color: var(--text);
  cursor: pointer;
  font: 700 12px/1 var(--font-mono);
}

.dataset-select:hover {
  border-color: var(--border-light);
}

.dataset-select:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 1px;
}

.dataset-id {
  font-weight: 700;
}

.dataset-status {
  padding: 2px 6px;
  border: 1px solid var(--border-light);
  border-radius: 999px;
  color: var(--text-muted);
  font: 700 9px/1.4 var(--font-mono);
  letter-spacing: 0.05em;
}

.status-ok {
  border-color: var(--success);
  color: var(--success);
}

.status-unregistered {
  border-color: var(--warning);
  color: var(--warning);
}

.status-error {
  border-color: var(--danger);
  color: var(--danger);
}

.dataset-count {
  color: var(--text-muted);
  font: 500 11px/1 var(--font-mono);
}

.dataset-facts,
.preview-facts {
  display: grid;
  gap: 8px 16px;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  margin: 10px 0 0;
}

.dataset-facts dt,
.preview-facts dt {
  color: var(--text-muted);
  font: 700 9px/1.4 var(--font-mono);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.dataset-facts dd,
.preview-facts dd {
  margin: 2px 0 0;
  font-size: 12px;
  line-height: 1.45;
}

.dataset-note {
  margin: 9px 0 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.5;
}

.dataset-note-warn {
  color: var(--text-dim);
}

.dataset-actions {
  justify-content: flex-end;
  margin-top: 10px;
}

.mono {
  font-family: var(--font-mono);
}

.break {
  overflow-wrap: anywhere;
}

.field-label {
  color: var(--text-muted);
  font: 700 9px/1.4 var(--font-mono);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.text-input {
  min-width: 0;
  flex: 1 1 220px;
  min-height: 32px;
  padding: 0 9px;
  border: 1px solid var(--border-light);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  font: 500 12px/1 var(--font-mono);
}

.tiny-input {
  flex: 0 0 78px;
  max-width: 78px;
}

.limit-form {
  flex-direction: column;
  gap: 4px;
}

.path-form {
  align-items: flex-end;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.path-form .field-label {
  width: 100%;
}

.preview-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}

.preview-table caption {
  caption-side: top;
  padding-bottom: 8px;
  color: var(--text-muted);
  font: 400 11px/1.5 var(--font);
  text-align: left;
}

.preview-table th,
.preview-table td {
  padding: 5px 6px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
  max-width: 260px;
  overflow-wrap: anywhere;
}

.preview-table thead th {
  color: var(--text-muted);
  font: 700 9px/1.4 var(--font-mono);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.panel-message {
  margin: 0;
  padding: 12px;
  border: 1px dashed var(--border-light);
  border-radius: 6px;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.unavailable-block {
  padding: 12px;
  border: 1px dashed var(--border-light);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text-dim);
  font-size: 12px;
  line-height: 1.55;
}

.unavailable-block strong {
  display: block;
  margin-bottom: 5px;
  color: var(--text);
  font: 700 10px/1.4 var(--font-mono);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.unavailable-block p {
  margin: 0 0 6px;
}

.panel-footnote {
  margin: 10px 0 0;
  color: var(--text-muted);
  font-size: 10px;
  line-height: 1.5;
}
</style>
