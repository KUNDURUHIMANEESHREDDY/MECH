<template>
  <section class="activation-surface" data-testid="activation-browser">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Explore / Live tensors</p>
        <h2 class="surface-title">Activations</h2>
        <p class="surface-description">
          Reads the residual stream and post-GELU MLP activations that GPT-2 produced for one prompt
          (<code>POST /api/gpt2/activations</code>, then one
          <code>POST /api/gpt2/layer_activations</code> per layer). Every number below came from a
          response; the grid is built from the returned matrices.
        </p>
      </div>
      <form class="capture-form" @submit.prevent="runCapture">
        <label class="field-label" for="activation-prompt">Prompt</label>
        <input
          id="activation-prompt"
          v-model="prompt"
          class="text-input"
          type="text"
          autocomplete="off"
          spellcheck="false"
          data-testid="activations-prompt"
          :disabled="isCapturing"
        />
        <button
          class="primary-button"
          type="submit"
          data-testid="activations-capture"
          :disabled="isCapturing || !prompt.trim()"
        >
          {{ captureButtonLabel }}
        </button>
      </form>
      <p class="help-copy" role="status" data-testid="activation-prompt-rotation">{{ promptRotationNote }}</p>
    </header>

    <p v-if="stale" class="notice notice-warn" role="status" data-testid="activations-stale">
      <strong>Stale capture.</strong>
      The matrix below was captured for “{{ capture?.prompt }}”; the prompt field now reads
      “{{ prompt.trim() }}”. Recapture before treating these values as the current prompt's.
    </p>

    <div
      v-if="failure"
      class="notice"
      :class="failure.kind === 'unsupported' ? 'notice-unsupported' : 'notice-error'"
      role="alert"
      data-testid="activations-failure"
    >
      <strong>{{ failure.kind === 'unsupported' ? 'NOT AVAILABLE' : 'Capture failed' }}</strong>
      <p>{{ failure.message }}</p>
      <p v-if="failure.kind === 'unsupported'" class="notice-foot">
        No matrix is drawn for this state: an empty matrix would read as zero activation, and zero
        activation is a measurement the backend did not make.
      </p>
    </div>

    <div v-if="isCapturing" class="notice" role="status" aria-live="polite" data-testid="activations-progress">
      <strong>Capturing…</strong>
      <p>{{ progress.done }} of {{ progress.total }} layers returned.</p>
      <div class="progress-track" aria-hidden="true"><span class="progress-fill" :style="{ width: progressPercent }" /></div>
    </div>

    <p v-if="!capture && !failure && !isCapturing" class="notice" data-testid="activations-idle">
      <strong>Idle.</strong>
      Nothing has been captured yet. Capture runs real forward passes, so it is never started on
      mount.
    </p>

    <template v-if="capture">
      <section class="panel" aria-labelledby="activation-contract-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Capability probe</p>
            <h3 id="activation-contract-title">What the capture reported</h3>
            <p class="panel-description">
              Shapes returned by <code>POST /api/gpt2/activations</code> for layer
              {{ formatInteger(probeLayer) }}.
            </p>
          </div>
          <span class="source-badge">{{ probeProvenance || 'PROVENANCE NOT RETURNED' }}</span>
        </header>
        <dl class="fact-grid" data-testid="activations-probe">
          <div>
            <dt>resid_shape</dt>
            <dd class="mono">{{ formatShape(capture.probe.resid_shape) }}</dd>
          </div>
          <div>
            <dt>attn_shape</dt>
            <dd class="mono">{{ formatShape(capture.probe.attn_shape) }}</dd>
          </div>
          <div>
            <dt>mlp_shape</dt>
            <dd class="mono">{{ formatShape(capture.probe.mlp_shape) }}</dd>
          </div>
          <div>
            <dt>Model layers</dt>
            <dd>{{ architectureLayers.length ? architectureLayers.length : 'not returned' }}</dd>
          </div>
          <div>
            <dt>Prompt echoed by backend</dt>
            <dd>{{ asString(capture.probe.prompt) || 'not returned' }}</dd>
          </div>
        </dl>
        <p v-if="residualConvention" class="help-copy">
          Residual convention, as returned by <code>POST /api/gpt2/architecture</code>:
          {{ residualConvention }}
        </p>
      </section>

      <p v-if="failedLayers.length" class="notice notice-warn" role="status" data-testid="activations-partial">
        <strong>Partial capture.</strong>
        {{ failedLayers.length }} of {{ columns.length }} layers returned no data, so their columns
        read “—” rather than zero. First failure: L{{ failedLayers[0].layer }} —
        {{ failedLayers[0].error }}
      </p>

      <div v-if="okLayerCount === 0" class="notice notice-unsupported" role="alert" data-testid="activations-nodata">
        <strong>NOT AVAILABLE</strong>
        <p>No layer returned activation data for this prompt, so there is no matrix to display.</p>
        <p>{{ firstFailureMessage }}</p>
      </div>

      <div v-else class="surface-grid">
        <section class="panel grid-panel" aria-labelledby="activation-grid-title">
          <header class="panel-heading">
            <div>
              <p class="panel-kicker">Token × layer</p>
              <h3 id="activation-grid-title">Captured activation magnitude</h3>
              <p class="panel-description">
                Select a cell with the keyboard or pointer to read its values in the details pane.
              </p>
            </div>
            <fieldset class="metric-set">
              <legend>Cell metric</legend>
              <label>
                <input v-model="metric" type="radio" value="resid_l2" />
                <span><code>resid_post</code> L2</span>
              </label>
              <label>
                <input v-model="metric" type="radio" value="mlp_mean" />
                <span><code>mlp_post</code> mean</span>
              </label>
            </fieldset>
          </header>

          <p v-if="tokens.length === 0" class="panel-message" role="status" data-testid="activations-empty">
            The backend returned this capture with zero token positions. An empty capture is not a
            capture of zeros, so no matrix is drawn.
          </p>

          <template v-else>
            <div class="grid-scroll">
            <table class="grid" data-testid="activations-grid">
              <caption>
                Rows are token positions in the prompt; columns are transformer layers. A cell is
                <template v-if="metric === 'resid_l2'">
                  the L2 norm of that token's <code>resid_post</code> row over the returned
                  <code>d_model</code> components, in raw activation units (no normalisation).
                </template>
                <template v-else>
                  the arithmetic mean of that token's <code>mlp_post</code> row over the returned
                  <code>d_mlp</code> components, in raw post-GELU activation units.
                </template>
                Cell values are computed in this view from the numbers the backend returned; they
                are marked as derived in the details pane.
              </caption>
              <thead>
                <tr>
                  <th scope="col" class="row-head">Token position</th>
                  <th v-for="layer in columns" :key="layer" scope="col">L{{ layer }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in pagedRows" :key="row.index">
                  <th scope="row" class="row-head">
                    <span class="token-pos">#{{ row.index }}</span>
                    <code>«{{ tokens[row.index] }}»</code>
                  </th>
                  <td v-for="(layer, columnIndex) in columns" :key="layer">
                    <button
                      type="button"
                      class="cell"
                      :class="{
                        'is-selected': isSelected(row.index, layer),
                        'is-missing': row.values[columnIndex] === null,
                      }"
                      :style="cellStyle(row.values[columnIndex])"
                      :aria-current="isSelected(row.index, layer) ? 'true' : undefined"
                      :aria-label="cellLabel(row.index, columnIndex)"
                      :data-testid="`activations-cell-${row.index}-${layer}`"
                      @click="selectCell(row.index, layer)"
                    >
                      {{ formatNumber(row.values[columnIndex], 4) }}
                    </button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <footer v-if="pageCount > 1" class="pager">
            <button class="quiet-button" type="button" data-testid="activations-page-prev" :disabled="page === 0" @click="stepPage(-1)">
              Previous tokens
            </button>
            <span class="pager-text">
              Token positions {{ pageStart }}–{{ pageEnd }} of {{ tokens.length }}
            </span>
            <button
              class="quiet-button"
              type="button"
              data-testid="activations-page-next"
              :disabled="page >= pageCount - 1"
              @click="stepPage(1)"
            >
              Next tokens
            </button>
          </footer>
          <p v-else class="panel-footnote">
            {{ tokens.length }} token position(s) returned; the whole capture fits on one page.
          </p>
          </template>
        </section>

        <section class="panel detail-panel" aria-labelledby="activation-detail-title" data-testid="activations-detail">
          <header class="panel-heading">
            <div>
              <p class="panel-kicker">Selected cell</p>
              <h3 id="activation-detail-title">
                <template v-if="selected">Token #{{ selected.token }} · Layer L{{ selected.layer }}</template>
                <template v-else>Select a cell</template>
              </h3>
            </div>
          </header>

          <p v-if="!selected" class="panel-message">
            No cell is selected. Choose a cell in the matrix to read its values, site identity and
            provenance.
          </p>

          <p v-else-if="!selectedRecord" class="panel-message panel-message-error" role="alert">
            Layer L{{ selected.layer }} returned no data in this capture, so it has no values to
            show (a failed layer is not a layer of zeros).
          </p>

          <template v-else>
            <dl class="fact-grid">
              <div>
                <dt>Token position</dt>
                <dd>{{ selected.token }} of {{ tokens.length - 1 }} (zero-based)</dd>
              </div>
              <div>
                <dt>Token text</dt>
                <dd class="mono">«{{ tokens[selected.token] }}»</dd>
              </div>
              <div>
                <dt>Site</dt>
                <dd class="mono">{{ sitePath || siteLabel || 'not returned' }}</dd>
              </div>
              <div>
                <dt>Layer</dt>
                <dd>L{{ selected.layer }}<template v-if="siteLabel"> · {{ siteLabel }}</template></dd>
              </div>
              <div>
                <dt>d_model / d_mlp</dt>
                <dd>{{ formatInteger(selectedRecord.d_model) }} / {{ formatInteger(selectedRecord.d_mlp) }}</dd>
              </div>
            </dl>

            <h4 class="detail-heading">resid_post — first {{ previewCount(selectedResid.length) }} of {{ formatInteger(selectedRecord.d_model) }} components</h4>
            <ol class="value-strip" aria-label="Returned resid_post components for the selected token, from dimension 0">
              <li v-for="(value, index) in selectedResid.slice(0, DIM_PREVIEW)" :key="`resid-${index}`">
                <span class="dim">d{{ index }}</span>
                <span class="mono">{{ formatNumber(value, 4) }}</span>
              </li>
            </ol>

            <h4 class="detail-heading">mlp_post — first {{ previewCount(selectedMlp.length) }} of {{ formatInteger(selectedRecord.d_mlp) }} components</h4>
            <ol class="value-strip" aria-label="Returned mlp_post components for the selected token, from dimension 0">
              <li v-for="(value, index) in selectedMlp.slice(0, DIM_PREVIEW)" :key="`mlp-${index}`">
                <span class="dim">d{{ index }}</span>
                <span class="mono">{{ formatNumber(value, 4) }}</span>
              </li>
            </ol>

            <h4 class="detail-heading">Derived from the returned rows <span class="derived-badge">COMPUTED IN VIEW</span></h4>
            <dl class="fact-grid">
              <div>
                <dt>resid_post L2</dt>
                <dd class="mono">{{ formatNumber(selectedResidL2, 4) }} <span class="unit">raw activation units</span></dd>
              </div>
              <div>
                <dt>mlp_post mean</dt>
                <dd class="mono">{{ formatNumber(selectedMlpMean, 6) }} <span class="unit">raw activation units</span></dd>
              </div>
            </dl>

            <h4 class="detail-heading">Returned by the backend for this layer</h4>
            <dl class="fact-grid">
              <div>
                <dt>stats.resid_l2</dt>
                <dd class="mono">{{ formatNumber(asNumber(recordStats.resid_l2), 4) }}</dd>
              </div>
              <div>
                <dt>stats.mlp_mean</dt>
                <dd class="mono">{{ formatNumber(asNumber(recordStats.mlp_mean), 6) }}</dd>
              </div>
              <div>
                <dt>stats.mlp_sparsity</dt>
                <dd class="mono">{{ formatNumber(asNumber(recordStats.mlp_sparsity), 4) }}</dd>
              </div>
            </dl>
            <p class="help-copy">
              Those three are the response's own layer-level statistics over the whole
              sequence × width matrix — not per-token values. A missing statistic renders “—”.
            </p>

            <footer class="provenance-block">
              <p><span class="source-badge">PROVENANCE</span> {{ selectedProvenance || 'not returned' }}</p>
              <p v-if="selectedFieldProvenance"><span class="source-badge">FIELDS</span> {{ selectedFieldProvenance }}</p>
              <p class="help-copy">
                Prompt echoed by this layer's response:
                “{{ asString(selectedRecord.prompt) || 'not returned' }}”.
              </p>
            </footer>
          </template>
        </section>
      </div>

      <section class="panel compare-panel" aria-labelledby="activation-compare-title" data-testid="activation-comparison">
        <header class="panel-heading"><div><p class="panel-kicker">Prompt comparison</p><h3 id="activation-compare-title">Compare layer activations</h3><p class="panel-description">Capture a second prompt on the same loaded model. Deltas are comparison prompt minus original prompt, aligned by token position and layer index.</p></div></header>
        <form class="comparison-form" @submit.prevent="runComparison"><label class="field-label" for="activation-compare-prompt">Comparison prompt</label><input id="activation-compare-prompt" v-model="comparisonPrompt" class="text-input" :disabled="isComparing" /><button class="primary-button" type="submit" :disabled="isComparing || !comparisonPrompt.trim()">{{ isComparing ? 'Capturing comparison…' : comparison ? 'Recapture comparison' : 'Capture comparison prompt' }}</button></form>
        <p v-if="comparisonError" class="notice notice-error" role="alert">{{ comparisonError }}</p>
        <p v-else-if="comparison" class="help-copy">Comparing “{{ capture.prompt }}” with “{{ comparison.prompt }}”. Token positions are aligned by index, not by matching token text; differing tokenization is shown below.</p>
        <div v-if="comparison" class="comparison-table-wrap"><table class="grid comparison-table"><thead><tr><th scope="col">Token index</th><th v-for="layer in columns" :key="layer" scope="col">L{{ layer }} Δ</th></tr></thead><tbody><tr v-for="row in comparisonGrid.slice(0, PAGE_SIZE)" :key="row.token"><th scope="row" class="row-head"><span class="token-pos">#{{ row.token }}</span><code>{{ row.text }}</code><small>vs {{ comparisonTokens[row.token] || 'token not returned' }}</small></th><td v-for="cell in row.values" :key="cell.layer"><button type="button" class="cell" :class="{ 'is-missing': cell.delta===null }" :style="cell.delta===null?{}:{background: 'color-mix(in srgb, ' + (cell.delta >= 0 ? '#0f9f75' : '#d97757') + ' ' + Math.round(8+Math.min(1,Math.abs(cell.delta)/(Math.abs(cell.base||0)+Math.abs(cell.compare||0)+1))*60) + '%, transparent)'}" :aria-label="'Token '+row.token+', layer '+cell.layer+', delta '+formatNumber(cell.delta,4)" @click="selectCell(row.token,cell.layer)">{{ formatNumber(cell.delta,4) }}</button></td></tr></tbody></table></div>
      </section>

      <section class="panel distribution-panel" aria-labelledby="activation-distribution-title" data-testid="activation-distribution">
        <header class="panel-heading"><div><p class="panel-kicker">Neuron-level statistics</p><h3 id="activation-distribution-title">MLP activation distribution</h3><p class="panel-description">Values from the selected token’s returned <code>mlp_post</code> vector. Select a bar to identify its neuron dimension.</p></div><span class="source-badge">{{ selected ? 'TOKEN #'+selected.token+' · L'+selected.layer : 'NO SELECTION' }}</span></header>
        <div v-if="!selectedRecord || !selectedMlp.length" class="panel-message">Select a captured token × layer cell with MLP values to inspect its neuron distribution.</div>
        <template v-else>
          <dl class="fact-grid distribution-stats"><div><dt>Components returned</dt><dd>{{ distributionStats.count }}</dd></div><div><dt>Mean</dt><dd class="mono">{{ formatNumber(distributionStats.mean,6) }}</dd></div><div><dt>Standard deviation</dt><dd class="mono">{{ formatNumber(distributionStats.std,6) }}</dd></div><div><dt>Minimum / maximum</dt><dd class="mono">{{ formatNumber(distributionStats.min,4) }} / {{ formatNumber(distributionStats.max,4) }}</dd></div><div><dt>Positive / zero / negative</dt><dd>{{ distributionStats.positive }} / {{ distributionStats.zero }} / {{ distributionStats.negative }}</dd></div></dl>
          <div class="neuron-bars" role="group" aria-label="MLP neuron activation dimensions"><button v-for="item in mlpDistribution" :key="item.index" type="button" class="neuron-bar" :class="{ 'neuron-bar--selected': selectedDimension===item.index }" :title="'Neuron '+item.index+': '+formatNumber(item.value,6)" :aria-label="'Select MLP neuron '+item.index+', activation '+formatNumber(item.value,6)" @click="locateNeuron(item.index)"><span class="bar-track"><i :style="{height: Math.max(2, Math.abs(item.value ?? 0)/(Math.max(Math.abs(distributionStats.min ?? 0),Math.abs(distributionStats.max ?? 0),1e-9))*100)+'%',background:(item.value ?? 0)>=0?'#2563eb':'#d97757'}"></i></span><span class="bar-index">{{ item.index }}</span></button></div>
          <div v-if="selectedDimension!==null" class="neuron-locator" role="status"><span class="source-badge">NEURON LOCATED</span><strong>Layer {{ selected.layer }} · MLP neuron {{ selectedDimension }}</strong><span>Token #{{ selected.token }} ({{ tokens[selected.token] }}) · activation {{ selectedNeuronValue===null?'not returned':formatNumber(selectedNeuronValue,6) }}</span><p class="help-copy">This identifies the activation vector dimension for the selected token and layer. It does not claim a named semantic feature unless a feature-mapping endpoint provides one.</p><div class="actions"><button type="button" class="primary-button" @click="openLinkedTool('neuroninspector')">Open neuron in workspace →</button><button type="button" class="primary-button" @click="openLinkedTool('saelab')">Inspect SAE at this layer →</button></div></div>
        </template>
      </section>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue';
import { api } from '../services/api';
import { setResearchSelection, takeResearchSelection } from '../services/researchSelectionContext';
import {
  asArray,
  asNumber,
  asString,
  describeError,
  fieldProvenanceSummary,
  formatInteger,
  formatNumber,
  formatShape,
  gpt2Activations,
  gpt2Architecture,
  gpt2LayerActivations,
  isRecord,
  isUnsupportedError,
  numberMatrix,
  provenanceOf,
  rowL2,
  statusOf,
  tokenLabels,
  type JsonRecord,
  type NumberRow,
} from '../services/interpretabilityClient';

type Metric = 'resid_l2' | 'mlp_mean';

interface LayerEntry {
  layer: number;
  record: JsonRecord | null;
  error: string;
}

interface Capture {
  prompt: string;
  probe: JsonRecord;
  architecture: JsonRecord;
  layers: LayerEntry[];
}

interface Failure {
  kind: 'unsupported' | 'error';
  message: string;
}

/** Components rendered per row in the details pane; totals are always stated. */
const DIM_PREVIEW = 8;
const PAGE_SIZE = 8;

const linkedContext = takeResearchSelection();
const prompt = ref(linkedContext?.prompt || '');
const capture = ref<Capture | null>(null);
const isCapturing = ref(false);
const progress = reactive({ done: 0, total: 0 });
const failure = ref<Failure | null>(null);
const metric = ref<Metric>('resid_l2');
const selected = ref<{ token: number; layer: number } | null>(null);
const page = ref(0);
const comparisonPrompt = ref('');
const comparison = ref<Capture | null>(null);
const isComparing = ref(false);
const comparisonError = ref('');
const selectedDimension = ref<number | null>(null);
const promptRotationNote = ref('Waiting for live GPT-2 prompt generation…');
let promptRotationTimer: ReturnType<typeof setInterval> | null = null;
let promptRotationBusy = false;


/** A newer capture invalidates the older loop's remaining requests. */
let captureSequence = 0;

const captureButtonLabel = computed(() => {
  if (isCapturing.value) return `Capturing ${progress.done}/${progress.total}…`;
  return capture.value ? 'Recapture' : 'Capture activations';
});

const stale = computed(() => Boolean(capture.value) && capture.value?.prompt !== prompt.value.trim());

const progressPercent = computed(() =>
  progress.total > 0 ? `${Math.round((progress.done / progress.total) * 100)}%` : '0%',
);

const architectureLayers = computed<JsonRecord[]>(() =>
  capture.value ? asArray(capture.value.architecture.layers).filter(isRecord) : [],
);

const residualConvention = computed(() => {
  for (const layer of architectureLayers.value) {
    const value = asString(layer.residual_convention);
    if (value) return value;
  }
  return '';
});

const probeLayer = computed(() => asNumber(capture.value?.probe.layer));
const probeProvenance = computed(() => provenanceOf(capture.value?.probe ?? null));

const columns = computed<number[]>(() => (capture.value?.layers ?? []).map((entry) => entry.layer));

const comparisonTokens = computed<string[]>(() => {
  for (const entry of comparison.value?.layers ?? []) {
    if (!entry.record || statusOf(entry.record) !== 'ok') continue;
    const labels = tokenLabels(entry.record);
    if (labels.length) return labels;
  }
  return [];
});

const tokens = computed<string[]>(() => {
  for (const entry of capture.value?.layers ?? []) {
    if (!entry.record || statusOf(entry.record) !== 'ok') continue;
    const labels = tokenLabels(entry.record);
    if (labels.length) return labels;
  }
  return [];
});

const failedLayers = computed(() => (capture.value?.layers ?? []).filter((entry) => !entry.record));

const okLayerCount = computed(
  () => (capture.value?.layers ?? []).filter((entry) => entry.record && statusOf(entry.record) === 'ok').length,
);

const firstFailureMessage = computed(() => {
  const first = failedLayers.value[0];
  if (!first) return 'The backend returned no layer data.';
  return `L${first.layer}: ${first.error || 'the backend returned no reason'}`;
});

const grid = computed<(number | null)[][]>(() => {
  const cols = columns.value;
  const rows = tokens.value.map(() => cols.map(() => null as number | null));
  const key = metric.value === 'resid_l2' ? 'resid_post' : 'mlp_post';
  for (let columnIndex = 0; columnIndex < cols.length; columnIndex += 1) {
    const entry = capture.value?.layers[columnIndex];
    if (!entry?.record || statusOf(entry.record) !== 'ok') continue;
    const matrix = numberMatrix(entry.record[key]);
    for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
      const row = matrix[rowIndex];
      if (!row || row.length === 0) continue;
      rows[rowIndex][columnIndex] = metric.value === 'resid_l2' ? rowL2(row) : meanOf(row);
    }
  }
  return rows;
});

const gridMax = computed(() => {
  let max = 0;
  for (const row of grid.value) {
    for (const value of row) {
      if (typeof value === 'number' && Math.abs(value) > max) max = Math.abs(value);
    }
  }
  return max;
});


const selectedNeuronValue = computed(() => selectedDimension.value === null ? null : selectedMlp.value[selectedDimension.value] ?? null);
const mlpDistribution = computed(() => selectedMlp.value.map((value, index) => ({ index, value })).filter(item => item.value !== null).slice(0, 64));
const distributionStats = computed(() => {
  const values = selectedMlp.value.filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  if (!values.length) return { count: 0, mean: null as number | null, min: null as number | null, max: null as number | null, std: null as number | null, positive: 0, zero: 0, negative: 0 };
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return { count: values.length, mean, min: Math.min(...values), max: Math.max(...values), std: Math.sqrt(variance), positive: values.filter(v => v > 0).length, zero: values.filter(v => v === 0).length, negative: values.filter(v => v < 0).length };
});
const comparisonGrid = computed(() => {
  if (!capture.value || !comparison.value) return [] as Array<{token:number; text:string; values:Array<{layer:number; base:number|null; compare:number|null; delta:number|null}>}>;
  const baseTokens = tokens.value;
  return baseTokens.map((text, token) => ({ token, text, values: columns.value.map(layer => {
    const baseRecord = capture.value!.layers.find(entry=>entry.layer===layer)?.record ?? null;
    const compareRecord = comparison.value!.layers.find(entry=>entry.layer===layer)?.record ?? null;
    const base = metricValue(baseRecord, token, metric.value);
    const other = metricValue(compareRecord, token, metric.value);
    return { layer, base, compare: other, delta: base !== null && other !== null ? other-base : null };
  }).filter(item=>item.layer!==undefined) }));
});

const pageCount = computed(() => Math.max(1, Math.ceil(tokens.value.length / PAGE_SIZE)));
const pageStart = computed(() => (tokens.value.length === 0 ? 0 : page.value * PAGE_SIZE));
const pageEnd = computed(() => Math.min(tokens.value.length - 1, pageStart.value + PAGE_SIZE - 1));

const pagedRows = computed(() =>
  grid.value
    .map((values, index) => ({ index, values }))
    .slice(pageStart.value, pageStart.value + PAGE_SIZE),
);

const selectedEntry = computed<LayerEntry | null>(() => {
  if (!selected.value || !capture.value) return null;
  return capture.value.layers.find((entry) => entry.layer === selected.value?.layer) ?? null;
});

const selectedRecord = computed<JsonRecord | null>(() => {
  const record = selectedEntry.value?.record ?? null;
  return record && statusOf(record) === 'ok' ? record : null;
});

const selectedResid = computed<NumberRow>(() => {
  if (!selectedRecord.value || !selected.value) return [];
  const matrix = numberMatrix(selectedRecord.value.resid_post);
  return matrix[selected.value.token] ?? [];
});

const selectedMlp = computed<NumberRow>(() => {
  if (!selectedRecord.value || !selected.value) return [];
  const matrix = numberMatrix(selectedRecord.value.mlp_post);
  return matrix[selected.value.token] ?? [];
});

const selectedResidL2 = computed(() => rowL2(selectedResid.value));
const selectedMlpMean = computed(() => meanOf(selectedMlp.value));

const recordStats = computed<JsonRecord>(() =>
  selectedRecord.value && isRecord(selectedRecord.value.stats) ? selectedRecord.value.stats : {},
);

const selectedProvenance = computed(() => provenanceOf(selectedRecord.value));
const selectedFieldProvenance = computed(() => fieldProvenanceSummary(selectedRecord.value));

const selectedSiteLayer = computed<JsonRecord | null>(() => {
  if (!selected.value) return null;
  return architectureLayers.value.find((layer) => asNumber(layer.layer_index) === selected.value?.layer) ?? null;
});

const siteLabel = computed(() => (selectedSiteLayer.value ? asString(selectedSiteLayer.value.label) : ''));
const sitePath = computed(() => {
  const layer = selectedSiteLayer.value;
  if (!layer) return '';
  const components = asArray(layer.components).filter(isRecord);
  const mlp = components.find((component) => asString(component.id) === 'mlp');
  return mlp ? asString(mlp.path) : asString(layer.path);
});

function openLinkedTool(route:string) { setResearchSelection({prompt:capture.value?.prompt || prompt.value.trim(),layer:selected.value?.layer,token:selected.value?.token,neuron:selectedDimension.value ?? undefined,origin:'activation-browser'}); window.location.hash=route; }

async function fetchGeneratedPrompt(): Promise<string> {
  const response = await api.gpt2FreshPrompt();
  const generated = isRecord(response) ? asString(response.prompt)?.trim() : undefined;
  if (!isRecord(response) || response.status !== 'ok' || response.provenance !== 'live' || response.attested !== true || !generated) {
    throw new Error(isRecord(response) ? asString(response.error) || 'GPT-2 did not return an attested, live prompt.' : 'GPT-2 returned an unreadable response.');
  }
  return generated;
}

async function rotatePrompts(): Promise<void> {
  if (promptRotationBusy) return;
  promptRotationBusy = true;
  const previous = [prompt.value.trim(), comparisonPrompt.value.trim()].filter(Boolean);
  try {
    let nextPrompt = await fetchGeneratedPrompt();
    for (let attempt = 0; previous.includes(nextPrompt) && attempt < 3; attempt += 1) nextPrompt = await fetchGeneratedPrompt();
    let nextComparison = await fetchGeneratedPrompt();
    for (let attempt = 0; (nextComparison === nextPrompt || previous.includes(nextComparison)) && attempt < 3; attempt += 1) nextComparison = await fetchGeneratedPrompt();
    if (previous.includes(nextPrompt) || nextComparison === nextPrompt || previous.includes(nextComparison)) {
      promptRotationNote.value = 'GPT-2 did not produce two distinct new prompts; current inputs were kept.';
      return;
    }
    prompt.value = nextPrompt;
    comparisonPrompt.value = nextComparison;
    promptRotationNote.value = 'Both inputs were generated by live GPT-2 · ' + new Date().toLocaleTimeString();
  } catch (error) {
    promptRotationNote.value = 'Live GPT-2 prompt generation unavailable: ' + describeError(error) + '. No canned input was substituted.';
  } finally {
    promptRotationBusy = false;
  }
}

onMounted(() => {
  void rotatePrompts();
  promptRotationTimer = setInterval(() => { void rotatePrompts(); }, 120_000);
});
onBeforeUnmount(() => {
  if (promptRotationTimer !== null) clearInterval(promptRotationTimer);
});

async function runCapture() {
  const targetPrompt = prompt.value.trim();
  if (!targetPrompt) return;
  const sequence = (captureSequence += 1);
  isCapturing.value = true;
  failure.value = null;
  capture.value = null;
  selected.value = null;
  page.value = 0;
  progress.done = 0;
  progress.total = 0;
  try {
    const probe = await gpt2Activations({ prompt: targetPrompt, layer: 0 });
    if (sequence !== captureSequence) return;
    if (statusOf(probe) !== 'ok') {
      failure.value = failureFromRecord(probe);
      return;
    }

    const architecture = await gpt2Architecture();
    if (sequence !== captureSequence) return;
    const layerCount = asArray(architecture.layers).filter(isRecord).length;
    if (statusOf(architecture) !== 'ok' || layerCount === 0) {
      failure.value = {
        kind: statusOf(architecture) === 'unavailable' ? 'unsupported' : 'error',
        message:
          `The layer axis is unknown: POST /api/gpt2/architecture returned ` +
          `${statusOf(architecture) || 'no status'}. ${describeStatusRecord(architecture)}`,
      };
      return;
    }

    progress.total = layerCount;
    const layers: LayerEntry[] = [];
    for (let layer = 0; layer < layerCount; layer += 1) {
      try {
        const record = await gpt2LayerActivations({ prompt: targetPrompt, layer });
        if (sequence !== captureSequence) return;
        if (statusOf(record) === 'ok') {
          layers.push({ layer, record, error: '' });
        } else {
          layers.push({ layer, record: null, error: describeStatusRecord(record) });
        }
      } catch (error) {
        if (sequence !== captureSequence) return;
        layers.push({ layer, record: null, error: describeError(error) });
      }
      progress.done = layer + 1;
    }

    capture.value = { prompt: targetPrompt, probe, architecture, layers };
    const firstOk = layers.find((entry) => entry.record);
    const firstTokenCount = firstOk?.record ? tokenLabels(firstOk.record).length : 0;
    const requestedLayer = linkedContext?.layer;
    const chosenLayer = requestedLayer != null && layers.some((entry) => entry.layer === requestedLayer && entry.record) ? requestedLayer : firstOk?.layer;
    const tokenCount = chosenLayer == null ? 0 : tokenLabels(layers.find((entry) => entry.layer === chosenLayer)?.record || {}).length;
    const requestedToken = linkedContext?.token ?? 0;
    selected.value = chosenLayer != null && tokenCount > 0 ? { token: Math.max(0, Math.min(requestedToken, tokenCount - 1)), layer: chosenLayer } : firstOk && firstTokenCount > 0 ? { token: 0, layer: firstOk.layer } : null;
    page.value = selected.value ? Math.floor(selected.value.token / PAGE_SIZE) : 0;
  } catch (error) {
    if (sequence !== captureSequence) return;
    failure.value = {
      kind: isUnsupportedError(error) ? 'unsupported' : 'error',
      message: describeError(error),
    };
  } finally {
    if (sequence === captureSequence) isCapturing.value = false;
  }
}

function failureFromRecord(record: JsonRecord): Failure {
  return {
    kind: statusOf(record) === 'unavailable' ? 'unsupported' : 'error',
    message: describeStatusRecord(record),
  };
}

function describeStatusRecord(record: JsonRecord): string {
  return (
    asString(record.error) ||
    asString(record.reason) ||
    `The backend returned status “${statusOf(record) || 'none'}” without a reason.`
  );
}

function metricValue(record: JsonRecord | null, token: number, selectedMetric: Metric): number | null {
  if (!record || statusOf(record)!=='ok') return null;
  const matrix=numberMatrix(record[selectedMetric==='resid_l2'?'resid_post':'mlp_post']);
  const row=matrix[token];
  if(!row?.length) return null;
  return selectedMetric==='resid_l2'?rowL2(row):meanOf(row);
}

async function runComparison() {
  const target=comparisonPrompt.value.trim();
  if(!target || isComparing.value || !capture.value) return;
  isComparing.value=true; comparisonError.value=''; comparison.value=null;
  try {
    const probe=await gpt2Activations({prompt:target,layer:0});
    if(statusOf(probe)!=='ok') throw new Error(describeStatusRecord(probe));
    const architecture=await gpt2Architecture();
    const count=asArray(architecture.layers).filter(isRecord).length;
    if(statusOf(architecture)!=='ok'||!count) throw new Error('The backend did not return the model layer list for comparison.');
    const layers:LayerEntry[]=[];
    for(let layer=0;layer<count;layer+=1) {
      try { const record=await gpt2LayerActivations({prompt:target,layer}); layers.push(statusOf(record)==='ok'?{layer,record,error:''}:{layer,record:null,error:describeStatusRecord(record)}); }
      catch(error) { layers.push({layer,record:null,error:describeError(error)}); }
    }
    comparison.value={prompt:target,probe,architecture,layers};
  } catch(error) { comparisonError.value=describeError(error); }
  finally { isComparing.value=false; }
}

function locateNeuron(dimension:number) {
  selectedDimension.value=dimension;
  if (selected.value) {
    // Keep the exact token and layer selected while identifying the MLP dimension.
    selected.value={...selected.value};
  }
}

function meanOf(row: NumberRow): number | null {
  if (!row.length) return null;
  let total = 0;
  for (const cell of row) {
    if (typeof cell !== 'number') return null;
    total += cell;
  }
  return total / row.length;
}

function stepPage(delta: number) {
  page.value = Math.min(Math.max(0, page.value + delta), pageCount.value - 1);
}

function selectCell(token: number, layer: number) {
  // The details pane resolves the layer from the capture, so a failed layer
  // renders "no data" instead of whatever layer was selected before it.
  selected.value = { token, layer };
  selectedDimension.value = null;
}

function isSelected(token: number, layer: number): boolean {
  return selected.value?.token === token && selected.value?.layer === layer;
}

function cellStyle(value: number | null): Record<string, string> {
  if (typeof value !== 'number' || gridMax.value <= 0) return {};
  const intensity = Math.min(1, Math.abs(value) / gridMax.value);
  return { background: `color-mix(in srgb, var(--primary) ${Math.round(8 + intensity * 52)}%, transparent)` };
}

function cellLabel(rowIndex: number, columnIndex: number): string {
  const layer = columns.value[columnIndex];
  const value = grid.value[rowIndex]?.[columnIndex] ?? null;
  const token = tokens.value[rowIndex] ?? '';
  const metricLabel = metric.value === 'resid_l2' ? 'resid_post L2' : 'mlp_post mean';
  const valueLabel = value === null ? 'no value returned' : formatNumber(value, 4);
  return `Token position ${rowIndex}, “${token}”, layer ${layer}: ${metricLabel} ${valueLabel}`;
}

function previewCount(available: number): number {
  return Math.min(DIM_PREVIEW, available);
}
</script>

<style scoped>
.activation-surface {
  min-height: 100%;
  min-width: 0;
  padding: clamp(14px, 2vw, 24px);
  border-radius: var(--radius-lg);
  background: var(--bg);
  color: var(--text);
  color-scheme: light;
  font-family: var(--font);
}

.surface-header,
.panel-heading,
.capture-form,
.pager,
.provenance-block p {
  display: flex;
  align-items: center;
  gap: 10px;
}

.surface-header {
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
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
.help-copy {
  margin: 7px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.help-copy {
  margin-top: 12px;
  font-size: 11px;
}

.capture-form {
  flex: 0 1 420px;
  align-items: flex-end;
  flex-wrap: wrap;
}

.field-label {
  width: 100%;
  color: var(--text-muted);
  font: 700 10px/1 var(--font-mono);
  letter-spacing: 0.06em;
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

.primary-button,
.quiet-button {
  min-height: 32px;
  padding: 0 11px;
  border: 1px solid var(--primary);
  border-radius: 6px;
  background: var(--primary);
  color: #ffffff;
  cursor: pointer;
  font: 700 11px/1 var(--font);
}

.quiet-button {
  border-color: var(--border);
  background: var(--surface);
  color: var(--text-muted);
}

.quiet-button:hover:not(:disabled) {
  color: var(--text);
  border-color: var(--border-light);
}

.primary-button:disabled,
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

.progress-track {
  height: 4px;
  margin-top: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--border);
}

.progress-fill {
  display: block;
  height: 100%;
  background: var(--primary);
  transition: width 120ms linear;
}

.panel {
  margin-bottom: 14px;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.panel-heading {
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 12px;
}

.panel-heading h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 700;
}

.source-badge,
.derived-badge {
  display: inline-block;
  padding: 3px 6px;
  border: 1px solid var(--border-light);
  border-radius: 999px;
  color: var(--text-muted);
  font: 700 9px/1 var(--font-mono);
  letter-spacing: 0.06em;
  white-space: nowrap;
}

.derived-badge {
  margin-left: 6px;
  border-style: dashed;
  vertical-align: middle;
}

.fact-grid {
  display: grid;
  gap: 8px 16px;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  margin: 0;
}

.fact-grid dt {
  color: var(--text-muted);
  font: 700 9px/1.4 var(--font-mono);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.fact-grid dd {
  margin: 2px 0 0;
  font-size: 12px;
  line-height: 1.45;
  overflow-wrap: anywhere;
}

.mono {
  font-family: var(--font-mono);
}

.unit {
  color: var(--text-muted);
  font-size: 10px;
}

.surface-grid {
  display: flex;
  align-items: stretch;
  gap: 14px;
  flex-wrap: wrap;
}

.surface-grid > .panel {
  flex: 1 1 420px;
  min-width: 0;
}

.grid-scroll {
  overflow-x: auto;
}

.grid {
  width: 100%;
  border-collapse: collapse;
  font: 500 11px/1.2 var(--font-mono);
}

.grid caption {
  caption-side: top;
  padding-bottom: 8px;
  color: var(--text-muted);
  font: 400 11px/1.5 var(--font);
  text-align: left;
}

.grid th,
.grid td {
  padding: 3px;
  text-align: center;
}

.grid thead th {
  color: var(--text-muted);
  font-size: 10px;
  letter-spacing: 0.04em;
}

.row-head {
  text-align: left;
  white-space: nowrap;
}

.token-pos {
  color: var(--text-muted);
  margin-right: 5px;
}

.cell {
  display: block;
  width: 100%;
  min-width: 66px;
  min-height: 30px;
  padding: 0 4px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--surface);
  color: var(--text);
  cursor: pointer;
  font: 500 11px/1 var(--font-mono);
}

.cell:hover {
  border-color: var(--primary);
}

.cell:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 1px;
}

.cell.is-selected {
  border-color: var(--primary);
  box-shadow: inset 0 0 0 2px var(--primary);
}

.cell.is-missing {
  color: var(--text-muted);
  border-style: dashed;
  cursor: default;
}

.metric-set {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin: 0;
  padding: 0;
  border: 0;
}

.metric-set legend {
  width: 100%;
  padding: 0;
  color: var(--text-muted);
  font: 700 9px/1.4 var(--font-mono);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.metric-set label {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--text-dim);
}

.metric-set input {
  accent-color: var(--primary);
}

.detail-panel {
  min-width: 0;
}

.detail-heading {
  margin: 16px 0 6px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.value-strip {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin: 0;
  padding: 0;
  list-style: none;
}

.value-strip li {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 5px 7px;
  border: 1px solid var(--border);
  border-radius: 5px;
  background: var(--surface-2);
  font-size: 11px;
}

.value-strip .dim {
  color: var(--text-muted);
  font: 700 9px/1 var(--font-mono);
}

.provenance-block {
  margin-top: 16px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}

.provenance-block p {
  margin: 0 0 5px;
  align-items: flex-start;
  flex-wrap: wrap;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.5;
  overflow-wrap: anywhere;
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

.panel-message-error {
  border-style: solid;
  color: var(--danger);
}

.pager {
  justify-content: space-between;
  margin-top: 10px;
}

.pager-text {
  color: var(--text-muted);
  font-size: 11px;
}

.panel-footnote {
  margin: 10px 0 0;
  color: var(--text-muted);
  font-size: 10px;
}

@media (max-width: 760px) {
  .surface-grid {
    flex-direction: column;
  }

  .surface-grid > .panel {
    flex: auto;
  }
}

@media (prefers-reduced-motion: reduce) {
  .progress-fill {
    transition: none;
  }
}

/* Shared MECH interaction contract: visible keyboard focus and restrained motion. */
.activation-surface :is(button, input, select, textarea, a, [tabindex]):focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}
.activation-surface button,
.activation-surface input,
.activation-surface select {
  transition: background-color 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
}
@media (prefers-reduced-motion: reduce) {
  .activation-surface *,
  .activation-surface *::before,
  .activation-surface *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }
}


.comparison-form{display:flex;align-items:flex-end;gap:9px;flex-wrap:wrap;padding:12px;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface-2)}.comparison-form .field-label{display:grid;flex:1 1 260px;gap:5px;color:var(--text-muted);font-size:10px}.comparison-form .text-input{min-height:34px;padding:0 10px;border:1px solid var(--border);border-radius:5px;background:var(--surface);color:var(--text)}.comparison-table-wrap{max-width:100%;overflow:auto}.comparison-table{min-width:540px}.comparison-table .row-head small{display:block;color:var(--text-muted);font-size:9px;font-weight:400}.distribution-stats{grid-template-columns:repeat(auto-fit,minmax(120px,1fr))}.neuron-bars{display:flex;align-items:stretch;gap:3px;min-height:130px;max-height:180px;padding:10px 6px 4px;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface-2);overflow-x:auto}.neuron-bar{display:flex;flex-direction:column;align-items:center;gap:4px;flex:0 0 22px;min-width:22px;padding:0 1px;border:1px solid transparent;border-radius:4px;background:transparent}.neuron-bar:hover,.neuron-bar--selected{border-color:var(--primary);background:var(--accent-soft)}.bar-track{display:flex;align-items:flex-end;justify-content:center;width:100%;height:112px;border-bottom:1px solid var(--border)}.bar-track i{display:block;width:9px;min-height:2px;border-radius:3px 3px 0 0}.bar-index{color:var(--text-muted);font:9px var(--font-mono)}.neuron-locator{display:grid;gap:6px;margin-top:12px;padding:12px;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface)}.neuron-locator strong{font-size:13px}.neuron-locator>span:not(.source-badge){color:var(--text-muted);font-size:11px}.neuron-locator p{margin:0}


.comparison-form{display:flex;align-items:flex-end;gap:9px;flex-wrap:wrap;padding:12px;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface-2)}.comparison-form .field-label{display:grid;flex:1 1 260px;gap:5px;color:var(--text-muted);font-size:10px}.comparison-form .text-input{min-height:34px;padding:0 10px;border:1px solid var(--border);border-radius:5px;background:var(--surface);color:var(--text)}.comparison-table-wrap{max-width:100%;overflow:auto}.comparison-table{min-width:540px}.comparison-table .row-head small{display:block;color:var(--text-muted);font-size:9px;font-weight:400}.distribution-stats{grid-template-columns:repeat(auto-fit,minmax(120px,1fr))}.neuron-bars{display:flex;align-items:stretch;gap:3px;min-height:130px;max-height:180px;padding:10px 6px 4px;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface-2);overflow-x:auto}.neuron-bar{display:flex;flex-direction:column;align-items:center;gap:4px;flex:0 0 22px;min-width:22px;padding:0 1px;border:1px solid transparent;border-radius:4px;background:transparent}.neuron-bar:hover,.neuron-bar--selected{border-color:var(--primary);background:var(--accent-soft)}.bar-track{display:flex;align-items:flex-end;justify-content:center;width:100%;height:112px;border-bottom:1px solid var(--border)}.bar-track i{display:block;width:9px;min-height:2px;border-radius:3px 3px 0 0}.bar-index{color:var(--text-muted);font:9px var(--font-mono)}.neuron-locator{display:grid;gap:6px;margin-top:12px;padding:12px;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface)}.neuron-locator strong{font-size:13px}.neuron-locator>span:not(.source-badge){color:var(--text-muted);font-size:11px}.neuron-locator p{margin:0}
</style>
