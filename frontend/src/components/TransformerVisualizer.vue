<template>
  <section
    class="transformer-visualizer"
    aria-labelledby="transformer-visualizer-title"
    :aria-busy="isBusy"
  >
    <header class="visualizer-header">
      <div class="title-lockup">
        <span class="title-mark" aria-hidden="true">
          <Network :size="20" :stroke-width="1.8" />
        </span>
        <div>
          <div class="title-kicker">MECH / Mechanistic interpretability</div>
          <h1 id="transformer-visualizer-title">Transformer Visualizer</h1>
          <p>Follow the returned GPT-2 architecture, then inspect layer and attention-head data.</p>
        </div>
      </div>

      <div class="header-actions">
        <div class="runtime-chip" :class="`runtime-chip--${runtimeTone}`" role="status" aria-live="polite">
          <span class="runtime-chip__dot" aria-hidden="true" />
          {{ runtimeLabel }}
        </div>
        <button
          class="refresh-button"
          type="button"
          :disabled="architectureState === 'loading' || isOffline"
          aria-label="Refresh transformer architecture"
          @click="loadArchitecture()"
        >
          <RefreshCw :class="{ spinning: architectureState === 'loading' }" :size="15" aria-hidden="true" />
          Refresh
        </button>
      </div>
    </header>

    <div
      v-if="statusMessage"
      class="state-banner"
      :class="`state-banner--${statusTone}`"
      :role="statusTone === 'error' ? 'alert' : 'status'"
    >
      <WifiOff v-if="isOffline" :size="17" aria-hidden="true" />
      <AlertTriangle v-else :size="17" aria-hidden="true" />
      <div>
        <strong>{{ statusTitle }}</strong>
        <span>{{ statusMessage }}</span>
      </div>
      <button
        v-if="!isOffline"
        class="state-banner__action"
        type="button"
        :disabled="architectureState === 'loading'"
        @click="loadArchitecture()"
      >
        Retry
      </button>
    </div>

    <section v-if="architectureState === 'loading' && !architecture" class="loading-panel" aria-live="polite">
      <div class="loading-panel__mark" aria-hidden="true">
        <RefreshCw class="spinning" :size="20" />
      </div>
      <div>
        <strong>Reading architecture metadata</strong>
        <span>Requesting the current model topology from the MECH runtime…</span>
      </div>
      <div class="skeleton-grid" aria-hidden="true">
        <span v-for="item in 6" :key="item" />
      </div>
    </section>

    <section
      v-else-if="architectureState === 'empty'"
      class="message-panel"
      aria-labelledby="architecture-empty-title"
    >
      <span class="message-panel__mark" aria-hidden="true"><Box :size="22" /></span>
      <div>
        <span class="section-kicker">No topology</span>
        <h2 id="architecture-empty-title">No transformer layers were returned.</h2>
        <p>The runtime responded, but its architecture payload did not contain a usable layer list.</p>
      </div>
      <button type="button" :disabled="isOffline" @click="loadArchitecture()">Request again</button>
    </section>

    <section
      v-else-if="architectureState === 'error' && !architecture"
      class="message-panel message-panel--error"
      aria-labelledby="architecture-error-title"
    >
      <span class="message-panel__mark" aria-hidden="true"><AlertTriangle :size="22" /></span>
      <div>
        <span class="section-kicker">Architecture unavailable</span>
        <h2 id="architecture-error-title">The runtime did not return usable architecture data.</h2>
        <p>{{ architectureError || 'No readable error was provided.' }}</p>
      </div>
      <button type="button" :disabled="isOffline" @click="loadArchitecture()">Try again</button>
    </section>

    <template v-else-if="architecture">
      <form class="prompt-command" novalidate @submit.prevent="runPrompt()">
        <div class="prompt-command__label">
          <span class="section-kicker">Prompt workbench</span>
          <label for="transformer-prompt">Run a prompt to request cached attention data</label>
        </div>
        <textarea
          id="transformer-prompt"
          v-model="prompt"
          rows="2"
          maxlength="256"
          placeholder="Enter a short prompt…"
          :disabled="promptState === 'loading' || isOffline"
          :aria-invalid="Boolean(promptError)"
          :aria-describedby="promptError ? 'transformer-prompt-error' : 'transformer-prompt-help'"
          @keydown="onPromptKeydown"
        />
        <div class="prompt-command__actions">
          <span id="transformer-prompt-help">Ctrl / ⌘ + Enter to run</span>
          <button
            class="run-button"
            type="submit"
            data-testid="transformer-run"
            :disabled="!canRunPrompt"
          >
            <RefreshCw v-if="promptState === 'loading'" class="spinning" :size="15" aria-hidden="true" />
            <Play v-else :size="15" aria-hidden="true" />
            {{ promptState === 'loading' ? 'Running…' : 'Run prompt' }}
          </button>
        </div>
        <p v-if="promptError" id="transformer-prompt-error" class="inline-error" role="alert" data-testid="transformer-prompt-error">{{ promptError }}</p>
      </form>

      <section class="overview-panel" aria-labelledby="architecture-overview-title">
        <header class="panel-header">
          <div>
            <span class="section-kicker">Semantic architecture</span>
            <h2 id="architecture-overview-title">Residual-stream topology</h2>
          </div>
          <div class="panel-header__meta">
            <span
              v-if="lastUpdatedLabel"
              class="received-at"
              :title="lastUpdatedLabel"
            >
              Received {{ lastUpdatedLabel }}
            </span>
            <span
              class="provenance-badge"
              :class="`provenance-badge--${architectureProvenance}`"
              :title="provenanceDescription(architectureProvenance, architecture)"
            >
              <span class="provenance-badge__dot" aria-hidden="true" />
              {{ architectureProvenance }}
            </span>
          </div>
        </header>

        <ol class="architecture-flow" aria-label="Transformer data flow">
          <li class="architecture-node">
            <span class="architecture-node__index">01</span>
            <span class="architecture-node__icon" aria-hidden="true"><Database :size="18" /></span>
            <div class="architecture-node__body">
              <strong>Input embeddings</strong>
              <span>{{ moduleLabel('wte', 'Token embedding') }}</span>
              <code>{{ moduleShape('wte') }}</code>
              <span>{{ moduleLabel('wpe', 'Position embedding') }}</span>
              <code>{{ moduleShape('wpe') }}</code>
            </div>
          </li>
          <li class="flow-arrow" aria-hidden="true"><ChevronRight :size="17" /></li>
          <li class="architecture-node architecture-node--selected">
            <span class="architecture-node__index">02</span>
            <span class="architecture-node__icon" aria-hidden="true"><Layers :size="18" /></span>
            <div class="architecture-node__body">
              <strong>Transformer block {{ selectedLayer ?? '—' }}</strong>
              <span>{{ selectedArchitectureLayer?.label || selectedArchitectureLayer?.path || 'Selected block' }}</span>
              <div class="component-chips" aria-label="Returned block components">
                <span v-for="component in layerComponents" :key="componentKey(component)">
                  {{ componentLabel(component) }}
                </span>
                <span v-if="!layerComponents.length">Component detail unavailable</span>
              </div>
            </div>
          </li>
          <li class="flow-arrow" aria-hidden="true"><ChevronRight :size="17" /></li>
          <li class="architecture-node">
            <span class="architecture-node__index">03</span>
            <span class="architecture-node__icon" aria-hidden="true"><Braces :size="18" /></span>
            <div class="architecture-node__body">
              <strong>Readout</strong>
              <span>{{ moduleLabel('ln_f', 'Final normalization') }}</span>
              <span>{{ moduleLabel('lm_head', 'Language-model head') }}</span>
              <code>{{ formatInteger(architectureValue('vocab_size', 'vocabulary_size')) }} vocabulary entries</code>
            </div>
          </li>
        </ol>

        <div class="residual-strip">
          <span>Residual stream</span>
          <code>d_model {{ formatInteger(architectureValue('d_model', 'hidden_size')) }}</code>
          <span v-if="architectureValue('activation')">Block activation {{ textValue(architectureValue('activation')) }}</span>
        </div>

        <dl v-if="architectureStats.length" class="architecture-stats" data-testid="transformer-architecture">
          <div v-for="stat in architectureStats" :key="stat.label" class="architecture-stat">
            <dt>{{ stat.label }}</dt>
            <dd>{{ stat.value }}</dd>
          </div>
        </dl>
        <p class="provenance-note">
          {{ provenanceDescription(architectureProvenance, architecture) }}
        </p>
      </section>

      <div class="workspace-grid">
        <aside class="panel layer-panel" aria-labelledby="layer-list-title">
          <header class="panel-header panel-header--compact">
            <div>
              <span class="section-kicker">Block navigator</span>
              <h2 id="layer-list-title">Layers</h2>
            </div>
            <span class="count-badge">{{ layerOptions.length }}</span>
          </header>

          <div v-if="layerOptions.length" class="layer-list" data-testid="transformer-layers" aria-label="Transformer layers">
            <button
              v-for="layer in layerOptions"
              :key="layer"
              class="layer-option"
              :class="{ 'layer-option--selected': selectedLayer === layer }"
              type="button"
              :aria-pressed="selectedLayer === layer"
              @click="selectLayer(layer)"
            >
              <span class="layer-option__index">L{{ layer }}</span>
              <span class="layer-option__path">
                {{ architectureLayerRecord(layer)?.label || `transformer.h.${layer}` }}
              </span>
            </button>
          </div>
          <div v-else class="compact-empty">No layer indices were returned.</div>

          <div class="layer-source">
            <span>Source topology</span>
            <code>{{ architectureValue('model_name', 'model') || 'model unavailable' }}</code>
          </div>
        </aside>

        <div class="inspector-column">
          <section class="panel layer-detail" aria-labelledby="layer-detail-title">
            <header class="panel-header">
              <div>
                <span class="section-kicker">Layer inspector</span>
                <h2 id="layer-detail-title">
                  {{ selectedLayer === null ? 'Select a layer' : `Layer ${selectedLayer}` }}
                </h2>
              </div>
              <span
                class="provenance-badge"
                :class="`provenance-badge--${layerProvenance}`"
                :title="provenanceDescription(layerProvenance, layerDetail)"
              >
                <span class="provenance-badge__dot" aria-hidden="true" />
                {{ layerProvenance }}
              </span>
            </header>

            <div v-if="layerState === 'loading'" class="inline-loading" role="status">
              <RefreshCw class="spinning" :size="17" aria-hidden="true" />
              Loading returned layer metadata…
            </div>
            <div v-else-if="layerError" class="inline-message inline-message--error" role="alert">
              <AlertTriangle :size="17" aria-hidden="true" />
              <div><strong>Layer inspection failed.</strong><span>{{ layerError }}</span></div>
            </div>
            <template v-else-if="layerDetail">
              <div class="layer-path-row">
                <code>{{ textValue(layerDetail.path) || textValue(layerDetail.label) || `transformer.h.${selectedLayer}` }}</code>
                <span>{{ layerComponents.length }} components returned in architecture</span>
              </div>

              <ol class="component-flow" aria-label="Returned layer component order">
                <li v-for="(component, index) in layerComponents" :key="componentKey(component)">
                  <span>{{ componentLabel(component) }}</span>
                  <code v-if="componentPath(component)">{{ componentPath(component) }}</code>
                </li>
                <li v-if="!layerComponents.length" class="component-flow__empty">
                  <span>Component order not returned</span>
                </li>
              </ol>

              <dl v-if="layerStats.length" class="detail-stats">
                <div v-for="row in layerStats" :key="row.label">
                  <dt>{{ row.label }}</dt>
                  <dd>{{ row.value }}</dd>
                </div>
              </dl>

              <div class="head-selector-heading">
                <div>
                  <span class="section-kicker">Attention heads</span>
                  <strong>Returned layer summaries</strong>
                </div>
                <span>{{ headOptions.length }} selectable</span>
              </div>

              <div v-if="headOptions.length" class="head-grid" aria-label="Attention heads">
                <button
                  v-for="head in headOptions"
                  :key="head"
                  class="head-option"
                  :class="{ 'head-option--selected': selectedHead === head }"
                  type="button"
                  :aria-pressed="selectedHead === head"
                  @click="selectHead(head)"
                >
                  <span class="head-option__label">H{{ head }}</span>
                  <span v-if="numberValue(headSummary(head)?.q_weight_l2) !== undefined" class="head-option__metric">
                    Q {{ formatMetric(numberValue(headSummary(head)?.q_weight_l2)) }}
                  </span>
                  <span v-else class="head-option__metric head-option__metric--empty">Pattern only</span>
                </button>
              </div>
              <div v-else class="compact-empty">No attention-head indices were returned for this layer.</div>
            </template>
            <div v-else class="compact-empty">Layer metadata is not available.</div>
          </section>

          <section class="panel attention-panel" aria-labelledby="attention-title">
            <header class="panel-header">
              <div>
                <span class="section-kicker">Attention readout</span>
                <h2 id="attention-title">
                  {{ selectedHead === null ? 'Select an attention head' : `Layer ${selectedLayer} / Head ${selectedHead}` }}
                </h2>
              </div>
              <span
                class="provenance-badge"
                :class="`provenance-badge--${headProvenance}`"
                :title="provenanceDescription(headProvenance, headDetail)"
              >
                <span class="provenance-badge__dot" aria-hidden="true" />
                {{ headProvenance }}
              </span>
            </header>

            <div v-if="headState === 'loading'" class="inline-loading" role="status">
              <RefreshCw class="spinning" :size="17" aria-hidden="true" />
              Requesting the attention matrix…
            </div>
            <div v-else-if="headError" class="inline-message inline-message--error" role="alert">
              <AlertTriangle :size="17" aria-hidden="true" />
              <div><strong>Attention head unavailable.</strong><span>{{ headError }}</span></div>
            </div>
            <template v-else-if="headMatrix.length">
              <div class="matrix-meta">
                <span>{{ headMatrix.length }} query rows</span>
                <span>{{ headMatrix[0]?.length || 0 }} key columns</span>
                <span>Color normalized to returned maximum {{ formatMetric(headMatrixMaximum) }}</span>
              </div>
              <div class="matrix-scroll" tabindex="0" aria-label="Scrollable attention matrix">
                <table
                  class="attention-matrix"
                  :style="{ '--matrix-cell': `${matrixCellSize}px` }"
                >
                  <caption class="visually-hidden">
                    Attention values returned for layer {{ selectedLayer }}, head {{ selectedHead }}
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Query / key</th>
                      <th v-for="(token, index) in headTokens" :key="`column-${index}-${token}`" scope="col">
                        <span :title="formatToken(token)">{{ formatToken(token) }}</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, rowIndex) in headMatrix" :key="`row-${rowIndex}`">
                      <th scope="row">
                        <span :title="formatToken(headTokens[rowIndex] || '')">{{ formatToken(headTokens[rowIndex] || `t${rowIndex}`) }}</span>
                      </th>
                      <td
                        v-for="(cell, columnIndex) in row"
                        :key="`cell-${rowIndex}-${columnIndex}`"
                        :style="cellStyle(cell)"
                        :title="cellTitle(cell, rowIndex, columnIndex)"
                      >
                        {{ cell === null ? '—' : formatMetric(cell, 2) }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p v-if="headText(headDetail, 'prompt')" class="matrix-prompt">
                Returned prompt: <q>{{ headText(headDetail, 'prompt') }}</q>
              </p>
            </template>
            <div v-else class="empty-attention">
              <span class="empty-attention__mark" aria-hidden="true"><Search :size="20" /></span>
              <strong>No attention matrix is available.</strong>
              <span>Run a prompt, then select a layer and head. No pattern is inferred or generated here.</span>
            </div>
          </section>
        </div>

        <aside class="right-rail">
          <section class="panel prompt-result" aria-labelledby="prompt-result-title">
            <header class="panel-header panel-header--compact">
              <div>
                <span class="section-kicker">Prompt result</span>
                <h2 id="prompt-result-title">Returned inference</h2>
              </div>
              <span
                class="provenance-badge"
                :class="`provenance-badge--${promptProvenance}`"
                :title="provenanceDescription(promptProvenance, promptResult)"
              >
                <span class="provenance-badge__dot" aria-hidden="true" />
                {{ promptProvenance }}
              </span>
            </header>

            <div v-if="promptState === 'loading'" class="inline-loading" role="status">
              <RefreshCw class="spinning" :size="17" aria-hidden="true" />
              Running the prompt endpoint…
            </div>
            <div v-else-if="promptError" class="inline-message inline-message--error" role="alert" data-testid="transformer-run-error">
              <AlertTriangle :size="17" aria-hidden="true" />
              <div><strong>Prompt run failed.</strong><span>{{ promptError }}</span></div>
            </div>
            <template v-else-if="promptResult">
              <div v-if="promptNextToken" class="next-token" data-testid="transformer-next-token">
                <span>Next token</span>
                <code>{{ formatToken(promptNextToken) }}</code>
              </div>

              <div class="result-summary" data-testid="transformer-prompt-summary">
                <span>{{ promptTokens.length }} returned tokens</span>
                <span>{{ promptPredictions.length }} returned predictions</span>
              </div>

              <div v-if="promptTokens.length" class="token-strip" data-testid="transformer-tokens" aria-label="Returned prompt tokens">
                <span v-for="(token, index) in promptTokens" :key="`prompt-token-${index}-${token}`">
                  {{ formatToken(token) }}
                </span>
              </div>

              <ol v-if="promptPredictions.length" class="prediction-list" aria-label="Returned top predictions">
                <li v-for="(prediction, index) in promptPredictions" :key="`prediction-${index}-${prediction.token}`">
                  <span>{{ String(index + 1).padStart(2, '0') }}</span>
                  <code>{{ formatToken(prediction.token) }}</code>
                  <span v-if="prediction.logit !== undefined">logit {{ formatMetric(prediction.logit) }}</span>
                  <span v-if="prediction.probability !== undefined">p {{ formatProbability(prediction.probability) }}</span>
                </li>
              </ol>
              <div v-else class="compact-empty">No prediction list was returned.</div>

              <p class="provenance-note provenance-note--right">
                {{ provenanceDescription(promptProvenance, promptResult) }}
              </p>
            </template>
            <div v-else class="empty-attention">
              <span class="empty-attention__mark" aria-hidden="true"><Cpu :size="20" /></span>
              <strong>No prompt result yet.</strong>
              <span>The output area stays empty until the endpoint returns data.</span>
            </div>
          </section>

          <section class="panel head-detail" aria-labelledby="head-detail-title">
            <header class="panel-header panel-header--compact">
              <div>
                <span class="section-kicker">Head detail</span>
                <h2 id="head-detail-title">
                  {{ selectedHead === null ? 'No head selected' : `L${selectedLayer}H${selectedHead}` }}
                </h2>
              </div>
            </header>

            <template v-if="headSummaryStats.length">
              <dl class="head-stats">
                <div v-for="row in headSummaryStats" :key="row.label">
                  <dt>{{ row.label }}</dt>
                  <dd>{{ row.value }}</dd>
                </div>
              </dl>
              <div v-if="headTopToken" class="top-attention">
                <span>Last-token top attention</span>
                <div>
                  <code>{{ formatToken(headTopToken) }}</code>
                  <strong v-if="headTopWeight !== undefined">{{ formatMetric(headTopWeight) }}</strong>
                </div>
              </div>
              <p class="source-caption">Weight and top-token values come from the returned layer metadata.</p>
            </template>
            <div v-else class="compact-empty">
              No weight summary was returned. The attention matrix remains available when present.
            </div>
          </section>
        </aside>
      </div>
    </template>
  </section>
</template>

<script setup lang="ts">
import {
  AlertTriangle,
  Box,
  Braces,
  ChevronRight,
  Cpu,
  Database,
  Layers,
  Network,
  Play,
  RefreshCw,
  Search,
  WifiOff,
} from 'lucide-vue-next';
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { api } from '../services/api';

type JsonRecord = Record<string, unknown>;
type Provenance = 'live' | 'reference' | 'unavailable';
type ArchitectureState = 'idle' | 'loading' | 'ready' | 'empty' | 'error';
type DetailState = 'idle' | 'loading' | 'ready' | 'error';
type PromptState = 'idle' | 'loading' | 'ready' | 'error';

interface StatRow {
  label: string;
  value: string;
}

interface Prediction {
  token: string;
  logit?: number;
  probability?: number;
}

const architecture = ref<JsonRecord | null>(null);
const architectureState = ref<ArchitectureState>('idle');
const architectureError = ref('');
const architectureWarning = ref('');
const architectureProvenance = ref<Provenance>('unavailable');
const lastUpdated = ref<Date | null>(null);

const selectedLayer = ref<number | null>(null);
const selectedHead = ref<number | null>(null);
const layerDetail = ref<JsonRecord | null>(null);
const layerState = ref<DetailState>('idle');
const layerError = ref('');
const layerProvenance = ref<Provenance>('unavailable');
const headDetail = ref<JsonRecord | null>(null);
const headState = ref<DetailState>('idle');
const headError = ref('');
const headProvenance = ref<Provenance>('unavailable');

const prompt = ref('The capital of France is');
const promptResult = ref<JsonRecord | null>(null);
const promptState = ref<PromptState>('idle');
const promptError = ref('');
const promptProvenance = ref<Provenance>('unavailable');

const browserOnline = ref(typeof navigator === 'undefined' ? true : navigator.onLine);
const backendUnavailable = ref(false);

let architectureRequestId = 0;
let layerRequestId = 0;
let headRequestId = 0;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function textValue(value: unknown): string {
  return typeof value === 'string' || typeof value === 'number' ? String(value) : '';
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function integerValue(value: unknown): number | undefined {
  const parsed = numberValue(value);
  return parsed !== undefined && Number.isInteger(parsed) ? parsed : undefined;
}

function firstValue(source: JsonRecord | null | undefined, keys: string[]): unknown {
  if (!source) return undefined;
  for (const key of keys) {
    if (source[key] !== undefined && source[key] !== null) return source[key];
  }
  return undefined;
}

function firstText(source: JsonRecord | null | undefined, keys: string[]): string {
  return textValue(firstValue(source, keys)).trim();
}

function firstNumber(source: JsonRecord | null | undefined, keys: string[]): number | undefined {
  return numberValue(firstValue(source, keys));
}

function arrayValue(source: JsonRecord | null | undefined, key: string): unknown[] {
  const value = source?.[key];
  return Array.isArray(value) ? value : [];
}

function responseError(value: unknown, label: string): string | null {
  if (!isRecord(value)) return `${label} returned an unreadable response.`;
  const explicit = firstText(value, ['error', 'message', 'detail']);
  if (explicit) return explicit;
  const status = firstText(value, ['status']).toLowerCase();
  if (['error', 'failed', 'unavailable'].includes(status)) {
    return `${label} returned status “${status}” without a readable result.`;
  }
  return null;
}

function requireResponse(value: unknown, label: string): JsonRecord {
  const error = responseError(value, label);
  if (error) throw new Error(error);
  if (!isRecord(value)) throw new Error(`${label} returned an unreadable response.`);
  return value;
}

function errorText(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return textValue(error) || 'Unknown runtime error.';
}

function looksOffline(error: unknown): boolean {
  const message = errorText(error).toLowerCase();
  return message.includes('failed to fetch')
    || message.includes('networkerror')
    || message.includes('load failed')
    || message.includes('timed out')
    || message.includes('connection');
}

function provenanceFrom(value: unknown, fallback: Provenance): Provenance {
  if (!isRecord(value)) return fallback;
  const raw = firstText(value, ['provenance']).toLowerCase();
  if (raw === 'live') return 'live';
  if (raw === 'seeded' || raw === 'reference') return 'reference';
  if (raw === 'unavailable') return 'unavailable';
  return fallback;
}

function provenanceDescription(provenance: Provenance, source: JsonRecord | null): string {
  if (provenance === 'live') return 'The backend marked this response as live model data.';
  if (provenance === 'unavailable') return 'No verified model data is currently available.';
  if (firstText(source, ['provenance']).toLowerCase() === 'seeded') {
    return 'The backend marked this response as seeded reference data; it is not live model output.';
  }
  if (firstText(source, ['provenance']).toLowerCase() === 'reference') {
    return 'The backend marked this response as reference data; it is not live model output.';
  }
  return 'The backend returned descriptive data without a live provenance marker.';
}

function formatInteger(value: unknown): string {
  const parsed = numberValue(value);
  return parsed === undefined ? '—' : new Intl.NumberFormat().format(parsed);
}

function formatMetric(value: unknown, digits = 4): string {
  const parsed = numberValue(value);
  return parsed === undefined ? '—' : parsed.toFixed(digits);
}

function formatProbability(value: unknown): string {
  const parsed = numberValue(value);
  return parsed === undefined ? '—' : `${(parsed * 100).toFixed(2)}%`;
}

function formatToken(value: string): string {
  return value
    .replaceAll('Ġ', '␣')
    .replaceAll('Ċ', '⏎')
    .replaceAll('\n', '⏎')
    .trim() || '∅';
}

function headText(source: JsonRecord | null, key: string): string {
  return firstText(source, [key]);
}

function architectureValue(...keys: string[]): unknown {
  return firstValue(architecture.value, keys);
}

function moduleById(id: string): JsonRecord | null {
  const module = arrayValue(architecture.value, 'modules').find((item) => isRecord(item) && firstText(item, ['id']) === id);
  return isRecord(module) ? module : null;
}

function moduleLabel(id: string, fallback: string): string {
  return firstText(moduleById(id), ['label', 'path']) || fallback;
}

function moduleShape(id: string): string {
  const shape = moduleById(id)?.shape;
  if (!Array.isArray(shape)) return 'shape unavailable';
  const dimensions = shape.map(numberValue).filter((item): item is number => item !== undefined);
  return dimensions.length ? `[${dimensions.join(' × ')}]` : 'shape unavailable';
}

function architectureLayers(): JsonRecord[] {
  return arrayValue(architecture.value, 'layers').filter(isRecord);
}

function architectureLayerRecord(index: number | null): JsonRecord | null {
  if (index === null) return null;
  return architectureLayers().find((layer) => integerValue(layer.layer_index) === index) ?? null;
}

function layerCountFrom(source: JsonRecord): number {
  const explicit = integerValue(firstValue(source, ['n_layers', 'num_layers', 'layer_count']));
  if (explicit !== undefined && explicit >= 0) return explicit;
  return arrayValue(source, 'layers').filter(isRecord).length;
}

const architectureLayersList = computed(() => architectureLayers());
const layerCount = computed(() => architecture.value ? layerCountFrom(architecture.value) : 0);
const layerOptions = computed(() => {
  const indices = architectureLayersList.value
    .map((layer) => integerValue(layer.layer_index))
    .filter((index): index is number => index !== undefined && index >= 0);
  for (let index = 0; index < layerCount.value; index += 1) indices.push(index);
  return [...new Set(indices)].sort((left, right) => left - right);
});
const selectedArchitectureLayer = computed(() => architectureLayerRecord(selectedLayer.value));

const architectureStats = computed<StatRow[]>(() => {
  const source = architecture.value;
  if (!source) return [];
  const raw: Array<[string, string]> = [
    ['Model', firstText(source, ['model_name', 'model', 'name'])],
    ['Family', firstText(source, ['model_type', 'architecture', 'family'])],
    ['Layers', formatInteger(firstNumber(source, ['n_layers', 'num_layers']))],
    ['Heads', formatInteger(firstNumber(source, ['n_heads', 'num_attention_heads']))],
    ['d_model', formatInteger(firstNumber(source, ['d_model', 'hidden_size']))],
    ['d_mlp', formatInteger(firstNumber(source, ['d_mlp', 'intermediate_size']))],
    ['d_head', formatInteger(firstNumber(source, ['d_head', 'head_dim']))],
    ['Vocabulary', formatInteger(firstNumber(source, ['vocab_size', 'vocabulary_size']))],
    ['Parameters', firstText(source, ['n_params_human']) || formatInteger(firstNumber(source, ['n_params', 'num_parameters']))],
    ['Device', firstText(source, ['device', 'dtype'])],
  ];
  return raw
    .filter(([, value]) => value && value !== '—')
    .map(([label, value]) => ({ label, value }));
});

const layerComponents = computed<JsonRecord[]>(() => {
  const source = selectedArchitectureLayer.value;
  return source ? arrayValue(source, 'components').filter(isRecord) : [];
});

function componentKey(component: JsonRecord): string {
  return firstText(component, ['id', 'path', 'type']) || 'component';
}

function componentLabel(component: JsonRecord): string {
  return firstText(component, ['id', 'type']) || 'component';
}

function componentPath(component: JsonRecord): string {
  return firstText(component, ['path']);
}

const headCount = computed(() => {
  const selectedRecord = selectedArchitectureLayer.value;
  return firstNumber(selectedRecord, ['num_attention_heads', 'n_heads'])
    ?? firstNumber(layerDetail.value, ['num_attention_heads', 'n_heads'])
    ?? firstNumber(architecture.value, ['n_heads', 'num_attention_heads'])
    ?? 0;
});

const headOptions = computed(() => {
  const returned = arrayValue(layerDetail.value, 'attention_heads')
    .filter(isRecord)
    .map((head) => integerValue(head.head_index))
    .filter((index): index is number => index !== undefined && index >= 0);
  if (returned.length) return [...new Set(returned)].sort((left, right) => left - right);
  return Array.from({ length: Math.max(0, headCount.value) }, (_, index) => index);
});

function headSummary(index: number): JsonRecord | null {
  const head = arrayValue(layerDetail.value, 'attention_heads').find((item) => (
    isRecord(item) && integerValue(item.head_index) === index
  ));
  return isRecord(head) ? head : null;
}

const layerStats = computed<StatRow[]>(() => {
  const source = layerDetail.value;
  if (!source) return [];
  const cache = source.has_activations === true
    ? 'available'
    : source.has_activations === false
      ? 'not populated'
      : '';
  const rows: Array<[string, string]> = [
    ['Path', firstText(source, ['path', 'label'])],
    ['Attention heads', formatInteger(firstNumber(source, ['num_attention_heads', 'n_heads']))],
    ['MLP neurons', formatInteger(firstNumber(source, ['num_mlp_neurons', 'd_mlp']))],
    ['Residual dimension', formatInteger(firstNumber(source, ['residual_stream_dim', 'd_model']))],
    ['Parameters', formatInteger(firstNumber(source, ['n_params', 'num_parameters']))],
    ['Activation cache', cache],
  ];
  return rows
    .filter(([, value]) => value && value !== '—')
    .map(([label, value]) => ({ label, value }));
});

const headSummaryStats = computed<StatRow[]>(() => {
  const summary = selectedHead.value === null ? null : headSummary(selectedHead.value);
  if (!summary) return [];
  const rows: Array<[string, string]> = [
    ['Query weight L2', formatMetric(summary.q_weight_l2)],
    ['Key weight L2', formatMetric(summary.k_weight_l2)],
    ['Value weight L2', formatMetric(summary.v_weight_l2)],
    ['Output weight L2', formatMetric(summary.o_weight_l2)],
    ['Head dimension', formatInteger(firstNumber(summary, ['d_head', 'head_dim']))],
  ];
  return rows
    .filter(([, value]) => value && value !== '—')
    .map(([label, value]) => ({ label, value }));
});

const headTopToken = computed(() => {
  const summary = selectedHead.value === null ? null : headSummary(selectedHead.value);
  return firstText(summary, ['top_token']);
});

const headTopWeight = computed(() => {
  const summary = selectedHead.value === null ? null : headSummary(selectedHead.value);
  return firstNumber(summary, ['top_attention_weight']);
});

function extractTokens(value: JsonRecord | null): string[] {
  const tokens = firstValue(value, ['str_tokens', 'tokens', 'token_strings']);
  if (!Array.isArray(tokens)) return [];
  return tokens
    .map((token) => typeof token === 'string' ? token : isRecord(token) ? firstText(token, ['token', 'text', 'value']) : '')
    .filter(Boolean);
}

const headTokens = computed(() => extractTokens(headDetail.value));
const headMatrix = computed<Array<Array<number | null>>>(() => {
  const matrix = firstValue(headDetail.value, ['matrix', 'attention_matrix']);
  if (!Array.isArray(matrix)) return [];
  return matrix.map((row) => (
    Array.isArray(row)
      ? row.map((cell) => numberValue(cell) ?? null)
      : []
  ));
});
const flattenedHeadMatrix = computed(() => headMatrix.value.flat());
const headMatrixMaximum = computed(() => {
  const values = flattenedHeadMatrix.value.filter((value): value is number => value !== null);
  return values.length ? Math.max(...values) : 0;
});
const matrixCellSize = computed(() => {
  const columns = headTokens.value.length || headMatrix.value[0]?.length || 1;
  if (columns > 24) return 25;
  if (columns > 16) return 29;
  return 34;
});

function cellStyle(cell: number | null): Record<string, string> {
  if (cell === null || !Number.isFinite(cell) || !headMatrixMaximum.value) {
    return { background: '#f1f5f9', color: '#647184' };
  }
  const ratio = Math.max(0, Math.min(1, cell / headMatrixMaximum.value));
  return {
    background: `rgba(37, 99, 235, ${(0.08 + ratio * 0.78).toFixed(3)})`,
    color: ratio > 0.58 ? '#ffffff' : '#182230',
  };
}

function cellTitle(cell: number | null, row: number, column: number): string {
  if (cell === null) return `Query ${row}, key ${column}: no value returned`;
  return `Query ${row}, key ${column}: ${formatMetric(cell, 6)}`;
}

function extractPredictions(value: JsonRecord | null): Prediction[] {
  const candidates = firstValue(value, ['top5', 'top_predictions', 'predictions', 'top16']);
  if (!Array.isArray(candidates)) return [];
  return candidates
    .map((item): Prediction | null => {
      if (typeof item === 'string') return item ? { token: item } : null;
      if (!isRecord(item)) return null;
      const token = firstText(item, ['token', 'token_str', 'text', 'value']);
      if (!token) return null;
      return {
        token,
        logit: firstNumber(item, ['logit', 'logit_value']),
        probability: firstNumber(item, ['probability', 'prob', 'score']),
      };
    })
    .filter((item): item is Prediction => item !== null)
    .slice(0, 5);
}

const promptTokens = computed(() => extractTokens(promptResult.value));
const promptPredictions = computed(() => extractPredictions(promptResult.value));
const promptNextToken = computed(() => firstText(promptResult.value, ['next_token', 'nextToken', 'next_token_text']));

const isOffline = computed(() => !browserOnline.value || backendUnavailable.value);
const canRunPrompt = computed(() => Boolean(prompt.value.trim()) && promptState.value !== 'loading' && !isOffline.value);
const isBusy = computed(() => (
  architectureState.value === 'loading'
  || layerState.value === 'loading'
  || headState.value === 'loading'
  || promptState.value === 'loading'
));
const runtimeTone = computed<'offline' | 'checking' | 'online'>(() => {
  if (isOffline.value) return 'offline';
  if (architectureState.value === 'loading') return 'checking';
  return 'online';
});
const runtimeLabel = computed(() => {
  if (!browserOnline.value) return 'Browser offline';
  if (backendUnavailable.value) return 'Runtime unavailable';
  if (architectureState.value === 'loading') return 'Checking runtime';
  return 'Runtime online';
});
const statusTone = computed<'offline' | 'error' | 'warning'>(() => {
  if (isOffline.value) return 'offline';
  return architectureError.value ? 'error' : 'warning';
});
const statusTitle = computed(() => {
  if (isOffline.value) return 'Runtime connection unavailable';
  return architectureError.value ? 'Architecture request failed' : 'Refresh incomplete';
});
const statusMessage = computed(() => {
  if (isOffline.value) {
    return 'Existing returned data is retained with its original provenance. Reconnect and retry to request current model data.';
  }
  if (architectureError.value) return architectureError.value;
  return architectureWarning.value;
});
const lastUpdatedLabel = computed(() => {
  if (!lastUpdated.value) return '';
  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(lastUpdated.value);
});

async function selectHead(index: number): Promise<void> {
  if (selectedLayer.value === null || !Number.isInteger(index) || index < 0) return;
  const requestId = ++headRequestId;
  selectedHead.value = index;
  headState.value = 'loading';
  headError.value = '';
  headDetail.value = null;
  headProvenance.value = 'unavailable';

  try {
    const response = requireResponse(
      await api.gpt2AttentionHead(selectedLayer.value, index),
      `Attention head L${selectedLayer.value}H${index}`,
    );
    if (requestId !== headRequestId) return;
    headDetail.value = response;
    headProvenance.value = provenanceFrom(response, 'reference');
    headState.value = 'ready';
  } catch (error) {
    if (requestId !== headRequestId) return;
    headDetail.value = null;
    headProvenance.value = 'unavailable';
    headError.value = errorText(error);
    headState.value = 'error';
  }
}

async function selectLayer(index: number, preferredHead = 0): Promise<void> {
  if (!Number.isInteger(index) || index < 0) return;
  const requestId = ++layerRequestId;
  ++headRequestId;
  const previousHead = preferredHead;
  selectedLayer.value = index;
  selectedHead.value = null;
  layerState.value = 'loading';
  layerError.value = '';
  layerDetail.value = null;
  layerProvenance.value = 'unavailable';
  headDetail.value = null;
  headState.value = 'idle';
  headError.value = '';
  headProvenance.value = 'unavailable';

  try {
    const response = requireResponse(await api.gpt2Layer(index), `Layer ${index}`);
    if (requestId !== layerRequestId) return;
    layerDetail.value = response;
    layerProvenance.value = provenanceFrom(response, 'reference');
    layerState.value = 'ready';
    if (headOptions.value.length) {
      const target = headOptions.value.includes(previousHead) ? previousHead : headOptions.value[0];
      await selectHead(target);
    }
  } catch (error) {
    if (requestId !== layerRequestId) return;
    layerDetail.value = null;
    layerProvenance.value = 'unavailable';
    layerError.value = errorText(error);
    layerState.value = 'error';
  }
}

async function loadArchitecture(silent = false): Promise<void> {
  const requestId = ++architectureRequestId;
  const previousLayer = selectedLayer.value;
  const previousHead = selectedHead.value ?? 0;
  architectureWarning.value = '';
  if (!silent) {
    architectureState.value = 'loading';
    architectureError.value = '';
  }

  try {
    const response = requireResponse(await api.gpt2Architecture(), 'GPT-2 architecture');
    if (requestId !== architectureRequestId) return;
    architecture.value = response;
    architectureProvenance.value = provenanceFrom(response, 'reference');
    backendUnavailable.value = false;
    lastUpdated.value = new Date();

    const availableLayers = layerOptions.value;
    if (!availableLayers.length) {
      selectedLayer.value = null;
      selectedHead.value = null;
      layerDetail.value = null;
      headDetail.value = null;
      architectureState.value = 'empty';
      return;
    }

    const nextLayer = previousLayer !== null && availableLayers.includes(previousLayer)
      ? previousLayer
      : availableLayers[0];
    architectureState.value = 'ready';
    architectureError.value = '';
    await selectLayer(nextLayer, previousHead);
  } catch (error) {
    if (requestId !== architectureRequestId) return;
    const message = errorText(error);
    if (silent && architecture.value) {
      architectureWarning.value = `Architecture refresh failed: ${message}`;
      return;
    }
    architecture.value = null;
    architectureState.value = 'error';
    architectureError.value = message;
    architectureProvenance.value = 'unavailable';
    selectedLayer.value = null;
    selectedHead.value = null;
    layerDetail.value = null;
    headDetail.value = null;
    backendUnavailable.value = looksOffline(error);
  }
}

async function runPrompt(): Promise<void> {
  const text = prompt.value.trim();
  if (!text) {
    promptError.value = 'Enter a prompt before running inference.';
    promptState.value = 'error';
    return;
  }
  if (isOffline.value) {
    promptError.value = 'Reconnect the runtime before running inference.';
    promptState.value = 'error';
    return;
  }
  if (promptState.value === 'loading') return;

  promptState.value = 'loading';
  promptError.value = '';
  promptResult.value = null;
  promptProvenance.value = 'unavailable';

  try {
    const response = requireResponse(await api.gpt2RunPrompt(text), 'GPT-2 prompt run');
    promptResult.value = response;
    promptProvenance.value = provenanceFrom(response, 'reference');
    promptState.value = 'ready';
    await loadArchitecture(true);
  } catch (error) {
    promptResult.value = null;
    promptProvenance.value = 'unavailable';
    promptError.value = errorText(error);
    promptState.value = 'error';
  }
}

function onPromptKeydown(event: KeyboardEvent): void {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && canRunPrompt.value) {
    event.preventDefault();
    void runPrompt();
  }
}

function handleOnline(): void {
  browserOnline.value = true;
  backendUnavailable.value = false;
  if (architectureState.value === 'error' || architectureState.value === 'empty') {
    void loadArchitecture();
  }
}

function handleOffline(): void {
  browserOnline.value = false;
}

onMounted(() => {
  window.addEventListener('online', handleOnline);
  window.addEventListener('offline', handleOffline);
  void loadArchitecture();
});

onBeforeUnmount(() => {
  window.removeEventListener('online', handleOnline);
  window.removeEventListener('offline', handleOffline);
  architectureRequestId += 1;
  layerRequestId += 1;
  headRequestId += 1;
});
</script>

<style scoped>
.transformer-visualizer {
  --tv-primary: #2563eb;
  --tv-primary-dark: #1d4ed8;
  --tv-primary-soft: #eaf1ff;
  --tv-ink: #182230;
  --tv-muted: #647184;
  --tv-subtle: #8c99a7;
  --tv-border: #d9e0e8;
  --tv-border-strong: #c8d1dc;
  --tv-surface: #ffffff;
  --tv-surface-soft: #f8fafc;
  --tv-canvas: #f4f6f8;
  min-width: 0;
  min-height: 100%;
  padding: 16px;
  background: var(--tv-canvas);
  color: var(--tv-ink);
  color-scheme: light;
  font-family: var(--font, "IBM Plex Sans", "Segoe UI", sans-serif);
  font-size: 12px;
}

.transformer-visualizer *,
.transformer-visualizer *::before,
.transformer-visualizer *::after {
  box-sizing: border-box;
}

.visualizer-header,
.title-lockup,
.header-actions,
.runtime-chip,
.panel-header,
.panel-header__meta,
.prompt-command,
.prompt-command__actions,
.architecture-flow,
.architecture-node,
.residual-strip,
.layer-path-row,
.head-selector-heading,
.matrix-meta,
.result-summary,
.top-attention > div,
.layer-source,
.state-banner,
.state-banner > div,
.next-token,
.source-caption {
  display: flex;
  align-items: center;
}

.visualizer-header {
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 12px;
}

.title-lockup {
  min-width: 0;
  gap: 11px;
}

.title-mark {
  display: grid;
  width: 40px;
  height: 40px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid var(--tv-border-strong);
  border-radius: 9px;
  background: var(--tv-surface);
  box-shadow: 0 1px 2px rgba(24, 34, 48, 0.05);
  color: var(--tv-primary);
}

.title-kicker,
.section-kicker,
.received-at,
.provenance-badge,
.count-badge,
.runtime-chip,
.matrix-meta,
.result-summary,
.layer-source,
.source-caption,
.matrix-prompt,
.provenance-note {
  font-family: var(--font-mono, "IBM Plex Mono", "Cascadia Code", monospace);
}

.title-kicker,
.section-kicker {
  color: var(--tv-muted);
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.visualizer-header h1 {
  margin: 2px 0 0;
  color: var(--tv-ink);
  font-size: clamp(21px, 2.2vw, 28px);
  font-weight: 730;
  letter-spacing: -0.035em;
  line-height: 1.1;
}

.visualizer-header p {
  margin: 4px 0 0;
  color: var(--tv-muted);
  font-size: 11px;
}

.header-actions,
.panel-header__meta,
.prompt-command__actions {
  gap: 8px;
  flex: 0 0 auto;
}

.runtime-chip,
.provenance-badge,
.count-badge {
  display: inline-flex;
  min-height: 25px;
  align-items: center;
  gap: 6px;
  padding: 0 8px;
  border: 1px solid var(--tv-border);
  border-radius: 999px;
  background: var(--tv-surface-soft);
  color: var(--tv-muted);
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.045em;
  text-transform: uppercase;
  white-space: nowrap;
}

.runtime-chip__dot,
.provenance-badge__dot {
  width: 6px;
  height: 6px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: currentColor;
}

.runtime-chip--online,
.provenance-badge--live {
  border-color: #b9dfd3;
  background: var(--success-soft, #e7f6f1);
  color: var(--success, #13795f);
}

.runtime-chip--checking {
  color: var(--warning, #946200);
}

.runtime-chip--offline,
.provenance-badge--unavailable {
  border-color: #efc2c7;
  background: var(--danger-soft, #fff0f1);
  color: var(--danger, #bf3f4d);
}

.provenance-badge--reference {
  border-color: #ecd79c;
  background: var(--warning-soft, #fff6dc);
  color: var(--warning, #946200);
}

.refresh-button,
.run-button,
.state-banner__action,
.message-panel > button {
  display: inline-flex;
  min-height: 34px;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 0 12px;
  border: 1px solid var(--tv-border);
  border-radius: 6px;
  background: var(--tv-surface);
  color: var(--tv-ink);
  cursor: pointer;
  font-weight: 650;
}

.refresh-button:hover,
.state-banner__action:hover,
.message-panel > button:hover {
  border-color: var(--tv-border-strong);
  background: var(--tv-surface-soft);
}

.run-button,
.message-panel > button {
  border-color: var(--tv-primary);
  background: var(--tv-primary);
  color: #ffffff;
}

.run-button:hover,
.message-panel > button:hover {
  border-color: var(--tv-primary-dark);
  background: var(--tv-primary-dark);
}

.refresh-button:disabled,
.run-button:disabled,
.state-banner__action:disabled,
.message-panel > button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.transformer-visualizer button:focus-visible,
.transformer-visualizer textarea:focus-visible,
.matrix-scroll:focus-visible {
  outline: 2px solid var(--tv-primary);
  outline-offset: 2px;
}

.state-banner {
  gap: 10px;
  margin-bottom: 12px;
  padding: 10px 12px;
  border: 1px solid var(--tv-border);
  border-radius: 7px;
  background: var(--tv-surface);
  color: var(--tv-ink);
}

.state-banner > div {
  align-items: flex-start;
  flex: 1;
  flex-direction: column;
  gap: 2px;
}

.state-banner strong {
  font-size: 11px;
}

.state-banner span {
  color: var(--tv-muted);
  font-size: 10px;
}

.state-banner--offline,
.state-banner--error {
  border-color: #efc2c7;
  background: var(--danger-soft, #fff0f1);
  color: var(--danger, #bf3f4d);
}

.state-banner--warning {
  border-color: #ecd79c;
  background: var(--warning-soft, #fff6dc);
  color: var(--warning, #946200);
}

.state-banner__action {
  min-height: 29px;
  padding: 0 9px;
  font-size: 10px;
}

.loading-panel,
.message-panel {
  min-height: 310px;
  border: 1px solid var(--tv-border);
  border-radius: 8px;
  background: var(--tv-surface);
  box-shadow: 0 1px 2px rgba(24, 34, 48, 0.04);
}

.loading-panel {
  display: grid;
  place-items: center;
  align-content: center;
  gap: 10px;
  padding: 40px;
  text-align: center;
}

.loading-panel__mark,
.message-panel__mark,
.empty-attention__mark {
  display: grid;
  width: 42px;
  height: 42px;
  place-items: center;
  border: 1px solid var(--tv-border);
  border-radius: 9px;
  background: var(--tv-surface-soft);
  color: var(--tv-primary);
}

.loading-panel strong,
.message-panel h2,
.empty-attention strong {
  display: block;
  color: var(--tv-ink);
  font-size: 13px;
}

.loading-panel span,
.message-panel p,
.empty-attention > span:last-child {
  color: var(--tv-muted);
  font-size: 10px;
}

.skeleton-grid {
  display: grid;
  width: min(420px, 80vw);
  grid-template-columns: repeat(6, 1fr);
  gap: 6px;
  margin-top: 12px;
}

.skeleton-grid span {
  height: 30px;
  border-radius: 5px;
  background: var(--tv-surface-soft);
  animation: tv-pulse 1.4s ease-in-out infinite;
}

.message-panel {
  display: flex;
  min-height: 210px;
  align-items: center;
  justify-content: center;
  gap: 14px;
  padding: 34px;
}

.message-panel > div {
  max-width: 560px;
}

.message-panel h2 {
  margin: 3px 0 5px;
  font-size: 17px;
}

.message-panel p {
  margin: 0;
  line-height: 1.5;
}

.message-panel--error .message-panel__mark {
  border-color: #efc2c7;
  background: var(--danger-soft, #fff0f1);
  color: var(--danger, #bf3f4d);
}

.prompt-command {
  display: grid;
  grid-template-columns: minmax(150px, 0.65fr) minmax(280px, 1.8fr) auto;
  gap: 10px;
  margin-bottom: 12px;
  padding: 10px;
  border: 1px solid var(--tv-border);
  border-radius: 8px;
  background: var(--tv-surface);
  box-shadow: 0 1px 2px rgba(24, 34, 48, 0.04);
}

.prompt-command__label {
  align-self: center;
}

.prompt-command label {
  display: block;
  margin-top: 2px;
  color: var(--tv-ink);
  font-size: 10px;
  font-weight: 650;
}

.prompt-command textarea {
  width: 100%;
  min-height: 50px;
  max-height: 120px;
  padding: 8px 9px;
  border: 1px solid var(--tv-border);
  border-radius: 6px;
  background: var(--tv-surface-soft);
  color: var(--tv-ink);
  font: 11px/1.45 var(--font, "IBM Plex Sans", "Segoe UI", sans-serif);
  resize: vertical;
}

.prompt-command textarea::placeholder {
  color: var(--tv-subtle);
}

.prompt-command textarea:focus {
  border-color: var(--tv-primary);
  outline: 0;
  box-shadow: 0 0 0 2px var(--tv-primary-soft);
}

.prompt-command__actions {
  align-items: flex-end;
  flex-direction: column;
  justify-content: center;
  gap: 5px;
}

.prompt-command__actions > span {
  color: var(--tv-muted);
  font: 9px/1.2 var(--font-mono, "IBM Plex Mono", monospace);
}

.inline-error {
  grid-column: 2 / -1;
  margin: -2px 0 0;
  color: var(--danger, #bf3f4d);
  font-size: 10px;
}

.overview-panel,
.panel {
  min-width: 0;
  border: 1px solid var(--tv-border);
  border-radius: 8px;
  background: var(--tv-surface);
  box-shadow: 0 1px 2px rgba(24, 34, 48, 0.04);
}

.overview-panel {
  margin-bottom: 12px;
  padding: 12px;
}

.panel {
  overflow: hidden;
}

.panel-header {
  justify-content: space-between;
  gap: 12px;
  min-height: 54px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--tv-border);
}

.panel-header--compact {
  min-height: 49px;
}

.panel-header h2 {
  margin: 2px 0 0;
  color: var(--tv-ink);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.received-at {
  color: var(--tv-muted);
  font-size: 9px;
}

.architecture-flow {
  display: grid;
  grid-template-columns: minmax(190px, 1fr) 20px minmax(220px, 1.35fr) 20px minmax(170px, 0.9fr);
  align-items: stretch;
  gap: 5px;
  padding: 2px 0 10px;
  list-style: none;
}

.architecture-node {
  position: relative;
  min-width: 0;
  align-items: flex-start;
  gap: 8px;
  padding: 10px;
  border: 1px solid var(--tv-border);
  border-radius: 7px;
  background: var(--tv-surface-soft);
}

.architecture-node--selected {
  border-color: #b9ccef;
  background: #f5f8ff;
  box-shadow: inset 3px 0 0 var(--tv-primary);
}

.architecture-node__index {
  position: absolute;
  top: 7px;
  right: 8px;
  color: var(--tv-subtle);
  font: 8px/1 var(--font-mono, monospace);
}

.architecture-node__icon {
  display: grid;
  width: 30px;
  height: 30px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid var(--tv-border);
  border-radius: 6px;
  background: var(--tv-surface);
  color: var(--tv-primary);
}

.architecture-node__body {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 3px;
  padding-right: 18px;
}

.architecture-node__body strong {
  color: var(--tv-ink);
  font-size: 11px;
}

.architecture-node__body > span:not(.architecture-node__index),
.architecture-node__body code {
  overflow: hidden;
  color: var(--tv-muted);
  font-size: 9px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.architecture-node__body code {
  font-family: var(--font-mono, monospace);
}

.component-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 3px;
  margin-top: 3px;
}

.component-chips span {
  padding: 2px 5px;
  border: 1px solid #d6e2f7;
  border-radius: 4px;
  background: var(--tv-surface);
  color: #365a95;
  font: 8px/1.3 var(--font-mono, monospace);
}

.flow-arrow {
  display: grid;
  place-items: center;
  color: var(--tv-border-strong);
}

.residual-strip {
  gap: 8px;
  min-height: 30px;
  padding: 5px 8px;
  border: 1px dashed #b9ccef;
  border-radius: 6px;
  background: #f7faff;
  color: var(--tv-muted);
  font-size: 9px;
}

.residual-strip span:first-child {
  color: var(--tv-ink);
  font-weight: 650;
}

.residual-strip code {
  color: #365a95;
  font-family: var(--font-mono, monospace);
}

.architecture-stats {
  display: grid;
  grid-template-columns: repeat(10, minmax(74px, 1fr));
  gap: 1px;
  margin: 10px 0 0;
  overflow: hidden;
  border: 1px solid var(--tv-border);
  border-radius: 6px;
  background: var(--tv-border);
}

.architecture-stat {
  min-width: 0;
  padding: 7px 8px;
  background: var(--tv-surface);
}

.architecture-stat dt {
  overflow: hidden;
  color: var(--tv-muted);
  font: 8px/1.2 var(--font-mono, monospace);
  text-overflow: ellipsis;
  text-transform: uppercase;
  white-space: nowrap;
}

.architecture-stat dd {
  overflow: hidden;
  margin: 3px 0 0;
  color: var(--tv-ink);
  font: 650 10px/1.2 var(--font-mono, monospace);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.provenance-note {
  margin: 8px 0 0;
  color: var(--tv-muted);
  font-size: 9px;
  line-height: 1.45;
}

.workspace-grid {
  display: grid;
  grid-template-columns: minmax(180px, 0.48fr) minmax(460px, 1.55fr) minmax(290px, 0.78fr);
  align-items: start;
  gap: 12px;
}

.inspector-column,
.right-rail {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 12px;
}

.layer-panel {
  position: sticky;
  top: 0;
}

.count-badge {
  min-width: 26px;
  justify-content: center;
  border-color: var(--tv-border);
  background: var(--tv-surface-soft);
}

.layer-list {
  display: grid;
  max-height: min(520px, 56vh);
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 5px;
  padding: 8px;
  overflow: auto;
}

.layer-option {
  display: flex;
  min-width: 0;
  min-height: 44px;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
  gap: 2px;
  padding: 6px 7px;
  border: 1px solid var(--tv-border);
  border-radius: 5px;
  background: var(--tv-surface);
  color: var(--tv-ink);
  cursor: pointer;
  text-align: left;
}

.layer-option:hover {
  border-color: var(--tv-border-strong);
  background: var(--tv-surface-soft);
}

.layer-option--selected {
  border-color: var(--tv-primary);
  background: var(--tv-primary-soft);
  box-shadow: inset 2px 0 0 var(--tv-primary);
}

.layer-option__index {
  color: var(--tv-primary-dark);
  font: 700 9px/1.1 var(--font-mono, monospace);
}

.layer-option__path {
  width: 100%;
  overflow: hidden;
  color: var(--tv-muted);
  font: 8px/1.1 var(--font-mono, monospace);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.layer-source {
  justify-content: space-between;
  gap: 8px;
  padding: 8px 10px;
  border-top: 1px solid var(--tv-border);
  background: var(--tv-surface-soft);
  color: var(--tv-muted);
  font-size: 8px;
}

.layer-source code {
  overflow: hidden;
  color: var(--tv-ink);
  font-size: 8px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.layer-detail,
.attention-panel,
.prompt-result,
.head-detail {
  padding-bottom: 10px;
}

.layer-path-row {
  justify-content: space-between;
  gap: 10px;
  padding: 9px 12px 0;
}

.layer-path-row code {
  overflow: hidden;
  color: var(--tv-ink);
  font: 650 9px/1.3 var(--font-mono, monospace);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.layer-path-row span {
  flex: 0 0 auto;
  color: var(--tv-muted);
  font-size: 8px;
}

.component-flow {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 5px;
  margin: 9px 12px 0;
  padding: 0;
  list-style: none;
}

.component-flow li {
  min-width: 0;
  padding: 7px;
  border: 1px solid var(--tv-border);
  border-radius: 5px;
  background: var(--tv-surface-soft);
}

.component-flow li:not(:last-child) {
  position: relative;
}

.component-flow li:not(:last-child)::after {
  position: absolute;
  z-index: 1;
  top: 50%;
  right: -7px;
  color: var(--tv-border-strong);
  content: "›";
  transform: translateY(-52%);
}

.component-flow span,
.component-flow code {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.component-flow span {
  color: var(--tv-ink);
  font-size: 9px;
  font-weight: 650;
}

.component-flow code {
  margin-top: 2px;
  color: var(--tv-muted);
  font: 7px/1.2 var(--font-mono, monospace);
}

.component-flow__empty {
  grid-column: 1 / -1;
}

.detail-stats,
.head-stats {
  display: grid;
  gap: 1px;
  margin: 9px 12px 0;
  overflow: hidden;
  border: 1px solid var(--tv-border);
  border-radius: 5px;
  background: var(--tv-border);
}

.detail-stats {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.detail-stats > div,
.head-stats > div {
  min-width: 0;
  padding: 7px 8px;
  background: var(--tv-surface);
}

.detail-stats dt,
.head-stats dt {
  overflow: hidden;
  color: var(--tv-muted);
  font: 8px/1.2 var(--font-mono, monospace);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.detail-stats dd,
.head-stats dd {
  overflow: hidden;
  margin: 2px 0 0;
  color: var(--tv-ink);
  font: 650 10px/1.25 var(--font-mono, monospace);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.head-selector-heading {
  justify-content: space-between;
  gap: 10px;
  margin: 11px 12px 0;
}

.head-selector-heading > div {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.head-selector-heading strong {
  font-size: 10px;
}

.head-selector-heading > span {
  color: var(--tv-muted);
  font: 8px/1.2 var(--font-mono, monospace);
}

.head-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 5px;
  margin: 7px 12px 0;
}

.head-option {
  display: flex;
  min-width: 0;
  min-height: 39px;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  padding: 4px;
  border: 1px solid var(--tv-border);
  border-radius: 5px;
  background: var(--tv-surface-soft);
  color: var(--tv-ink);
  cursor: pointer;
}

.head-option:hover {
  border-color: var(--tv-border-strong);
  background: var(--tv-surface);
}

.head-option--selected {
  border-color: var(--tv-primary);
  background: var(--tv-primary-soft);
  color: var(--tv-primary-dark);
}

.head-option__label {
  font: 700 9px/1 var(--font-mono, monospace);
}

.head-option__metric {
  max-width: 100%;
  overflow: hidden;
  color: var(--tv-muted);
  font: 7px/1.1 var(--font-mono, monospace);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.head-option__metric--empty {
  color: var(--tv-subtle);
}

.inline-loading,
.inline-message {
  display: flex;
  align-items: center;
  gap: 9px;
  margin: 10px 12px;
  padding: 12px;
  border: 1px solid var(--tv-border);
  border-radius: 6px;
  background: var(--tv-surface-soft);
  color: var(--tv-muted);
  font-size: 10px;
}

.inline-message > div {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.inline-message strong {
  color: var(--tv-ink);
  font-size: 10px;
}

.inline-message--error {
  border-color: #efc2c7;
  background: var(--danger-soft, #fff0f1);
  color: var(--danger, #bf3f4d);
}

.compact-empty {
  margin: 10px 12px;
  padding: 13px;
  border: 1px dashed var(--tv-border-strong);
  border-radius: 6px;
  color: var(--tv-muted);
  font-size: 10px;
  line-height: 1.45;
  text-align: center;
}

.matrix-meta {
  flex-wrap: wrap;
  gap: 5px 10px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--tv-border);
  color: var(--tv-muted);
  font-size: 8px;
}

.matrix-scroll {
  max-height: 440px;
  margin: 9px 12px;
  overflow: auto;
  border: 1px solid var(--tv-border);
  border-radius: 6px;
  background: var(--tv-surface-soft);
}

.attention-matrix {
  width: max-content;
  min-width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  table-layout: fixed;
  font: 8px/1 var(--font-mono, monospace);
}

.attention-matrix th,
.attention-matrix td {
  width: var(--matrix-cell);
  min-width: var(--matrix-cell);
  height: var(--matrix-cell);
  padding: 0;
  border-right: 1px solid rgba(148, 163, 184, 0.22);
  border-bottom: 1px solid rgba(148, 163, 184, 0.22);
  text-align: center;
}

.attention-matrix th {
  position: sticky;
  z-index: 2;
  background: #f1f5f9;
  color: var(--tv-muted);
  font-weight: 650;
}

.attention-matrix thead th {
  top: 0;
}

.attention-matrix th:first-child {
  z-index: 3;
  left: 0;
  width: calc(var(--matrix-cell) * 1.8);
  min-width: calc(var(--matrix-cell) * 1.8);
}

.attention-matrix th span,
.attention-matrix td {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.attention-matrix th span {
  padding: 3px;
}

.attention-matrix td {
  color: var(--tv-ink);
}

.matrix-prompt {
  margin: 0 12px;
  color: var(--tv-muted);
  font-size: 8px;
}

.empty-attention {
  display: flex;
  min-height: 175px;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 5px;
  margin: 10px 12px;
  padding: 20px;
  border: 1px dashed var(--tv-border-strong);
  border-radius: 7px;
  background: var(--tv-surface-soft);
  text-align: center;
}

.empty-attention__mark {
  margin-bottom: 3px;
}

.result-summary {
  justify-content: space-between;
  gap: 8px;
  padding: 9px 12px 0;
  color: var(--tv-muted);
  font-size: 8px;
}

.next-token {
  justify-content: space-between;
  gap: 8px;
  margin: 9px 12px 0;
  padding: 9px 10px;
  border: 1px solid #b9ccef;
  border-radius: 6px;
  background: #f5f8ff;
}

.next-token span {
  color: var(--tv-muted);
  font: 8px/1 var(--font-mono, monospace);
  text-transform: uppercase;
}

.next-token code {
  overflow: hidden;
  color: var(--tv-primary-dark);
  font: 700 13px/1.2 var(--font-mono, monospace);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.token-strip {
  display: flex;
  gap: 4px;
  margin: 8px 12px 0;
  padding: 0 0 3px;
  overflow-x: auto;
}

.token-strip span {
  flex: 0 0 auto;
  max-width: 100px;
  overflow: hidden;
  padding: 3px 5px;
  border: 1px solid var(--tv-border);
  border-radius: 4px;
  background: var(--tv-surface-soft);
  color: var(--tv-ink);
  font: 8px/1.2 var(--font-mono, monospace);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.prediction-list {
  display: flex;
  flex-direction: column;
  gap: 1px;
  margin: 9px 12px 0;
  padding: 0;
  overflow: hidden;
  border: 1px solid var(--tv-border);
  border-radius: 6px;
  background: var(--tv-border);
  list-style: none;
}

.prediction-list li {
  display: grid;
  grid-template-columns: 24px minmax(0, 1fr) auto;
  gap: 6px;
  align-items: center;
  padding: 6px 7px;
  background: var(--tv-surface);
  color: var(--tv-muted);
  font: 8px/1.2 var(--font-mono, monospace);
}

.prediction-list li > span:last-child {
  grid-column: 3;
}

.prediction-list code {
  overflow: hidden;
  color: var(--tv-ink);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.provenance-note--right {
  margin: 9px 12px 0;
}

.head-stats {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.top-attention {
  margin: 9px 12px 0;
  padding: 9px;
  border: 1px solid var(--tv-border);
  border-radius: 6px;
  background: var(--tv-surface-soft);
}

.top-attention > span {
  color: var(--tv-muted);
  font: 8px/1 var(--font-mono, monospace);
  text-transform: uppercase;
}

.top-attention > div {
  justify-content: space-between;
  gap: 8px;
  margin-top: 4px;
}

.top-attention code {
  overflow: hidden;
  color: var(--tv-ink);
  font-size: 10px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.top-attention strong {
  color: var(--tv-primary-dark);
  font: 700 10px/1 var(--font-mono, monospace);
}

.source-caption {
  margin: 7px 12px 0;
  color: var(--tv-muted);
  font-size: 8px;
  line-height: 1.4;
}

.visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  margin: -1px;
  padding: 0;
  border: 0;
  clip: rect(0 0 0 0);
  white-space: nowrap;
}

.spinning {
  animation: tv-spin 0.9s linear infinite;
}

@keyframes tv-spin {
  to { transform: rotate(360deg); }
}

@keyframes tv-pulse {
  0%, 100% { opacity: .55; }
  50% { opacity: 1; }
}

@media (max-width: 1240px) {
  .workspace-grid {
    grid-template-columns: minmax(170px, 0.42fr) minmax(440px, 1.5fr);
  }

  .right-rail {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .architecture-stats {
    grid-template-columns: repeat(5, minmax(74px, 1fr));
  }
}

@media (max-width: 900px) {
  .visualizer-header {
    align-items: flex-start;
    flex-direction: column;
  }

  .header-actions {
    width: 100%;
    justify-content: space-between;
  }

  .prompt-command {
    grid-template-columns: minmax(0, 1fr) auto;
  }

  .prompt-command__label {
    grid-column: 1 / -1;
  }

  .inline-error {
    grid-column: 1 / -1;
  }

  .architecture-flow {
    grid-template-columns: 1fr;
  }

  .flow-arrow {
    min-height: 17px;
    transform: rotate(90deg);
  }

  .workspace-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .layer-panel {
    position: static;
  }

  .layer-list {
    max-height: 190px;
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .right-rail {
    grid-column: auto;
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 620px) {
  .transformer-visualizer {
    padding: 10px;
  }

  .visualizer-header h1 {
    font-size: 21px;
  }

  .visualizer-header p {
    line-height: 1.45;
  }

  .title-mark {
    width: 36px;
    height: 36px;
  }

  .prompt-command {
    grid-template-columns: minmax(0, 1fr);
  }

  .prompt-command__actions {
    align-items: stretch;
  }

  .prompt-command__actions > span {
    align-self: flex-end;
  }

  .run-button {
    width: 100%;
  }

  .architecture-stats {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .residual-strip {
    align-items: flex-start;
    flex-direction: column;
    gap: 3px;
  }

  .layer-list {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .component-flow {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .component-flow li::after {
    display: none;
  }

  .detail-stats {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .head-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .layer-path-row {
    align-items: flex-start;
    flex-direction: column;
  }

  .message-panel {
    align-items: flex-start;
    flex-direction: column;
  }
}

@media (prefers-reduced-motion: reduce) {
  .spinning,
  .skeleton-grid span {
    animation: none;
  }
}
</style>
