<template>
  <section class="lens-surface" data-testid="logit-lens">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Explore / Projection</p>
        <h2 class="surface-title">Logit Lens</h2>
        <p class="surface-description">
          Unembeds the residual stream recorded after each transformer block
          (<code>POST /api/gpt2/logit_lens_all</code>) to show what the model <em>would</em> predict
          if it stopped there. The model does not stop there.
        </p>
      </div>
      <form class="run-form" @submit.prevent="runLens">
        <label class="field-label" for="logit-lens-prompt">Prompt</label>
        <input
          id="logit-lens-prompt"
          v-model="prompt"
          class="text-input"
          type="text"
          autocomplete="off"
          spellcheck="false"
          data-testid="logitlens-prompt"
          :disabled="isRunning"
        />
        <button class="primary-button" type="submit" data-testid="logitlens-run" :disabled="isRunning || !prompt.trim()">
          {{ isRunning ? 'Projecting…' : 'Run logit lens' }}
        </button>
      </form>
    </header>

    <p v-if="stale" class="notice notice-warn" role="status" data-testid="logitlens-stale">
      <strong>Stale result.</strong>
      The rows below are for “{{ result?.prompt }}”; the prompt field now reads
      “{{ prompt.trim() }}”. Run again before treating them as the current prompt's projection.
    </p>

    <div
      v-if="failure"
      class="notice"
      :class="failure.kind === 'unsupported' ? 'notice-unsupported' : 'notice-error'"
      role="alert"
      data-testid="logitlens-failure"
    >
      <strong>{{ failure.kind === 'unsupported' ? 'NOT AVAILABLE' : 'Projection failed' }}</strong>
      <p>{{ failure.message }}</p>
      <p v-if="failure.kind === 'unsupported'" class="notice-foot">
        An empty top-k list would read as a distribution of zero probability; the backend returned
        no projection, so no rows are drawn.
      </p>
    </div>

    <div v-if="isRunning" class="notice" role="status" aria-live="polite" data-testid="logitlens-loading">
      <strong>Running the projection…</strong>
      <p>The backend runs one cached forward pass and unembeds every layer.</p>
    </div>

    <p v-if="!result && !failure && !isRunning" class="notice" data-testid="logitlens-idle">
      <strong>Idle.</strong> Nothing is projected until you ask. This surface never runs a forward
      pass on mount.
    </p>

    <template v-if="result">
      <section class="panel" aria-labelledby="lens-caveat-title" data-testid="logitlens-projection-note">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Read this first</p>
            <h3 id="lens-caveat-title">A projection, not the model's finished computation</h3>
          </div>
          <span class="source-badge">{{ methodLabel }}</span>
        </header>
        <p class="caveat-copy">
          Each row applies the model's final layer norm and the tied unembedding matrix to the
          residual stream recorded <em>after</em> that block. That is what the model would have
          predicted had it stopped at that layer — GPT-2 keeps computing, so a confident-looking
          early row is not the model's answer.
        </p>
        <p v-if="methodLabel !== 'PROJECTION NOT RETURNED'" class="caveat-copy">
          Projection convention returned by the backend: <code>{{ methodLabel }}</code> — a raw
          logit-lens projection (no trained translator / affine probe).
        </p>
          <p class="caveat-copy">
            Only the returned entries per layer are shown. Each layer's own truncation contract
            (how many entries of how many vocabulary entries were returned, and the probability mass
            they account for) is printed with that layer — a truncated list is not a distribution
            over the vocabulary, so do not sum it and call the total 1.
          </p>
      </section>

      <section class="panel" aria-labelledby="lens-layers-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Per-layer top-k</p>
            <h3 id="lens-layers-title">What each layer projects to</h3>
            <p class="panel-description">
              {{ layerRows.length }} layer row(s) returned for
              “{{ asString(result.prompt) || prompt.trim() }}”.
            </p>
          </div>
          <span class="source-badge">{{ provenance || 'PROVENANCE NOT RETURNED' }}</span>
        </header>

        <p v-if="layerRows.length === 0" class="panel-message" role="status" data-testid="logitlens-empty">
          The backend returned no layer rows. An empty projection is not a projection of zero
          probability.
        </p>

        <p v-else-if="emptyLayerCount > 0" class="notice notice-warn" role="status" data-testid="logitlens-partial">
          <strong>Partial result.</strong>
          {{ emptyLayerCount }} layer row(s) carried no top-k tokens. Those rows read “not returned”
          rather than an empty prediction.
        </p>

        <ol v-if="layerRows.length" class="layer-list" data-testid="logitlens-layers">
          <li
            v-for="(row, index) in layerRows"
            :key="`${row.layer}-${index}`"
            class="layer-card"
            :data-testid="`logitlens-layer-${row.layer}`"
          >
            <div class="layer-head">
              <span class="layer-index">L{{ row.layer }}</span>
              <span v-if="row.topToken" class="top-token mono">«{{ row.topToken }}»</span>
              <span v-else class="top-token muted">top token not returned</span>
              <span v-if="row.topProbability !== null" class="top-prob mono">
                p = {{ formatNumber(row.topProbability, 6) }}
              </span>
            </div>
            <ol v-if="row.tokens.length" class="token-row" :aria-label="`Returned top-k tokens for layer ${row.layer}`">
              <li v-for="(token, tokenIndex) in row.tokens" :key="`${row.layer}-${tokenIndex}`">
                <span class="token-text mono">«{{ token.token }}»</span>
                <span class="token-prob mono">p {{ formatNumber(token.probability, 6) }}</span>
                <span v-if="token.logit !== null" class="token-logit mono">logit {{ formatNumber(token.logit, 4) }}</span>
                <span v-if="token.rank !== null" class="token-rank mono">rank {{ formatInteger(token.rank) }}</span>
                <span v-else class="token-rank muted">rank not returned</span>
              </li>
            </ol>
            <p v-else class="layer-empty">This layer's response carried no <code>top_k_tokens</code>.</p>
            <footer class="layer-contract" :data-testid="`logitlens-contract-${row.layer}`">
              <p class="mono">
                probabilities: {{ row.reported === null ? '—' : formatInteger(row.reported) }}
                of {{ row.total === null ? '—' : formatInteger(row.total) }} vocabulary entries ·
                reported mass {{ formatNumber(row.reportedMass, 6) }} ·
                truncated: {{ row.truncated === null ? 'not returned' : row.truncated ? 'yes' : 'no' }}
              </p>
              <p class="layer-contract-note">{{ truncationNote(row) }}</p>
            </footer>
          </li>
        </ol>
      </section>

      <section class="panel" aria-labelledby="lens-trajectory-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Across layers</p>
            <h3 id="lens-trajectory-title">Rank trajectory</h3>
            <p class="panel-description" v-if="hasRankData">
              Vocabulary rank per layer for the tracked token, as returned by the backend (1 = the
              model's top prediction at that layer).
            </p>
          </div>
        </header>

        <div v-if="hasRankData" data-testid="logitlens-trajectory">
          <label class="field-label" for="lens-tracked-token">Tracked token</label>
          <select
            id="lens-tracked-token"
            v-model="trackedToken"
            class="text-input select-input"
            data-testid="logitlens-tracked-token"
          >
            <option v-for="token in trackableTokens" :key="token" :value="token">«{{ token }}»</option>
          </select>

          <figure class="trajectory-figure">
            <svg
              class="trajectory-chart"
              :viewBox="`0 0 ${chart.width} ${chart.height}`"
              role="img"
              :aria-label="chartLabel"
              data-testid="logitlens-trajectory-chart"
            >
              <line
                v-for="tick in chart.ticks"
                :key="`tick-${tick.rank}`"
                :x1="chart.left"
                :x2="chart.right"
                :y1="tick.y"
                :y2="tick.y"
                class="axis-line"
              />
              <text
                v-for="tick in chart.ticks"
                :key="`label-${tick.rank}`"
                :x="chart.left - 6"
                :y="tick.y + 3"
                class="axis-label"
                text-anchor="end"
              >
                {{ tick.rank }}
              </text>
              <text
                v-for="tick in chart.layerTicks"
                :key="`layer-${tick.layer}`"
                :x="tick.x"
                :y="chart.height - 4"
                class="axis-label"
                text-anchor="middle"
              >
                L{{ tick.layer }}
              </text>
              <template v-for="segment in chart.segments" :key="segment.id">
                <polyline :points="segment.points" class="trajectory-line" fill="none" />
              </template>
              <circle
                v-for="point in chart.points"
                :key="`point-${point.layer}`"
                :cx="point.x"
                :cy="point.y"
                r="3"
                class="trajectory-point"
              />
            </svg>
            <figcaption class="panel-footnote">
              A gap means the token was not among that layer's returned top-k entries; the line is
              not interpolated across a gap.
            </figcaption>
          </figure>

          <table class="rank-table" data-testid="logitlens-rank-table">
            <caption>
              Exact returned values for «{{ activeTrackedToken }}». “—” means the token was not returned
              in that layer's top-k, so its rank is unknown — not zero and not infinity.
            </caption>
            <thead>
              <tr>
                <th scope="col">Layer</th>
                <th scope="col">Rank</th>
                <th scope="col">Probability</th>
                <th scope="col">Logit</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="point in trajectoryRows" :key="`row-${point.layer}`">
                <th scope="row">L{{ point.layer }}</th>
                <td class="mono">{{ point.rank === null ? '—' : formatInteger(point.rank) }}</td>
                <td class="mono">{{ formatNumber(point.probability, 6) }}</td>
                <td class="mono">{{ formatNumber(point.logit, 4) }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div v-else class="unavailable-block" role="note" data-testid="logitlens-rank-unavailable">
          <strong>NOT AVAILABLE — vocabulary rank trajectory</strong>
          <p>
            This response's tokens carry no <code>rank</code> field, so no per-layer vocabulary rank
            exists to plot. <code>POST /api/gpt2/logit_lens_all</code> returned each layer's top-k
            tokens and probabilities only.
          </p>
          <p>
            The array order inside <code>top_k_tokens</code> is a returned ordering, not a vocabulary
            rank, so this view does not relabel it as one. No trajectory is drawn.
          </p>
        </div>
      </section>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import {
  asArray,
  asBoolean,
  asNumber,
  asString,
  describeError,
  formatInteger,
  formatNumber,
  gpt2LogitLensAll,
  isRecord,
  isUnsupportedError,
  provenanceOf,
  statusOf,
  type JsonRecord,
} from '../services/interpretabilityClient';

interface LensToken {
  token: string;
  probability: number | null;
  rank: number | null;
  logit: number | null;
}

interface LensLayer {
  layer: number | null;
  topToken: string;
  topProbability: number | null;
  tokens: LensToken[];
  /** Truncation contract, as returned per layer by the backend. */
  truncated: boolean | null;
  reported: number | null;
  total: number | null;
  reportedMass: number | null;
}

interface Failure {
  kind: 'unsupported' | 'error';
  message: string;
}

const prompt = ref('The capital of France is');
const isRunning = ref(false);
const result = ref<JsonRecord | null>(null);
const failure = ref<Failure | null>(null);
const trackedToken = ref('');
let runSequence = 0;

const stale = computed(() => Boolean(result.value) && asString(result.value?.prompt) !== prompt.value.trim());
const provenance = computed(() => provenanceOf(result.value));
const methodLabel = computed(() => asString(result.value?.method) || 'PROJECTION NOT RETURNED');

const layerRows = computed<LensLayer[]>(() => {
  const rows = asArray(result.value?.layers).filter(isRecord);
  return rows.map((row) => {
    const tokens = asArray(row.top_k_tokens)
      .filter(isRecord)
      .map((token) => ({
        token: asString(token.token),
        probability: asNumber(token.prob),
        rank: asNumber(token.rank),
        logit: asNumber(token.logit),
      }));
    const topToken = asString(row.top_token) || tokens[0]?.token || '';
    const topProbability = asNumber(row.top_token_prob) ?? tokens[0]?.probability ?? null;
    return {
      layer: asNumber(row.layer),
      topToken,
      topProbability,
      tokens,
      truncated: asBoolean(row.probability_is_truncated),
      reported: asNumber(row.probabilities_reported),
      total: asNumber(row.probabilities_total),
      reportedMass: asNumber(row.probabilities_reported_mass),
    };
  });
});

/** One honest sentence per layer about whether the returned list is the whole distribution. */
function truncationNote(row: LensLayer): string {
  if (row.truncated === true) {
    return 'Truncated: these are the top entries only; their probabilities sum to the reported mass below, not to 1.';
  }
  if (row.truncated === false) {
    return 'Not truncated: this layer returned the full distribution.';
  }
  return 'The response did not say whether this list is truncated; the probabilities shown are those of the returned entries only.';
}

const emptyLayerCount = computed(() => layerRows.value.filter((row) => row.tokens.length === 0).length);

const hasRankData = computed(() => layerRows.value.some((row) => row.tokens.some((token) => token.rank !== null)));

const trackableTokens = computed<string[]>(() => {
  const seen: string[] = [];
  for (const row of layerRows.value) {
    for (const token of row.tokens) {
      if (token.rank === null || !token.token) continue;
      if (!seen.includes(token.token)) seen.push(token.token);
    }
  }
  return seen.slice(0, 12);
});

const activeTrackedToken = computed(() => {
  if (trackedToken.value && trackableTokens.value.includes(trackedToken.value)) return trackedToken.value;
  return trackableTokens.value[0] ?? '';
});

const trajectoryRows = computed(() =>
  layerRows.value.map((row) => {
    const match = row.tokens.find((token) => token.token === activeTrackedToken.value) ?? null;
    return {
      layer: row.layer,
      rank: match?.rank ?? null,
      probability: match?.probability ?? null,
      logit: match?.logit ?? null,
    };
  }),
);

const trajectoryPoints = computed(() =>
  trajectoryRows.value.filter((row): row is { layer: number; rank: number; probability: number | null; logit: number | null } =>
    row.layer !== null && row.rank !== null,
  ),
);

const chart = computed(() => {
  const width = 520;
  const height = 160;
  const left = 40;
  const right = width - 14;
  const top = 14;
  const bottom = height - 22;
  const rows = trajectoryRows.value.filter((row) => row.layer !== null);
  const ranks = trajectoryPoints.value.map((point) => point.rank);
  const maxRank = Math.max(5, ...ranks);
  const layerCount = Math.max(1, rows.length - 1);

  const xFor = (index: number) => left + (index / layerCount) * (right - left);
  const yFor = (rank: number) => bottom - ((maxRank - rank) / (maxRank - 1 || 1)) * (bottom - top);

  const points = rows.map((row, index) => ({
    layer: row.layer as number,
    x: xFor(index),
    y: row.rank === null ? bottom : yFor(row.rank),
  }));

  const segments: Array<{ id: string; points: string }> = [];
  let current: string[] = [];
  rows.forEach((row, index) => {
    if (row.rank === null) {
      if (current.length > 1) segments.push({ id: `seg-${segments.length}`, points: current.join(' ') });
      current = [];
      return;
    }
    current.push(`${xFor(index)},${yFor(row.rank)}`);
  });
  if (current.length > 1) segments.push({ id: `seg-${segments.length}`, points: current.join(' ') });

  const ticks = [1, Math.round(maxRank / 2), maxRank]
    .filter((rank, index, list) => list.indexOf(rank) === index)
    .map((rank) => ({ rank, y: yFor(rank) }));

  return {
    width,
    height,
    left,
    right,
    points: points.filter((point, index) => rows[index]?.rank !== null),
    segments,
    ticks,
    layerTicks: rows
      .map((row, index) => ({ layer: row.layer as number, x: xFor(index) }))
      .filter((tick, index) => index === 0 || index === rows.length - 1),
  };
});

const chartLabel = computed(() => {
  const descriptions = trajectoryPoints.value.map((point) => `layer ${point.layer} rank ${point.rank}`);
  return `Vocabulary rank of “${activeTrackedToken.value}” per layer: ${
    descriptions.length ? descriptions.join(', ') : 'no ranked layers returned'
  }.`;
});

async function runLens() {
  const targetPrompt = prompt.value.trim();
  if (!targetPrompt) return;
  const sequence = (runSequence += 1);
  isRunning.value = true;
  failure.value = null;
  result.value = null;
  trackedToken.value = '';
  try {
    const response = await gpt2LogitLensAll({ prompt: targetPrompt });
    if (sequence !== runSequence) return;
    if (statusOf(response) !== 'ok') {
      failure.value = {
        kind: statusOf(response) === 'unavailable' ? 'unsupported' : 'error',
        message:
          asString(response.error) ||
          asString(response.reason) ||
          `The backend returned status “${statusOf(response) || 'none'}” without a reason.`,
      };
      return;
    }
    result.value = response;
    trackedToken.value = defaultTrackedToken(response);
  } catch (error) {
    if (sequence !== runSequence) return;
    failure.value = {
      kind: isUnsupportedError(error) ? 'unsupported' : 'error',
      message: describeError(error),
    };
  } finally {
    if (sequence === runSequence) isRunning.value = false;
  }
}

/** Prefer the token with the best rank at the last returned layer. */
function defaultTrackedToken(response: JsonRecord): string {
  const rows = asArray(response.layers).filter(isRecord);
  const rankedIn = (row: JsonRecord) =>
    asArray(row.top_k_tokens)
      .filter(isRecord)
      .map((token) => ({ token: asString(token.token), rank: asNumber(token.rank) }))
      .filter((token) => token.token && token.rank !== null)
      .sort((a, b) => (a.rank as number) - (b.rank as number));

  const last = rows[rows.length - 1];
  if (last) {
    const ranked = rankedIn(last);
    if (ranked.length) return ranked[0].token;
  }
  // A response may rank some layers and not others; fall back to the last row
  // that actually carried a rank rather than leaving the selector blank.
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const ranked = rankedIn(rows[index]);
    if (ranked.length) return ranked[0].token;
  }
  return '';
}
</script>

<style scoped>
.lens-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.run-form,
.panel-heading,
.layer-head,
.token-row,
.token-row li {
  display: flex;
  gap: 10px;
}

.surface-header,
.panel-heading,
.layer-head {
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
.caveat-copy {
  margin: 7px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.55;
}

.caveat-copy {
  color: var(--text-dim);
}

.run-form {
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

.select-input {
  flex: 0 1 260px;
  margin-top: 4px;
}

.primary-button {
  min-height: 32px;
  padding: 0 11px;
  border: 1px solid var(--primary);
  border-radius: 6px;
  background: var(--primary);
  color: #ffffff;
  cursor: pointer;
  font: 700 11px/1 var(--font);
}

.primary-button:disabled {
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

.layer-list {
  display: grid;
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.layer-card {
  padding: 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
}

.layer-head {
  align-items: center;
  justify-content: flex-start;
}

.layer-index {
  min-width: 34px;
  color: var(--text-muted);
  font: 700 11px/1 var(--font-mono);
}

.top-token {
  font-size: 13px;
  font-weight: 600;
}

.top-prob,
.token-prob {
  color: var(--text-dim);
  font-size: 11px;
}

.token-row {
  margin: 8px 0 0;
  padding: 0;
  gap: 6px;
  flex-wrap: wrap;
  list-style: none;
}

.token-row li {
  align-items: baseline;
  gap: 6px;
  padding: 4px 7px;
  border: 1px solid var(--border);
  border-radius: 5px;
  background: var(--surface);
  font-size: 11px;
}

.token-text {
  font-weight: 600;
}

.token-rank {
  color: var(--text-muted);
  font-size: 10px;
}

.layer-empty,
.muted {
  margin: 6px 0 0;
  color: var(--text-muted);
  font-size: 11px;
}

.layer-contract {
  margin-top: 9px;
  padding-top: 7px;
  border-top: 1px solid var(--border);
}

.layer-contract p {
  margin: 0;
  color: var(--text-muted);
  font-size: 10px;
  line-height: 1.5;
}

.layer-contract-note {
  margin-top: 3px !important;
  color: var(--text-dim) !important;
}

.token-logit {
  color: var(--text-muted);
  font-size: 10px;
}

.mono {
  font-family: var(--font-mono);
}

.trajectory-figure {
  margin: 10px 0 0;
}

.trajectory-chart {
  width: 100%;
  max-width: 560px;
  height: auto;
  overflow: visible;
}

.axis-line {
  stroke: var(--border);
  stroke-width: 1;
}

.axis-label {
  fill: var(--text-muted);
  font: 500 9px var(--font-mono);
}

.trajectory-line {
  stroke: var(--primary);
  stroke-width: 2;
  stroke-linejoin: round;
}

.trajectory-point {
  fill: var(--primary);
}

.rank-table {
  width: 100%;
  margin-top: 10px;
  border-collapse: collapse;
  font-size: 11px;
}

.rank-table caption {
  caption-side: top;
  padding-bottom: 7px;
  color: var(--text-muted);
  font: 400 11px/1.5 var(--font);
  text-align: left;
}

.rank-table th,
.rank-table td {
  padding: 5px 7px;
  border-bottom: 1px solid var(--border);
  text-align: left;
}

.rank-table thead th {
  color: var(--text-muted);
  font: 700 9px/1.4 var(--font-mono);
  letter-spacing: 0.05em;
  text-transform: uppercase;
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

.unavailable-block p:last-child {
  margin-bottom: 0;
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

.panel-footnote {
  margin: 6px 0 0;
  color: var(--text-muted);
  font-size: 10px;
  line-height: 1.5;
}
</style>
