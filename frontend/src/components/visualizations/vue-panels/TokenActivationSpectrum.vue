<template>
  <section class="spectrum" :aria-labelledby="titleId">
    <header class="spectrum-head">
      <h3 :id="titleId">{{ title }}</h3>
      <span v-if="!loading && entries.length" class="spectrum-note">sorted by |activation|</span>
    </header>

    <div v-if="loading" class="spectrum-state" role="status" aria-live="polite">
      <span class="spectrum-spinner" aria-hidden="true"></span>
      <strong>Loading token activations…</strong>
    </div>

    <div v-else-if="!tokens.length" class="spectrum-state" role="status">
      <strong>No tokens to summarise</strong>
      <span>Run a prompt to get a token list for this neuron.</span>
    </div>

    <div v-else-if="!hasActivationData || !entries.length" class="spectrum-state" role="status">
      <strong>No token activations reported</strong>
      <span>This neuron returned no per-token activation values.</span>
    </div>

    <template v-else>
      <p :id="instructionsId" class="sr-only">
        Token activation spectrum. Each row pairs a token with its activation value. Bars extend right for
        positive activations and left for negative ones, scaled against the largest absolute value shown.
      </p>

      <ul class="spectrum-list" :aria-describedby="instructionsId">
        <li v-for="entry in entries" :key="entry.index" class="spectrum-row">
          <span class="spectrum-token" :title="entry.token">{{ entry.label }}</span>
          <span class="spectrum-track" aria-hidden="true">
            <span class="spectrum-zero"></span>
            <span
              v-if="entry.activation > 0"
              class="spectrum-fill positive"
              :style="{ left: '50%', width: `${entry.widthPercent}%` }"
            ></span>
            <span
              v-else-if="entry.activation < 0"
              class="spectrum-fill negative"
              :style="{ right: '50%', width: `${entry.widthPercent}%` }"
            ></span>
          </span>
          <span class="spectrum-value" :class="tone(entry.activation)">{{ formatValue(entry.activation) }}</span>
        </li>
      </ul>

      <footer class="spectrum-foot">
        <span>Showing {{ entries.length }} of {{ tokens.length }} token{{ tokens.length === 1 ? '' : 's' }}</span>
        <span class="spectrum-legend">
          <i class="dot positive" aria-hidden="true"></i>positive
          <i class="dot negative" aria-hidden="true"></i>negative
        </span>
      </footer>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, useId } from 'vue';

interface SpectrumEntry {
  index: number;
  token: string;
  label: string;
  activation: number;
  ratio: number;
  /** Half-track width as a percentage, rounded to keep inline styles readable. */
  widthPercent: number;
}

const LABEL_MAX_CHARS = 10;

const props = withDefaults(defineProps<{
  tokens: string[];
  activations: number[];
  maxTokens?: number;
  loading?: boolean;
  title?: string;
}>(), {
  tokens: () => [],
  activations: () => [],
  maxTokens: 50,
  loading: false,
  title: 'Token Activation Spectrum',
});

const titleId = `spectrum-${useId()}`;
const instructionsId = `${titleId}-instructions`;

const loading = computed(() => props.loading === true);

/** Mirrors the `Ġ`/`Ċ` convention used elsewhere in the explorer. */
function displayToken(token: string, maxChars = LABEL_MAX_CHARS): string {
  const cleaned = String(token ?? '')
    .replaceAll('\u0120', '\u2423')
    .replaceAll('\u010a', '\u23ce')
    .trim();
  if (!cleaned) return '(empty)';
  return cleaned.length > maxChars ? `${cleaned.slice(0, maxChars)}…` : cleaned;
}

const ranked = computed(() => props.tokens.map((token, index) => {
  const raw = props.activations[index];
  const supplied = typeof raw === 'number' && Number.isFinite(raw);
  return {
    index,
    token: String(token ?? ''),
    activation: supplied ? (raw as number) : 0,
    supplied,
  };
}));

/**
 * Distinguishes "every token scored 0" from "no scores were supplied at all".
 * Without this the panel would render a full list of 0.000 rows for a neuron
 * that simply has no per-token data.
 */
const hasActivationData = computed(() => ranked.value.some(entry => entry.supplied));

const maxTokens = computed(() => Math.max(0, Math.floor(props.maxTokens)));

const entries = computed<SpectrumEntry[]>(() => {
  if (!maxTokens.value) return [];

  // Strongest first, ties broken by token order so the list is stable.
  const ordered = [...ranked.value].sort(
    (a, b) => Math.abs(b.activation) - Math.abs(a.activation) || a.index - b.index,
  );

  const peak = ordered.reduce((maximum, entry) => Math.max(maximum, Math.abs(entry.activation)), 0);
  const scale = peak > 0 ? peak : 0.01;

  return ordered.slice(0, maxTokens.value).map(entry => {
    const ratio = Math.min(1, Math.abs(entry.activation) / scale);
    return {
      ...entry,
      label: displayToken(entry.token),
      ratio,
      widthPercent: Math.round(ratio * 5000) / 100,
    };
  });
});

function formatValue(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : '—';
}

function tone(value: number): 'positive' | 'negative' | 'neutral' {
  if (value > 0) return 'positive';
  if (value < 0) return 'negative';
  return 'neutral';
}
</script>

<style scoped>
.spectrum {
  --border: #d9dee8;
  --text: #20283a;
  --muted: #667085;
  --positive: #2563eb;
  --negative: #0e9384;
  display: grid;
  gap: 7px;
  min-width: 0;
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}

.spectrum-head {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 6px;
}

.spectrum-head h3 {
  margin: 0;
  color: var(--muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.spectrum-note {
  color: var(--muted);
  font-size: 9px;
}

.spectrum-list {
  display: grid;
  gap: 3px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.spectrum-row {
  display: grid;
  grid-template-columns: minmax(52px, 78px) minmax(90px, 1fr) 46px;
  align-items: center;
  gap: 8px;
  min-height: 20px;
}

.spectrum-token {
  overflow: hidden;
  color: #30384a;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 9px;
  text-align: right;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.spectrum-track {
  position: relative;
  display: block;
  height: 20px;
  background: #f1f4f9;
  border: 1px solid var(--border);
  border-radius: 3px;
}

.spectrum-zero {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 1px;
  background: #c7cdd9;
}

.spectrum-fill {
  position: absolute;
  top: 1px;
  bottom: 1px;
  border-radius: 2px;
}

.spectrum-fill.positive {
  background: var(--positive);
}

.spectrum-fill.negative {
  background: var(--negative);
}

.spectrum-value {
  font-size: 9px;
  font-variant-numeric: tabular-nums;
  text-align: right;
}

.spectrum-value.positive {
  color: var(--positive);
}

.spectrum-value.negative {
  color: var(--negative);
}

.spectrum-value.neutral {
  color: var(--muted);
}

.spectrum-foot {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  color: var(--muted);
  font-size: 9px;
}

.spectrum-legend {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.spectrum-legend .dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  margin-left: 6px;
  border-radius: 50%;
}

.spectrum-legend .dot:first-child {
  margin-left: 0;
}

.dot.positive {
  background: var(--positive);
}

.dot.negative {
  background: var(--negative);
}

.spectrum-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 18px 14px;
  color: var(--muted);
  background: #ffffff;
  border: 1px dashed var(--border);
  border-radius: 8px;
  text-align: center;
  font-size: 10px;
}

.spectrum-state strong {
  color: var(--text);
  font-size: 11px;
}

.spectrum-spinner {
  width: 18px;
  height: 18px;
  border: 2px solid #d9d6ef;
  border-top-color: #2563eb;
  border-radius: 50%;
  animation: spectrum-spin 0.8s linear infinite;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@keyframes spectrum-spin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 480px) {
  .spectrum-row {
    grid-template-columns: minmax(44px, 62px) minmax(70px, 1fr) 42px;
    gap: 6px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .spectrum-spinner {
    animation-duration: 1.8s;
  }
}
</style>
