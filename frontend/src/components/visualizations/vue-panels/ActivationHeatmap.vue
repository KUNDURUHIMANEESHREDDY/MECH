<template>
  <div class="act-panel">
    <!-- Loading / empty states -->
    <div v-if="loading" class="act-state" role="status" aria-live="polite">
      <span class="act-spinner" aria-hidden="true"></span>
      <strong>Loading neuron activations…</strong>
    </div>

    <div v-else-if="!values.length" class="act-state" role="status">
      <strong>No neuron activations to display</strong>
      <span>Run a prompt to populate activations for this layer.</span>
    </div>

    <template v-else>
      <!-- Zoom controls (semantic: real buttons + a live percentage) -->
      <div class="act-toolbar">
        <span id="zoomLabel" class="act-toolbar-label">Zoom</span>
        <div class="act-zoom" role="group" aria-labelledby="zoomLabel">
          <button
            type="button"
            :disabled="zoom <= ZOOM_MIN"
            aria-label="Zoom out"
            @click="nudgeZoom(-ZOOM_STEP)"
          >−</button>
          <output class="act-zoom-value" aria-live="off">{{ zoomPercent }}%</output>
          <button
            type="button"
            :disabled="zoom >= ZOOM_MAX"
            aria-label="Zoom in"
            @click="nudgeZoom(ZOOM_STEP)"
          >+</button>
        </div>
        <span class="act-hint">scroll to zoom · click a bar to select</span>
      </div>

      <p :id="instructionsId" class="sr-only">
        Neuron activation chart. Each bar is one neuron, scaled against the largest absolute activation in
        the set. Focus the chart and use the arrow keys to move between neurons, Enter to select the
        focused neuron, plus and minus to zoom, and 0 to reset the zoom.
      </p>

      <div class="act-scroll">
        <canvas
          ref="canvasRef"
          class="act-canvas"
          :width="pixelWidth"
          :height="pixelHeight"
          :style="{ width: `${width}px`, height: `${height}px` }"
          tabindex="0"
          role="img"
          :aria-label="canvasLabel"
          :aria-describedby="instructionsId"
          @click="onClick"
          @focus="onFocus"
          @blur="onBlur"
          @keydown="onKeydown"
        >{{ canvasLabel }}</canvas>
      </div>

      <p class="act-readout" role="status" aria-live="polite">
        <template v-if="activeIndex !== null">
          Neuron <strong>n{{ activeIndex }}</strong>, activation
          <span class="act-readout-value" :class="tone(values[activeIndex])">
            {{ formatValue(values[activeIndex]) }}
          </span><template v-if="activeIndex === neuronIndex">, <span class="act-readout-selected">selected</span></template>
        </template>
        <template v-else>
          {{ values.length }} neuron{{ values.length === 1 ? '' : 's' }} · peak {{ formatValue(maxAbs) }}
        </template>
      </p>

      <TokenActivationSpectrum
        v-if="selectedTokenActivations"
        :tokens="tokens"
        :activations="selectedTokenActivations"
      />
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, useId, watch } from 'vue';
import TokenActivationSpectrum from './TokenActivationSpectrum.vue';

/* ── geometry (ported from the previous panel) ──── */
const BASE_BAR_W = 16;
const BASE_BAR_H = 56;
const BAR_GAP = 4;
const LABEL_H = 14;
const PAD_X = 8;
const TOP_PAD = 18;
const ROW_BOTTOM_PAD = 8;
const BARS_AT_1X = 24;
const MIN_BARS_PER_ROW = 6;
const ZOOM_MIN = 0.3;
const ZOOM_MAX = 6;
const ZOOM_STEP = 0.1;

const POSITIVE = '#2563eb';
const POSITIVE_TRACK = '#bfdbfe';
const NEGATIVE = '#0e9384';
const NEGATIVE_TRACK = '#99ddd4';
const NEUTRAL = '#aab2c0';
const PRIMARY = '#2563eb';
const INK = '#1f2937';
const LABEL_INK = '#667085';

const props = withDefaults(defineProps<{
  activations: number[];
  neuronIndex: number | null;
  tokens?: string[];
  neuronTokenActivations?: Array<number[] | null | undefined>;
  loading?: boolean;
}>(), {
  activations: () => [],
  neuronIndex: null,
  tokens: () => [],
  neuronTokenActivations: () => [],
  loading: false,
});

const emit = defineEmits<{
  (event: 'selectNeuron', index: number): void;
}>();

const instructionsId = `act-heatmap-${useId()}`;
const canvasRef = ref<HTMLCanvasElement>();
const pixelRatio = ref(1);
const zoom = ref(1);
const focusIndex = ref<number | null>(null);
const canvasFocused = ref(false);

let drawFrame = 0;
let wheelTarget: HTMLElement | null = null;
let onWheel: ((event: WheelEvent) => void) | null = null;

/* ── derived data ────────────────────────────────── */
// Non-finite entries would poison the bar geometry, so they collapse to 0
// rather than producing NaN widths.
const values = computed(() => props.activations.map(value => (
  typeof value === 'number' && Number.isFinite(value) ? value : 0
)));

const loading = computed(() => props.loading === true);

const maxAbs = computed(() => values.value.reduce(
  (maximum, value) => Math.max(maximum, Math.abs(value)),
  0,
) || 0.01);

const barsPerRow = computed(() => Math.max(
  MIN_BARS_PER_ROW,
  Math.round(BARS_AT_1X / zoom.value),
));
const barW = computed(() => BASE_BAR_W * zoom.value);
const barH = computed(() => BASE_BAR_H * zoom.value);
const barPitch = computed(() => barW.value + BAR_GAP);
const rowH = computed(() => TOP_PAD + barH.value + LABEL_H + ROW_BOTTOM_PAD);
const rowCount = computed(() => Math.max(1, Math.ceil(values.value.length / barsPerRow.value)));
const width = computed(() => Math.round(PAD_X * 2 + barsPerRow.value * barPitch.value));
const height = computed(() => rowH.value * rowCount.value);
const pixelWidth = computed(() => Math.max(1, Math.round(width.value * pixelRatio.value)));
const pixelHeight = computed(() => Math.max(1, Math.round(height.value * pixelRatio.value)));

const zoomPercent = computed(() => Math.round(zoom.value * 100));

const selectedTokenActivations = computed<number[] | null>(() => {
  const index = props.neuronIndex;
  if (index === null || index === undefined) return null;
  const series = props.neuronTokenActivations?.[index];
  return Array.isArray(series) && series.length ? series : null;
});

const activeIndex = computed(() => {
  if (focusIndex.value !== null) return focusIndex.value;
  const index = props.neuronIndex;
  if (index === null || index === undefined) return null;
  return index >= 0 && index < values.value.length ? index : null;
});

const canvasLabel = computed(() => {
  if (!values.value.length) return 'Neuron activations: no data';
  const positive = values.value.filter(value => value > 0).length;
  return `Neuron activation chart with ${values.value.length} bars, ${positive} above zero, `
    + `peak absolute activation ${formatValue(maxAbs.value)}.`;
});

/* ── formatting helpers ─────────────────────────── */
function formatValue(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return '—';
  return value.toFixed(2);
}

function tone(value: number | undefined): 'positive' | 'negative' | 'neutral' {
  if (value === undefined || value === 0) return 'neutral';
  return value > 0 ? 'positive' : 'negative';
}

function barColor(value: number, isSelected: boolean): string {
  if (isSelected) return value >= 0 ? PRIMARY : '#0b7c70';
  if (value === 0) return NEUTRAL;
  return value > 0 ? POSITIVE : NEGATIVE;
}

function barTrackColor(value: number): string {
  if (value === 0) return NEUTRAL;
  return value > 0 ? POSITIVE_TRACK : NEGATIVE_TRACK;
}

/* ── rendering ──────────────────────────────────── */
function draw(): void {
  const canvas = canvasRef.value;
  if (!canvas || !values.value.length) return;
  const context = canvas.getContext('2d');
  if (!context) return;

  const ratio = pixelRatio.value;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width.value, height.value);
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, width.value, height.value);

  const scale = maxAbs.value;
  const half = barH.value / 2;
  const selected = props.neuronIndex;
  const focused = focusIndex.value;

  for (let index = 0; index < values.value.length; index++) {
    const row = Math.floor(index / barsPerRow.value);
    const col = index % barsPerRow.value;
    const x = PAD_X + col * barPitch.value;
    const rowTop = row * rowH.value;
    const value = values.value[index];

    // Zero baseline: positive bars grow up, negative bars grow down.
    const zeroY = rowTop + TOP_PAD + half;
    const magnitude = Math.abs(value) / scale;
    const barHeight = Math.max(1, magnitude * half);
    const barTop = value >= 0 ? zeroY - barHeight : zeroY;
    const isSelected = selected === index;

    // Full-height track keeps low-activation neurons locatable.
    context.fillStyle = barTrackColor(value);
    context.fillRect(x, rowTop + TOP_PAD, barW.value, barH.value);

    context.fillStyle = barColor(value, isSelected);
    context.fillRect(x, barTop, barW.value, barHeight);

    if (isSelected) {
      context.strokeStyle = '#b45309';
      context.lineWidth = 2;
      context.strokeRect(x - 1, rowTop + TOP_PAD - 1, barW.value + 2, barH.value + 2);
    }

    if (canvasFocused.value && focused === index) {
      context.strokeStyle = PRIMARY;
      context.lineWidth = 2;
      context.strokeRect(x - 1, rowTop + TOP_PAD - 1, barW.value + 2, barH.value + 2);
    }

    // value label above the plot area
    context.fillStyle = isSelected ? INK : LABEL_INK;
    context.font = '9px ui-sans-serif, system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'alphabetic';
    context.fillText(formatValue(value), x + barW.value / 2, rowTop + TOP_PAD - 5);

    // neuron label below the plot area
    context.fillStyle = isSelected ? INK : LABEL_INK;
    context.fillText(`n${index}`, x + barW.value / 2, rowTop + TOP_PAD + barH.value + 10);
  }

  // zero line
  context.strokeStyle = '#b9c0cf';
  context.lineWidth = 1;
  context.beginPath();
  for (let row = 0; row < rowCount.value; row++) {
    const y = Math.round(row * rowH.value + TOP_PAD + half) + 0.5;
    context.moveTo(PAD_X, y);
    context.lineTo(width.value - PAD_X, y);
  }
  context.stroke();
}

function scheduleDraw(): void {
  if (drawFrame) return;
  drawFrame = requestAnimationFrame(() => {
    drawFrame = 0;
    draw();
  });
}

/* ── zoom ────────────────────────────────────────── */
function setZoom(next: number): void {
  const clamped = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, next));
  if (clamped === zoom.value) return;
  zoom.value = clamped;
}

function nudgeZoom(delta: number): void {
  setZoom(Math.round((zoom.value + delta) * 100) / 100);
}

function handleWheel(event: WheelEvent): void {
  if (!event.deltaY) return;
  event.preventDefault();
  setZoom(zoom.value * (event.deltaY < 0 ? 1.2 : 1 / 1.2));
}

/* ── pointer interaction ─────────────────────────── */
function localPoint(event: MouseEvent): { x: number; y: number } {
  const rect = canvasRef.value?.getBoundingClientRect();
  if (!rect) return { x: 0, y: 0 };
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function indexFromPoint(x: number, y: number): number | null {
  if (!values.value.length) return null;
  const row = Math.floor(y / rowH.value);
  if (row < 0 || row >= rowCount.value) return null;
  // Ignore clicks in the padding above/below the plot area of that row.
  if (y < row * rowH.value + TOP_PAD - 2 || y > row * rowH.value + rowH.value - ROW_BOTTOM_PAD) return null;

  const col = Math.floor((x - PAD_X) / barPitch.value);
  if (col < 0 || col >= barsPerRow.value) return null;

  const index = row * barsPerRow.value + col;
  return index >= 0 && index < values.value.length ? index : null;
}

function onClick(event: MouseEvent): void {
  const { x, y } = localPoint(event);
  const index = indexFromPoint(x, y);
  if (index === null) return;
  focusIndex.value = index;
  emit('selectNeuron', index);
}

function onFocus(): void {
  canvasFocused.value = true;
  scheduleDraw();
}

function onBlur(): void {
  canvasFocused.value = false;
  scheduleDraw();
}

/* ── keyboard interaction ────────────────────────── */
function moveFocus(next: number): void {
  if (!values.value.length) return;
  const clamped = Math.min(values.value.length - 1, Math.max(0, next));
  focusIndex.value = clamped;
  scheduleDraw();
}

function onKeydown(event: KeyboardEvent): void {
  if (!values.value.length) return;
  const perRow = barsPerRow.value;
  const current = focusIndex.value ?? props.neuronIndex ?? 0;
  const key = event.key;

  if (key === 'ArrowRight') moveFocus(current + 1);
  else if (key === 'ArrowLeft') moveFocus(current - 1);
  else if (key === 'ArrowDown') moveFocus(current + perRow);
  else if (key === 'ArrowUp') moveFocus(current - perRow);
  else if (key === 'Home') moveFocus(Math.floor(current / perRow) * perRow);
  else if (key === 'End') moveFocus(Math.floor(current / perRow) * perRow + perRow - 1);
  else if (key === 'Enter' || key === ' ') {
    moveFocus(current);
    emit('selectNeuron', focusIndex.value ?? current);
  } else if (key === '+' || key === '=') setZoom(zoom.value + 0.1);
  else if (key === '-') setZoom(zoom.value - 0.1);
  else if (key === '0') setZoom(1);
  else return;

  event.preventDefault();
}

/* ── lifecycle ──────────────────────────────────── */
function syncPixelRatio(): void {
  const next = Math.min(2, window.devicePixelRatio || 1);
  if (pixelRatio.value !== next) pixelRatio.value = next;
}

watch(
  [values, () => props.neuronIndex, zoom, width, height, pixelRatio, canvasFocused, focusIndex],
  scheduleDraw,
  { flush: 'post' },
);

// Reset the keyboard cursor when a different layer or head supplies new data.
watch(values, () => {
  focusIndex.value = null;
});

onMounted(() => {
  syncPixelRatio();
  window.addEventListener('resize', syncPixelRatio);

  // Registered manually so the listener is non-passive and can cancel scroll.
  const canvas = canvasRef.value;
  if (canvas) {
    wheelTarget = canvas;
    onWheel = handleWheel;
    canvas.addEventListener('wheel', onWheel, { passive: false });
  }

  scheduleDraw();
});

onBeforeUnmount(() => {
  window.removeEventListener('resize', syncPixelRatio);
  if (wheelTarget && onWheel) wheelTarget.removeEventListener('wheel', onWheel);
  if (drawFrame) cancelAnimationFrame(drawFrame);
});
</script>

<style scoped>
.act-panel {
  --border: #d9dee8;
  --border-strong: #c7cdd9;
  --text: #20283a;
  --muted: #667085;
  --accent: #2563eb;
  display: grid;
  gap: 8px;
  min-width: 0;
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}

.act-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 9px;
}

.act-toolbar-label {
  color: var(--muted);
  font-size: 10px;
  font-weight: 600;
}

.act-zoom {
  display: flex;
  align-items: center;
  gap: 4px;
}

.act-zoom button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  padding: 0;
  color: var(--text);
  background: #ffffff;
  border: 1px solid var(--border-strong);
  border-radius: 5px;
  font: inherit;
  font-size: 13px;
  line-height: 1;
  cursor: pointer;
}

.act-zoom button:hover:not(:disabled) {
  color: #1d4ed8;
  background: #f8f7fd;
  border-color: #a9a1dc;
}

.act-zoom button:disabled {
  color: #98a2b3;
  background: #f8f9fb;
  cursor: not-allowed;
}

.act-zoom button:focus-visible,
.act-canvas:focus-visible {
  outline: 2px solid #315bc4;
  outline-offset: 1px;
}

.act-zoom-value {
  min-width: 38px;
  color: var(--text);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10px;
  text-align: center;
}

.act-hint {
  color: var(--muted);
  font-size: 9px;
}

.act-scroll {
  max-width: 100%;
  max-height: 420px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: #ffffff;
  scrollbar-color: #c5cad5 transparent;
  scrollbar-width: thin;
}

.act-canvas {
  display: block;
  cursor: pointer;
}

.act-readout {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px;
  margin: 0;
  color: var(--muted);
  font-size: 10px;
  line-height: 1.4;
  min-height: 14px;
}

.act-readout strong {
  color: #1f2937;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

.act-readout-value {
  font-variant-numeric: tabular-nums;
}

.act-readout-value.positive {
  color: #2563eb;
}

.act-readout-value.negative {
  color: #0e9384;
}

.act-readout-value.neutral {
  color: var(--muted);
}

.act-readout-selected {
  color: #92400e;
  font-weight: 600;
}

.act-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 26px 18px;
  color: var(--muted);
  background: #ffffff;
  border: 1px dashed var(--border-strong);
  border-radius: 8px;
  text-align: center;
  font-size: 11px;
}

.act-state strong {
  color: var(--text);
  font-size: 12px;
}

.act-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid #d9d6ef;
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: act-spin 0.8s linear infinite;
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

@keyframes act-spin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 560px) {
  .act-scroll {
    max-height: 300px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .act-spinner {
    animation-duration: 1.8s;
  }
}
</style>
