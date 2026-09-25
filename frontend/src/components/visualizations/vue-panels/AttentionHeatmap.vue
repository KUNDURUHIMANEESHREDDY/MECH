<template>
  <div class="attn-panel">
    <!-- Loading / empty states -->
    <div v-if="loading" class="attn-state" role="status" aria-live="polite">
      <span class="attn-spinner" aria-hidden="true"></span>
      <strong>Loading attention matrix…</strong>
    </div>

    <div v-else-if="!hasMatrix" class="attn-state" role="status">
      <strong>No attention matrix to display</strong>
      <span>Run a prompt to populate query/key attention weights.</span>
    </div>

    <template v-else>
      <!-- Toolbar -->
      <div class="attn-toolbar">
        <span class="attn-stat">{{ rows }} × {{ cols }} weights</span>
        <span class="attn-stat">peak {{ formatWeight(maxValue) }}</span>
        <div class="attn-scale" aria-hidden="true">
          <span class="attn-scale-edge">0.000</span>
          <span class="attn-swatches">
            <i v-for="(swatch, index) in LEGEND_SWATCHES" :key="index" :style="{ background: swatch }"></i>
          </span>
          <span class="attn-scale-edge">{{ formatWeight(maxValue) }}</span>
        </div>
      </div>

      <p :id="instructionsId" class="attn-instructions sr-only">
        Attention matrix. Rows are query tokens and columns are key tokens. Move the pointer over a cell to
        highlight its query token, click to select it, or focus the matrix and use the arrow keys to inspect
        cells and press Enter to select the query token.
      </p>

      <div class="attn-scroll">
        <canvas
          ref="canvasRef"
          class="attn-canvas"
          :width="pixelWidth"
          :height="pixelHeight"
          :style="{ width: `${width}px`, height: `${height}px` }"
          tabindex="0"
          role="img"
          :aria-label="canvasLabel"
          :aria-describedby="instructionsId"
          @mousemove="onMove"
          @mouseleave="onLeave"
          @click="onClick"
          @focus="onFocus"
          @blur="onBlur"
          @keydown="onKeydown"
        >{{ canvasLabel }}</canvas>

        <div
          v-if="tooltip"
          class="attn-tooltip"
          :style="tooltipStyle"
          aria-hidden="true"
        >
          <strong>{{ tooltip.from }}</strong>
          <span class="attn-tooltip-arrow">→</span>
          <span>{{ tooltip.to }}</span>
          <span class="attn-tooltip-value">{{ formatWeight(tooltip.value) }}</span>
        </div>
      </div>

      <p class="attn-readout" role="status" aria-live="polite">
        <template v-if="activeCell">{{ activeCell }}</template>
        <template v-else>Hover a cell, or use the arrow keys, to inspect attention weights.</template>
      </p>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, useId, watch } from 'vue';

/* ── geometry (ported from the previous panel) ──── */
const CELL_SIZE = 28;
const LABEL_WIDTH = 84;
const LABEL_HEIGHT = 20;
const CELL_INSET = 1;

const LABEL_MAX_CHARS = 8;

/**
 * Light -> deep blue ramp. Sized for a white surface, so the weakest
 * attention weights stay visible instead of disappearing into the page.
 */
const COLOR_STOPS: Array<[number, number, number]> = [
  [246, 248, 252],
  [219, 234, 254],
  [147, 197, 253],
  [37, 99, 235],
];
const LEGEND_SWATCHES = ['#F6F8FC', '#DBEAFE', '#93C5FD', '#2563EB'];

const INK = '#1f2937';
const PRIMARY = '#2563eb';
const LABEL_INK = '#667085';

interface Tooltip {
  from: string;
  to: string;
  value: number;
  x: number;
  y: number;
}

const props = withDefaults(defineProps<{
  matrix: number[][];
  tokens: string[];
  hoveredToken: number | null;
  loading?: boolean;
}>(), {
  matrix: () => [],
  tokens: () => [],
  hoveredToken: null,
  loading: false,
});

const emit = defineEmits<{
  (event: 'hoverToken', index: number | null): void;
  (event: 'selectToken', index: number): void;
}>();

const instructionsId = `attn-heatmap-${useId()}`;
const canvasRef = ref<HTMLCanvasElement>();
const pixelRatio = ref(1);
const tooltip = ref<Tooltip | null>(null);
const focusCell = ref<{ row: number; col: number } | null>(null);
const canvasFocused = ref(false);

let lastEmittedRow: number | null | undefined;
let drawFrame = 0;

/* ── derived shape ──────────────────────────────── */
const rows = computed(() => props.matrix.length);
const cols = computed(() => props.matrix[0]?.length ?? 0);
const hasMatrix = computed(() => rows.value > 0 && cols.value > 0);

const width = computed(() => LABEL_WIDTH + cols.value * CELL_SIZE);
const height = computed(() => LABEL_HEIGHT + rows.value * CELL_SIZE);
const pixelWidth = computed(() => Math.max(1, Math.round(width.value * pixelRatio.value)));
const pixelHeight = computed(() => Math.max(1, Math.round(height.value * pixelRatio.value)));

const maxValue = computed(() => {
  let peak = 0;
  for (const row of props.matrix) {
    if (!Array.isArray(row)) continue;
    for (const value of row) {
      if (typeof value === 'number' && Number.isFinite(value) && value > peak) peak = value;
    }
  }
  return peak;
});

const loading = computed(() => props.loading === true);

/* ── formatting helpers ─────────────────────────── */
function formatWeight(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : '—';
}

/** Mirrors the `Ġ`/`Ċ` convention used elsewhere in the explorer. */
function displayToken(token: string | undefined, maxChars = LABEL_MAX_CHARS): string {
  if (typeof token !== 'string' || !token) return '';
  const cleaned = token.replaceAll('\u0120', '\u2423').replaceAll('\u010a', '\u23ce').trim();
  return cleaned.length > maxChars ? cleaned.slice(0, maxChars) : cleaned;
}

/** Reads a matrix cell, tolerating ragged rows. Returns null when absent. */
function cellAt(row: number, col: number): number | null {
  const source = props.matrix[row];
  if (!Array.isArray(source) || col >= source.length) return null;
  const value = source[col];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function cellColor(value: number): string {
  // sqrt spreads low values so weak attention stays visible on a white surface
  const t = maxValue.value > 0 ? Math.sqrt(Math.max(0, value) / maxValue.value) : 0;
  const scaled = t * (COLOR_STOPS.length - 1);
  const index = Math.min(COLOR_STOPS.length - 2, Math.floor(scaled));
  const fraction = scaled - index;
  const from = COLOR_STOPS[index];
  const to = COLOR_STOPS[index + 1];
  const r = Math.round(from[0] + (to[0] - from[0]) * fraction);
  const g = Math.round(from[1] + (to[1] - from[1]) * fraction);
  const b = Math.round(from[2] + (to[2] - from[2]) * fraction);
  return `rgb(${r},${g},${b})`;
}

const canvasLabel = computed(() => {
  if (!hasMatrix.value) return 'Attention matrix: no data';
  const peak = formatWeight(maxValue.value);
  return `Attention matrix, ${rows.value} query tokens by ${cols.value} key tokens, peak weight ${peak}.`;
});

const tooltipStyle = computed(() => {
  const tip = tooltip.value;
  if (!tip) return {};
  const cardWidth = 210;
  const cardHeight = 46;
  return {
    left: `${Math.max(4, Math.min(tip.x, Math.max(4, width.value - cardWidth)))}px`,
    top: `${Math.max(4, Math.min(tip.y, Math.max(4, height.value - cardHeight)))}px`,
  };
});

const activeCell = computed(() => {
  const cell = focusCell.value;
  if (!cell) return '';
  const from = displayToken(props.tokens[cell.row], 24);
  const to = displayToken(props.tokens[cell.col], 24);
  const value = cellAt(cell.row, cell.col);
  if (value === null) return `Row ${cell.row}, column ${cell.col}: no weight reported.`;
  return `Query ${from || `(token ${cell.row})`} → key ${to || `(token ${cell.col})`}: ${formatWeight(value)} — row ${cell.row}, column ${cell.col}.`;
});

/* ── rendering ──────────────────────────────────── */
function draw(): void {
  const canvas = canvasRef.value;
  if (!canvas || !hasMatrix.value) return;
  const context = canvas.getContext('2d');
  if (!context) return;

  const ratio = pixelRatio.value;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width.value, height.value);
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, width.value, height.value);

  const highlight = props.hoveredToken;
  const focus = focusCell.value;

  // cells
  for (let row = 0; row < rows.value; row++) {
    const source = props.matrix[row];
    const span = Math.min(cols.value, Array.isArray(source) ? source.length : 0);
    for (let col = 0; col < span; col++) {
      const value = source[col];
      if (typeof value !== 'number' || !Number.isFinite(value)) continue;
      const x = LABEL_WIDTH + col * CELL_SIZE + CELL_INSET;
      const y = LABEL_HEIGHT + row * CELL_SIZE + CELL_INSET;
      const size = CELL_SIZE - CELL_INSET * 2;
      context.fillStyle = cellColor(value);
      context.fillRect(x, y, size, size);

      if (highlight === row || highlight === col) {
        context.strokeStyle = INK;
        context.lineWidth = 2;
        context.strokeRect(x, y, size, size);
      }
    }
  }

  // keyboard focus ring
  if (canvasFocused.value && focus) {
    const x = LABEL_WIDTH + focus.col * CELL_SIZE + CELL_INSET;
    const y = LABEL_HEIGHT + focus.row * CELL_SIZE + CELL_INSET;
    const size = CELL_SIZE - CELL_INSET * 2;
    context.strokeStyle = PRIMARY;
    context.lineWidth = 2;
    context.strokeRect(x - 1, y - 1, size + 2, size + 2);
  }

  // row labels
  context.fillStyle = LABEL_INK;
  context.font = '10px ui-sans-serif, system-ui, sans-serif';
  context.textAlign = 'right';
  context.textBaseline = 'middle';
  for (let row = 0; row < rows.value; row++) {
    const label = displayToken(props.tokens[row]);
    if (!label) continue;
    context.fillStyle = highlight === row ? INK : LABEL_INK;
    context.fillText(label, LABEL_WIDTH - 6, LABEL_HEIGHT + row * CELL_SIZE + CELL_SIZE / 2);
  }

  // column labels
  context.textAlign = 'center';
  context.textBaseline = 'top';
  for (let col = 0; col < cols.value; col++) {
    const label = displayToken(props.tokens[col]);
    if (!label) continue;
    context.fillStyle = highlight === col ? INK : LABEL_INK;
    context.fillText(label, LABEL_WIDTH + col * CELL_SIZE + CELL_SIZE / 2, 4);
  }
}

function scheduleDraw(): void {
  if (drawFrame) return;
  drawFrame = requestAnimationFrame(() => {
    drawFrame = 0;
    draw();
  });
}

/* ── pointer interaction ─────────────────────────── */
function localPoint(event: MouseEvent): { x: number; y: number } {
  const rect = canvasRef.value?.getBoundingClientRect();
  if (!rect) return { x: 0, y: 0 };
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function cellFromPoint(x: number, y: number): { row: number; col: number } | null {
  const col = Math.floor((x - LABEL_WIDTH) / CELL_SIZE);
  const row = Math.floor((y - LABEL_HEIGHT) / CELL_SIZE);
  if (row < 0 || row >= rows.value || col < 0 || col >= cols.value) return null;
  return { row, col };
}

/**
 * Rows are the unit of token highlighting, so only a row change is reported
 * upstream. This keeps the parent's `hoveredToken` accurate without emitting on
 * every pointer sample.
 */
function emitHoverRow(row: number | null): void {
  if (lastEmittedRow === row) return;
  lastEmittedRow = row;
  emit('hoverToken', row);
}

function onMove(event: MouseEvent): void {
  const { x, y } = localPoint(event);
  const cell = cellFromPoint(x, y);
  if (!cell || cellAt(cell.row, cell.col) === null) {
    emitHoverRow(null);
    tooltip.value = null;
    return;
  }
  emitHoverRow(cell.row);
  focusCell.value = cell;
  tooltip.value = {
    from: displayToken(props.tokens[cell.row], 24) || `token ${cell.row}`,
    to: displayToken(props.tokens[cell.col], 24) || `token ${cell.col}`,
    value: cellAt(cell.row, cell.col) as number,
    x: x + 12,
    y: y - 10,
  };
}

function onLeave(): void {
  emitHoverRow(null);
  tooltip.value = null;
}

function onClick(event: MouseEvent): void {
  const { x, y } = localPoint(event);
  const cell = cellFromPoint(x, y);
  if (!cell) return;
  focusCell.value = cell;
  emit('selectToken', cell.row);
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
function moveFocus(next: { row: number; col: number }): void {
  const row = Math.min(rows.value - 1, Math.max(0, next.row));
  const col = Math.min(cols.value - 1, Math.max(0, next.col));
  focusCell.value = { row, col };
  emitHoverRow(row);
  scheduleDraw();
}

function onKeydown(event: KeyboardEvent): void {
  if (!hasMatrix.value) return;
  const current = focusCell.value ?? { row: 0, col: 0 };
  const key = event.key;

  if (key === 'ArrowRight') moveFocus({ row: current.row, col: current.col + 1 });
  else if (key === 'ArrowLeft') moveFocus({ row: current.row, col: current.col - 1 });
  else if (key === 'ArrowDown') moveFocus({ row: current.row + 1, col: current.col });
  else if (key === 'ArrowUp') moveFocus({ row: current.row - 1, col: current.col });
  else if (key === 'Home') moveFocus({ row: current.row, col: 0 });
  else if (key === 'End') moveFocus({ row: current.row, col: cols.value - 1 });
  else if (key === 'Enter' || key === ' ') emit('selectToken', current.row);
  else if (key === 'Escape') {
    emitHoverRow(null);
    tooltip.value = null;
    return;
  } else {
    return;
  }

  event.preventDefault();
}

/* ── lifecycle ──────────────────────────────────── */
function syncPixelRatio(): void {
  const next = Math.min(2, window.devicePixelRatio || 1);
  if (pixelRatio.value !== next) pixelRatio.value = next;
}

watch(
  [() => props.matrix, () => props.tokens, () => props.hoveredToken, width, height, pixelRatio, canvasFocused],
  scheduleDraw,
  { flush: 'post' },
);

// A stale highlight from a previous token set must not linger in the readout.
watch(
  [() => props.matrix, () => props.tokens],
  () => {
    if (focusCell.value && (focusCell.value.row >= rows.value || focusCell.value.col >= cols.value)) {
      focusCell.value = null;
    }
    if (tooltip.value) tooltip.value = null;
  },
);

onMounted(() => {
  syncPixelRatio();
  window.addEventListener('resize', syncPixelRatio);
  scheduleDraw();
});

onBeforeUnmount(() => {
  window.removeEventListener('resize', syncPixelRatio);
  if (drawFrame) cancelAnimationFrame(drawFrame);
});
</script>

<style scoped>
.attn-panel {
  --border: #d9dee8;
  --border-strong: #c7cdd9;
  --text: #20283a;
  --muted: #667085;
  --accent: #2563eb;
  --accent-soft: #eaf1ff;
  display: grid;
  gap: 8px;
  min-width: 0;
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}

.attn-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 12px;
}

.attn-stat {
  color: var(--muted);
  font-size: 10px;
  font-variant-numeric: tabular-nums;
}

.attn-scale {
  display: flex;
  align-items: center;
  gap: 5px;
  margin-left: auto;
}

.attn-scale-edge {
  color: var(--muted);
  font-size: 9px;
  font-variant-numeric: tabular-nums;
}

.attn-swatches {
  display: inline-flex;
}

.attn-swatches i {
  display: block;
  width: 13px;
  height: 9px;
}

.attn-swatches i:first-child {
  border-radius: 2px 0 0 2px;
}

.attn-swatches i:last-child {
  border-radius: 0 2px 2px 0;
}

.attn-scroll {
  position: relative;
  max-width: 100%;
  max-height: 460px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: #ffffff;
  scrollbar-color: #c5cad5 transparent;
  scrollbar-width: thin;
}

.attn-canvas {
  display: block;
  cursor: crosshair;
}

.attn-canvas:focus-visible {
  outline: 2px solid #315bc4;
  outline-offset: -2px;
}

.attn-tooltip {
  position: absolute;
  z-index: 4;
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 4px;
  max-width: 210px;
  padding: 6px 8px;
  color: #30384a;
  background: #ffffff;
  border: 1px solid #b9c0cf;
  border-radius: 7px;
  box-shadow: 0 5px 16px rgb(16 24 40 / 13%);
  font-size: 10px;
  line-height: 1.3;
  pointer-events: none;
}

.attn-tooltip strong {
  color: #1f2937;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

.attn-tooltip-arrow {
  color: var(--muted);
}

.attn-tooltip-value {
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}

.attn-readout {
  margin: 0;
  color: var(--muted);
  font-size: 10px;
  line-height: 1.4;
  min-height: 14px;
}

.attn-state {
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

.attn-state strong {
  color: var(--text);
  font-size: 12px;
}

.attn-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid #d9d6ef;
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: attn-spin 0.8s linear infinite;
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

@keyframes attn-spin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 560px) {
  .attn-scale {
    margin-left: 0;
  }

  .attn-scroll {
    max-height: 320px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .attn-spinner {
    animation-duration: 1.8s;
  }
}
</style>
