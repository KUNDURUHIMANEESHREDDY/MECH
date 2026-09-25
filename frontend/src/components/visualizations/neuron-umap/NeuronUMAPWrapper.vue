<template>
  <section
    class="neuron-map-shell"
    :style="{ height: `${componentHeight}px` }"
    :aria-labelledby="titleId"
  >
    <header class="map-toolbar">
      <div class="heading">
        <h2 :id="titleId">{{ title }}</h2>
        <p>{{ mapStatus }}</p>
      </div>

      <div class="controls" aria-label="Neuron map controls">
        <form class="search-control" role="search" @submit.prevent="selectFirstSearchResult">
          <label :for="`${titleId}-search`">Find neuron</label>
          <div class="search-row">
            <input
              :id="`${titleId}-search`"
              ref="searchInput"
              v-model="searchQuery"
              type="search"
              autocomplete="off"
              placeholder="ID, layer, head, or token"
              :aria-describedby="`${titleId}-search-status`"
              @keydown.down.prevent="focusFirstListEntry"
              @keydown.esc="searchQuery = ''"
            />
            <button type="submit" :disabled="!searchResults.length">Go</button>
          </div>
          <span :id="`${titleId}-search-status`" class="sr-only" aria-live="polite">
            {{ searchResults.length }} matching neurons
          </span>
        </form>

        <label class="select-control">
          <span>Layer</span>
          <select :value="layerFilter ?? ''" @change="onLayerFilterChange">
            <option value="">All layers</option>
            <option v-for="layer in layers" :key="layer" :value="layer">Layer {{ layer }}</option>
          </select>
        </label>

        <label v-if="heads.length" class="select-control">
          <span>Head</span>
          <select :value="headFilter ?? ''" @change="onHeadFilterChange">
            <option value="">All heads</option>
            <option v-for="head in heads" :key="head" :value="head">Head {{ head }}</option>
          </select>
        </label>

        <label class="threshold-control" :for="`${titleId}-threshold`">
          <span>Threshold</span>
          <input
            :id="`${titleId}-threshold`"
            v-model.number="threshold"
            type="range"
            min="0"
            max="0.5"
            step="0.005"
            aria-describedby="`${titleId}-threshold-help`"
          />
          <output :for="`${titleId}-threshold`">{{ threshold.toFixed(2) }}</output>
        </label>
        <span :id="`${titleId}-threshold-help`" class="sr-only">
          Hide points whose absolute activation is below this fraction of the maximum absolute activation.
        </span>

        <div class="button-row" role="group" aria-label="Map view controls">
          <button type="button" :disabled="loading || !visiblePoints.length" @click="zoomAt(1.3)">Zoom +</button>
          <button type="button" :disabled="loading || !visiblePoints.length" @click="zoomAt(1 / 1.3)">Zoom −</button>
          <button type="button" :disabled="loading || !visiblePoints.length" @click="fitView">Fit</button>
          <button type="button" :disabled="loading || !visiblePoints.length" @click="exportPNG">PNG</button>
          <button
            type="button"
            :disabled="layerFilter === null && headFilter === null && threshold === 0"
            @click="resetFilters"
          >
            Reset
          </button>
        </div>
      </div>
    </header>

    <p :id="instructionsId" class="sr-only">
      Neuron map. Drag to pan, scroll or use the view buttons to zoom, and press F to fit. Use the point list to select a neuron with the keyboard.
    </p>

    <div class="map-workspace">
      <div ref="mapArea" class="map-area">
        <canvas
          ref="mapCanvas"
          class="map-canvas"
          tabindex="0"
          role="img"
          :aria-label="canvasLabel"
          :aria-describedby="instructionsId"
          @pointerdown="onPointerDown"
          @pointermove="onPointerMove"
          @pointerup="onPointerUp"
          @pointercancel="onPointerCancel"
          @pointerleave="onPointerLeave"
          @wheel="onWheel"
          @keydown="onMapKeydown"
        >
          {{ canvasLabel }}
        </canvas>

        <div class="legend" aria-label="Point color legend">
          <span><i class="positive-dot" aria-hidden="true"></i>Positive activation</span>
          <span><i class="negative-dot" aria-hidden="true"></i>Negative activation</span>
          <span><i class="neutral-dot" aria-hidden="true"></i>Neutral</span>
        </div>

        <div
          v-if="hoveredEntry"
          class="point-tooltip"
          :style="tooltipStyle"
          aria-hidden="true"
        >
          <strong>{{ hoveredEntry.point.id }}</strong>
          <span>{{ pointLabel(hoveredEntry.point) }}</span>
          <span v-if="hoveredEntry.point.topToken">Top token: {{ hoveredEntry.point.topToken }}</span>
        </div>

        <div v-if="loading" class="state-overlay" role="status" aria-live="polite">
          <span class="spinner" aria-hidden="true"></span>
          <strong>Loading neurons…</strong>
        </div>
        <div v-else-if="points.length === 0" class="state-overlay" role="status">
          <strong>No neuron data available</strong>
          <span>Run a prompt or load neuron data to populate this map.</span>
        </div>
        <div v-else-if="positionedPoints.length === 0" class="state-overlay" role="status">
          <strong>No usable embeddings available</strong>
          <span>The map only projects supplied embeddings with at least two finite dimensions.</span>
        </div>
        <div v-else-if="visiblePoints.length === 0" class="state-overlay" role="status">
          <strong>No neurons match the current filters</strong>
          <button type="button" @click="resetFilters">Reset filters</button>
        </div>
      </div>

      <aside class="point-panel" aria-label="Accessible neuron point list">
        <div v-if="selectedEntry" class="selected-summary">
          <span>Selected</span>
          <strong>{{ selectedEntry.point.id }}</strong>
          <small>{{ pointLabel(selectedEntry.point) }}</small>
        </div>

        <div class="list-heading">
          <strong>{{ searchQuery.trim() ? 'Search results' : 'Strongest visible points' }}</strong>
          <span v-if="!searchQuery.trim()">Top 50 by absolute activation</span>
          <span v-else>{{ searchResults.length }} matches</span>
        </div>

        <ul v-if="listEntries.length" ref="listElement" class="point-list" :aria-label="searchQuery.trim() ? 'Neuron search results' : 'Strongest visible neurons'">
          <li v-for="entry in listEntries" :key="entry.point.id">
            <button
              type="button"
              :aria-current="selectedId === entry.point.id ? 'true' : undefined"
              :title="pointDescription(entry.point)"
              @click="selectPositionedPoint(entry, true)"
            >
              <span class="point-row">
                <strong>{{ entry.point.id }}</strong>
                <span :class="['activation', activationTone(entry.point)]">
                  {{ formatActivation(entry.point.activation) }}
                </span>
              </span>
              <span class="point-meta">
                Layer {{ entry.point.layer }}<template v-if="entry.point.head !== undefined"> · Head {{ entry.point.head }}</template>
                · Neuron {{ entry.point.neuronIndex }}
              </span>
            </button>
          </li>
        </ul>
        <p v-else class="empty-list">
          {{ searchQuery.trim() ? 'No matching neurons.' : 'No points pass the current filters.' }}
        </p>
      </aside>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, useId, watch } from 'vue';
import { clearProjectionCache, projectNeurons } from './projection';
import type { NeuronPoint, NeuronUMapProps, Viewport } from './types';

interface PositionedPoint {
  point: NeuronPoint;
  x: number;
  y: number;
  sourceIndex: number;
}

interface DragState {
  pointerId: number;
  startX: number;
  startY: number;
  viewport: Viewport;
  moved: boolean;
}

const props = defineProps<{
  points: NeuronUMapProps['points'];
  selectedId: NeuronUMapProps['selectedId'];
  onSelectNeuron: NeuronUMapProps['onSelectNeuron'];
  height?: number;
  loading?: boolean;
  title?: string;
}>();

const MIN_ZOOM = 1;
const MAX_ZOOM = 40;
const GRID_CELLS = 96;
const HIT_RADIUS = 11;

const titleId = `neuron-map-${useId()}`;
const instructionsId = `${titleId}-instructions`;
const componentHeight = computed(() => Math.max(480, props.height ?? 520));
const title = computed(() => props.title ?? 'Neuron Map');
const loading = computed(() => props.loading ?? false);
const selectedId = computed(() => props.selectedId ?? null);

const mapArea = ref<HTMLDivElement>();
const mapCanvas = ref<HTMLCanvasElement>();
const searchInput = ref<HTMLInputElement>();
const listElement = ref<HTMLUListElement>();
const mapWidth = ref(0);
const mapHeight = ref(0);
const hoveredId = ref<string | null>(null);
const pointerPosition = ref({ x: 0, y: 0 });
const viewport = ref<Viewport>({ x: 0, y: 0, k: 1 });
const searchQuery = ref('');
const layerFilter = ref<number | null>(null);
const headFilter = ref<number | null>(null);
const threshold = ref(0.02);

let resizeObserver: ResizeObserver | undefined;
let drawFrame = 0;
let dragState: DragState | null = null;

const positionedPoints = computed<PositionedPoint[]>(() => {
  const usable: Array<{ point: NeuronPoint; sourceIndex: number }> = [];

  props.points.forEach((point, sourceIndex) => {
    if (!Array.isArray(point.embedding) || point.embedding.length < 2) return;
    if (!point.embedding.every(value => Number.isFinite(value))) return;
    usable.push({ point, sourceIndex });
  });

  if (!usable.length) return [];

  const dimension = usable[0].point.embedding.length;
  const compatible = usable.filter(item => item.point.embedding.length === dimension);
  if (!compatible.length) return [];

  // The shared projection cache is keyed by shape rather than values. Clear it
  // so replacement data with the same shape can never reuse stale positions.
  clearProjectionCache();
  const coordinates = projectNeurons(compatible.map(item => item.point));
  if (coordinates.length !== compatible.length) return [];

  return compatible.map((item, index) => ({
    ...item,
    x: coordinates[index]?.x ?? 0,
    y: coordinates[index]?.y ?? 0,
  }));
});

const invalidPointCount = computed(() => props.points.length - positionedPoints.value.length);
const maxAbsActivation = computed(() => positionedPoints.value.reduce(
  (maximum, entry) => Math.max(maximum, Math.abs(entry.point.activation)),
  0,
));
const thresholdCutoff = computed(() => threshold.value * maxAbsActivation.value);

const layers = computed(() => [...new Set(positionedPoints.value.map(entry => entry.point.layer))].sort((a, b) => a - b));
const heads = computed(() => [...new Set(
  positionedPoints.value
    .map(entry => entry.point.head)
    .filter((head): head is number => head !== undefined),
)].sort((a, b) => a - b));

const visiblePoints = computed(() => positionedPoints.value.filter(({ point }) => {
  if (layerFilter.value !== null && point.layer !== layerFilter.value) return false;
  if (headFilter.value !== null && point.head !== headFilter.value) return false;
  if (maxAbsActivation.value > 0 && Math.abs(point.activation) < thresholdCutoff.value) return false;
  return true;
}));

const activeCount = computed(() => visiblePoints.value.filter(({ point }) => {
  if (maxAbsActivation.value === 0) return false;
  return point.activation > thresholdCutoff.value || point.activation < -thresholdCutoff.value;
}).length);

const selectedEntry = computed(() => positionedPoints.value.find(entry => entry.point.id === selectedId.value) ?? null);
const hoveredEntry = computed(() => visiblePoints.value.find(entry => entry.point.id === hoveredId.value) ?? null);
const hiddenCount = computed(() => positionedPoints.value.length - visiblePoints.value.length);

const searchResults = computed(() => {
  const query = searchQuery.value.trim().toLowerCase();
  if (!query) return [];

  return positionedPoints.value.filter(({ point }) => {
    const searchable = [
      point.id,
      `layer ${point.layer}`,
      point.head === undefined ? '' : `head ${point.head}`,
      `neuron ${point.neuronIndex}`,
      point.topToken ?? '',
    ].join(' ').toLowerCase();
    return searchable.includes(query);
  }).slice(0, 12);
});

const listEntries = computed(() => {
  if (searchQuery.value.trim()) return searchResults.value;
  return [...visiblePoints.value]
    .sort((a, b) => Math.abs(b.point.activation) - Math.abs(a.point.activation) || a.point.id.localeCompare(b.point.id))
    .slice(0, 50);
});

const mapStatus = computed(() => {
  if (loading.value) return 'Loading supplied neuron embeddings…';
  if (!props.points.length) return 'Waiting for neuron data';
  if (!positionedPoints.value.length) return 'No projectable embeddings';
  return `${visiblePoints.value.length} of ${positionedPoints.value.length} points · ${activeCount.value} beyond threshold · ${Math.round(viewport.value.k * 100)}% zoom`;
});

const canvasLabel = computed(() => `${title.value}: ${visiblePoints.value.length} visible neurons, ${activeCount.value} beyond the activation threshold`);
const tooltipStyle = computed(() => ({
  left: `${Math.max(8, Math.min(pointerPosition.value.x + 14, Math.max(8, mapWidth.value - 224)))}px`,
  top: `${Math.max(8, Math.min(pointerPosition.value.y + 14, Math.max(8, mapHeight.value - 88)))}px`,
}));

const pointGrid = computed(() => {
  const grid = new Map<number, PositionedPoint[]>();
  for (const entry of visiblePoints.value) {
    const column = Math.min(GRID_CELLS - 1, Math.max(0, Math.floor(entry.x * GRID_CELLS)));
    const row = Math.min(GRID_CELLS - 1, Math.max(0, Math.floor(entry.y * GRID_CELLS)));
    const key = row * GRID_CELLS + column;
    const bucket = grid.get(key);
    if (bucket) bucket.push(entry);
    else grid.set(key, [entry]);
  }
  return grid;
});

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function normalizedViewport(next: Viewport): Viewport {
  const k = clamp(next.k, MIN_ZOOM, MAX_ZOOM);
  return {
    k,
    x: clamp(next.x, 0, Math.max(0, 1 - 1 / k)),
    y: clamp(next.y, 0, Math.max(0, 1 - 1 / k)),
  };
}

function fitView(): void {
  viewport.value = { x: 0, y: 0, k: 1 };
}

function zoomAt(factor: number, screenX = mapWidth.value / 2, screenY = mapHeight.value / 2): void {
  if (!mapWidth.value || !mapHeight.value) return;
  const previous = viewport.value;
  const k = clamp(previous.k * factor, MIN_ZOOM, MAX_ZOOM);
  const worldX = previous.x + screenX / (mapWidth.value * previous.k);
  const worldY = previous.y + screenY / (mapHeight.value * previous.k);
  viewport.value = normalizedViewport({
    k,
    x: worldX - screenX / (mapWidth.value * k),
    y: worldY - screenY / (mapHeight.value * k),
  });
}

function focusPoint(entry: PositionedPoint): void {
  if (!mapWidth.value || !mapHeight.value) return;
  const k = Math.max(viewport.value.k, 3);
  viewport.value = normalizedViewport({
    k,
    x: entry.x - 0.5 / k,
    y: entry.y - 0.5 / k,
  });
}

function formatActivation(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(4)}`;
}

function pointLabel(point: NeuronPoint): string {
  const head = point.head === undefined ? '' : ` · Head ${point.head}`;
  return `Layer ${point.layer}${head} · Activation ${formatActivation(point.activation)}`;
}

function pointDescription(point: NeuronPoint): string {
  const token = point.topToken ? ` · Top token ${point.topToken}` : '';
  return `${point.id} · ${pointLabel(point)} · Neuron ${point.neuronIndex}${token}`;
}

function activationTone(point: NeuronPoint): 'positive' | 'negative' | 'neutral' {
  if (maxAbsActivation.value === 0) return 'neutral';
  if (point.activation > thresholdCutoff.value) return 'positive';
  if (point.activation < -thresholdCutoff.value) return 'negative';
  return 'neutral';
}

function isVisible(entry: PositionedPoint): boolean {
  return visiblePoints.value.some(candidate => candidate.sourceIndex === entry.sourceIndex);
}

function selectPositionedPoint(entry: PositionedPoint, focus = false): void {
  if (!isVisible(entry)) resetFilters();
  props.onSelectNeuron?.(entry.point.id);
  if (focus) focusPoint(entry);
}

function clearSelection(): void {
  props.onSelectNeuron?.(null);
}

function resetFilters(): void {
  layerFilter.value = null;
  headFilter.value = null;
  threshold.value = 0;
}

function onLayerFilterChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value;
  layerFilter.value = value === '' ? null : Number(value);
}

function onHeadFilterChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value;
  headFilter.value = value === '' ? null : Number(value);
}

function selectFirstSearchResult(): void {
  const first = searchResults.value[0];
  if (!first) return;
  searchQuery.value = '';
  selectPositionedPoint(first, true);
}

function focusFirstListEntry(): void {
  listElement.value?.querySelector<HTMLButtonElement>('button')?.focus();
}

function localPointer(event: PointerEvent | WheelEvent): { x: number; y: number } {
  const rect = (event.currentTarget as HTMLCanvasElement).getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function hitTest(screenX: number, screenY: number): PositionedPoint | null {
  const currentViewport = viewport.value;
  const worldX = currentViewport.x + screenX / (mapWidth.value * currentViewport.k);
  const worldY = currentViewport.y + screenY / (mapHeight.value * currentViewport.k);
  const radiusX = HIT_RADIUS / (mapWidth.value * currentViewport.k);
  const radiusY = HIT_RADIUS / (mapHeight.value * currentViewport.k);
  const minimumColumn = Math.max(0, Math.floor((worldX - radiusX) * GRID_CELLS));
  const maximumColumn = Math.min(GRID_CELLS - 1, Math.floor((worldX + radiusX) * GRID_CELLS));
  const minimumRow = Math.max(0, Math.floor((worldY - radiusY) * GRID_CELLS));
  const maximumRow = Math.min(GRID_CELLS - 1, Math.floor((worldY + radiusY) * GRID_CELLS));
  const hitRadiusSquared = HIT_RADIUS * HIT_RADIUS;
  let closest: PositionedPoint | null = null;
  let closestDistance = hitRadiusSquared;

  for (let row = minimumRow; row <= maximumRow; row++) {
    for (let column = minimumColumn; column <= maximumColumn; column++) {
      for (const entry of pointGrid.value.get(row * GRID_CELLS + column) ?? []) {
        const deltaX = (entry.x - worldX) * mapWidth.value * currentViewport.k;
        const deltaY = (entry.y - worldY) * mapHeight.value * currentViewport.k;
        const distance = deltaX * deltaX + deltaY * deltaY;
        if (distance <= closestDistance) {
          closest = entry;
          closestDistance = distance;
        }
      }
    }
  }

  return closest;
}

function onPointerDown(event: PointerEvent): void {
  if (event.button !== 0) return;
  const position = localPointer(event);
  dragState = {
    pointerId: event.pointerId,
    startX: position.x,
    startY: position.y,
    viewport: { ...viewport.value },
    moved: false,
  };
  (event.currentTarget as HTMLCanvasElement).setPointerCapture?.(event.pointerId);
}

function onPointerMove(event: PointerEvent): void {
  const position = localPointer(event);
  pointerPosition.value = position;

  if (dragState?.pointerId === event.pointerId) {
    const deltaX = position.x - dragState.startX;
    const deltaY = position.y - dragState.startY;
    if (Math.hypot(deltaX, deltaY) > 3) dragState.moved = true;
    if (dragState.moved) {
      const k = dragState.viewport.k;
      viewport.value = normalizedViewport({
        k,
        x: dragState.viewport.x - deltaX / (mapWidth.value * k),
        y: dragState.viewport.y - deltaY / (mapHeight.value * k),
      });
    }
    return;
  }

  const hit = hitTest(position.x, position.y);
  hoveredId.value = hit?.point.id ?? null;
}

function onPointerUp(event: PointerEvent): void {
  if (dragState?.pointerId !== event.pointerId) return;
  const position = localPointer(event);
  const wasMoved = dragState.moved;
  dragState = null;
  (event.currentTarget as HTMLCanvasElement).releasePointerCapture?.(event.pointerId);

  if (!wasMoved) {
    const hit = hitTest(position.x, position.y);
    if (hit) selectPositionedPoint(hit);
    else clearSelection();
  }
}

function onPointerCancel(event: PointerEvent): void {
  if (dragState?.pointerId === event.pointerId) dragState = null;
  hoveredId.value = null;
}

function onPointerLeave(): void {
  if (!dragState) hoveredId.value = null;
}

function onWheel(event: WheelEvent): void {
  event.preventDefault();
  if (!event.deltaY) return;
  const position = localPointer(event);
  zoomAt(event.deltaY < 0 ? 1.2 : 1 / 1.2, position.x, position.y);
}

function onMapKeydown(event: KeyboardEvent): void {
  const key = event.key.toLowerCase();
  if (key === '+' || key === '=') {
    event.preventDefault();
    zoomAt(1.3);
  } else if (key === '-') {
    event.preventDefault();
    zoomAt(1 / 1.3);
  } else if (key === 'f' || key === 'home') {
    event.preventDefault();
    fitView();
  } else if (event.key === 'Enter' && hoveredEntry.value) {
    event.preventDefault();
    selectPositionedPoint(hoveredEntry.value);
  } else if (['arrowleft', 'arrowright', 'arrowup', 'arrowdown'].includes(key)) {
    event.preventDefault();
    const step = event.shiftKey ? 80 : 32;
    const current = viewport.value;
    viewport.value = normalizedViewport({
      k: current.k,
      x: current.x + (key === 'arrowright' ? step : key === 'arrowleft' ? -step : 0) / (mapWidth.value * current.k),
      y: current.y + (key === 'arrowdown' ? step : key === 'arrowup' ? -step : 0) / (mapHeight.value * current.k),
    });
  }
}

function onGlobalShortcut(event: KeyboardEvent): void {
  if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== 'k') return;
  const target = event.target as HTMLElement | null;
  const tag = target?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target?.isContentEditable) return;
  event.preventDefault();
  searchInput.value?.focus();
}

function drawMap(): void {
  const canvas = mapCanvas.value;
  if (!canvas || mapWidth.value <= 0 || mapHeight.value <= 0) return;
  const context = canvas.getContext('2d');
  if (!context) return;

  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const pixelWidth = Math.max(1, Math.round(mapWidth.value * dpr));
  const pixelHeight = Math.max(1, Math.round(mapHeight.value * dpr));
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;

  const currentViewport = viewport.value;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, mapWidth.value, mapHeight.value);
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, mapWidth.value, mapHeight.value);

  context.beginPath();
  context.strokeStyle = '#e8ebf1';
  context.lineWidth = 1;
  if (currentViewport.k <= 8) {
    const gridStep = 0.1;
    const startX = Math.floor(currentViewport.x / gridStep) * gridStep;
    const startY = Math.floor(currentViewport.y / gridStep) * gridStep;
    const endX = currentViewport.x + 1 / currentViewport.k;
    const endY = currentViewport.y + 1 / currentViewport.k;
    for (let x = startX; x <= endX + gridStep; x += gridStep) {
      const screenX = (x - currentViewport.x) * mapWidth.value * currentViewport.k;
      context.moveTo(screenX, 0);
      context.lineTo(screenX, mapHeight.value);
    }
    for (let y = startY; y <= endY + gridStep; y += gridStep) {
      const screenY = (y - currentViewport.y) * mapHeight.value * currentViewport.k;
      context.moveTo(0, screenY);
      context.lineTo(mapWidth.value, screenY);
    }
  }
  context.stroke();

  const radius = Math.min(4.4, 2.2 + Math.sqrt(currentViewport.k) * 0.48);
  const labelEveryPoint = currentViewport.k >= 3.5 && visiblePoints.value.length <= 400;
  const labelEntries: Array<{ entry: PositionedPoint; x: number; y: number }> = [];

  for (const entry of visiblePoints.value) {
    const screenX = (entry.x - currentViewport.x) * mapWidth.value * currentViewport.k;
    const screenY = (entry.y - currentViewport.y) * mapHeight.value * currentViewport.k;
    if (screenX < -8 || screenX > mapWidth.value + 8 || screenY < -8 || screenY > mapHeight.value + 8) continue;

    const isSelected = entry.point.id === selectedId.value;
    const isHovered = entry.point.id === hoveredId.value;
    const tone = activationTone(entry.point);

    context.beginPath();
    context.fillStyle = isSelected ? '#2563eb' : tone === 'positive' ? '#19764a' : tone === 'negative' ? '#b42318' : '#7b8494';
    context.arc(screenX, screenY, isSelected || isHovered ? radius + 1 : radius, 0, Math.PI * 2);
    context.fill();

    if (isSelected) {
      context.beginPath();
      context.strokeStyle = '#1d4ed8';
      context.lineWidth = 2;
      context.arc(screenX, screenY, radius + 4, 0, Math.PI * 2);
      context.stroke();
    }

    if (labelEveryPoint || isSelected || isHovered) {
      labelEntries.push({ entry, x: screenX, y: screenY });
    }
  }

  context.font = '11px Inter, ui-sans-serif, system-ui, sans-serif';
  context.textBaseline = 'middle';
  context.textAlign = 'left';
  for (const { entry, x, y } of labelEntries) {
    const label = `L${entry.point.layer}${entry.point.head === undefined ? '' : ` H${entry.point.head}`} · ${formatActivation(entry.point.activation)}`;
    const labelWidth = context.measureText(label).width + 10;
    const labelX = clamp(x + 9, 4, Math.max(4, mapWidth.value - labelWidth - 4));
    const labelY = y < 22 ? y + 9 : y - 21;
    context.fillStyle = 'rgba(255, 255, 255, 0.96)';
    context.fillRect(labelX, labelY, labelWidth, 18);
    context.strokeStyle = '#d7dce6';
    context.lineWidth = 1;
    context.strokeRect(labelX, labelY, labelWidth, 18);
    context.fillStyle = '#30384a';
    context.fillText(label, labelX + 5, labelY + 9);
  }

  if (hoveredEntry.value) {
    const entry = hoveredEntry.value;
    const screenX = (entry.x - currentViewport.x) * mapWidth.value * currentViewport.k;
    const screenY = (entry.y - currentViewport.y) * mapHeight.value * currentViewport.k;
    context.beginPath();
    context.strokeStyle = '#1f2937';
    context.lineWidth = 1.5;
    context.arc(screenX, screenY, radius + 5, 0, Math.PI * 2);
    context.stroke();
  }
}

function scheduleDraw(): void {
  if (drawFrame) return;
  drawFrame = requestAnimationFrame(() => {
    drawFrame = 0;
    drawMap();
  });
}

function exportPNG(): void {
  const canvas = mapCanvas.value;
  if (!canvas || !visiblePoints.value.length || typeof document === 'undefined') return;
  const width = mapWidth.value;
  const height = mapHeight.value;
  if (!width || !height) return;

  const output = document.createElement('canvas');
  const headerHeight = 48;
  const footerHeight = 28;
  output.width = Math.round(width * 2);
  output.height = Math.round((height + headerHeight + footerHeight) * 2);
  const context = output.getContext('2d');
  if (!context) return;
  context.scale(2, 2);
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, width, height + headerHeight + footerHeight);
  context.fillStyle = '#1f2937';
  context.font = '600 16px Inter, ui-sans-serif, system-ui, sans-serif';
  context.textBaseline = 'top';
  context.fillText(title.value, 14, 11);
  context.fillStyle = '#667085';
  context.font = '11px Inter, ui-sans-serif, system-ui, sans-serif';
  context.fillText(`${visiblePoints.value.length} visible · threshold ${threshold.value.toFixed(3)} · ${Math.round(viewport.value.k * 100)}% zoom`, 14, 31);
  context.drawImage(canvas, 0, headerHeight, width, height);
  context.strokeStyle = '#d7dce6';
  context.beginPath();
  context.moveTo(0, headerHeight + 0.5);
  context.lineTo(width, headerHeight + 0.5);
  context.stroke();

  output.toBlob(blob => {
    if (!blob || typeof URL.createObjectURL !== 'function') return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const safeTitle = title.value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'neuron-map';
    link.href = url;
    link.download = `${safeTitle}.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }, 'image/png');
}

watch(
  [visiblePoints, viewport, hoveredId, selectedId, threshold, mapWidth, mapHeight],
  scheduleDraw,
  { flush: 'post' },
);

watch(visiblePoints, entries => {
  if (hoveredId.value && !entries.some(entry => entry.point.id === hoveredId.value)) {
    hoveredId.value = null;
  }
});

onMounted(() => {
  const area = mapArea.value;
  if (area) {
    const updateSize = (width: number, height: number) => {
      const nextWidth = Math.max(0, Math.floor(width));
      const nextHeight = Math.max(0, Math.floor(height));
      if (mapWidth.value !== nextWidth) mapWidth.value = nextWidth;
      if (mapHeight.value !== nextHeight) mapHeight.value = nextHeight;
    };
    const initialRect = area.getBoundingClientRect();
    updateSize(initialRect.width, initialRect.height);
    resizeObserver = new ResizeObserver(entries => {
      const entry = entries[0];
      if (entry) updateSize(entry.contentRect.width, entry.contentRect.height);
    });
    resizeObserver.observe(area);
  }
  window.addEventListener('keydown', onGlobalShortcut);
  scheduleDraw();
});

onBeforeUnmount(() => {
  resizeObserver?.disconnect();
  window.removeEventListener('keydown', onGlobalShortcut);
  if (drawFrame) cancelAnimationFrame(drawFrame);
});
</script>

<style scoped>
.neuron-map-shell {
  --border: #d9dee8;
  --border-strong: #c7cdd9;
  --text: #20283a;
  --muted: #667085;
  --accent: #2563eb;
  --accent-soft: #eaf1ff;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  width: 100%;
  min-width: 0;
  min-height: 480px;
  overflow: hidden;
  color: var(--text);
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 10px;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  box-shadow: 0 1px 2px rgb(16 24 40 / 5%);
}

.map-toolbar {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
  padding: 10px 12px;
  background: #ffffff;
  border-bottom: 1px solid var(--border);
}

.heading {
  min-width: 145px;
  padding-top: 2px;
}

.heading h2 {
  margin: 0;
  color: var(--text);
  font-size: 13px;
  font-weight: 700;
  line-height: 1.3;
}

.heading p {
  margin: 3px 0 0;
  color: var(--muted);
  font-size: 10px;
  line-height: 1.35;
  font-variant-numeric: tabular-nums;
}

.controls {
  display: flex;
  flex: 1;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: flex-end;
  gap: 7px 9px;
}

.search-control,
.select-control,
.threshold-control {
  display: grid;
  gap: 3px;
  color: var(--muted);
  font-size: 10px;
  font-weight: 600;
}

.search-control {
  width: min(235px, 100%);
}

.search-row {
  display: flex;
  gap: 4px;
}

.search-control input,
.select-control select,
.threshold-control input {
  min-height: 30px;
  color: var(--text);
  background: #ffffff;
  border: 1px solid var(--border-strong);
  border-radius: 6px;
}

.search-control input {
  width: 100%;
  min-width: 0;
  padding: 5px 8px;
  font: inherit;
  font-size: 11px;
}

.select-control select {
  min-width: 88px;
  padding: 4px 24px 4px 7px;
  font: inherit;
  font-size: 11px;
}

.threshold-control {
  grid-template-columns: auto 82px 34px;
  align-items: center;
  gap: 6px;
  min-height: 30px;
}

.threshold-control input {
  width: 82px;
  min-height: 20px;
  accent-color: var(--accent);
}

.threshold-control output {
  color: var(--text);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}

.button-row {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

button {
  min-height: 30px;
  padding: 5px 8px;
  color: var(--text);
  background: #ffffff;
  border: 1px solid var(--border-strong);
  border-radius: 6px;
  font: inherit;
  font-size: 11px;
  font-weight: 600;
  line-height: 1;
  cursor: pointer;
}

button:hover:not(:disabled) {
  color: #1d4ed8;
  background: #f8f7fd;
  border-color: #a9a1dc;
}

button:focus-visible,
input:focus-visible,
select:focus-visible,
canvas:focus-visible {
  outline: 2px solid #315bc4;
  outline-offset: 2px;
}

button:disabled {
  color: #98a2b3;
  background: #f8f9fb;
  cursor: not-allowed;
}

.map-workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 238px;
  min-width: 0;
  min-height: 0;
}

.map-area {
  position: relative;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  background: #ffffff;
  border-right: 1px solid var(--border);
}

.map-canvas {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 180px;
  cursor: grab;
  touch-action: none;
}

.map-canvas:active {
  cursor: grabbing;
}

.legend {
  position: absolute;
  bottom: 10px;
  left: 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 7px 9px;
  color: var(--muted);
  background: rgb(255 255 255 / 94%);
  border: 1px solid var(--border);
  border-radius: 7px;
  font-size: 10px;
  line-height: 1.2;
  pointer-events: none;
}

.legend span {
  display: flex;
  align-items: center;
  gap: 6px;
}

.legend i {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.positive-dot {
  background: #19764a;
}

.negative-dot {
  background: #b42318;
}

.neutral-dot {
  background: #7b8494;
}

.point-tooltip {
  position: absolute;
  z-index: 4;
  display: grid;
  gap: 2px;
  max-width: 216px;
  padding: 7px 9px;
  color: #30384a;
  background: #ffffff;
  border: 1px solid #b9c0cf;
  border-radius: 7px;
  box-shadow: 0 5px 16px rgb(16 24 40 / 13%);
  font-size: 10px;
  line-height: 1.35;
  pointer-events: none;
}

.point-tooltip strong {
  color: #1f2937;
  font-size: 11px;
}

.state-overlay {
  position: absolute;
  inset: 0;
  z-index: 5;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 24px;
  color: var(--muted);
  background: rgb(255 255 255 / 96%);
  text-align: center;
  font-size: 12px;
}

.state-overlay strong {
  color: var(--text);
  font-size: 13px;
}

.state-overlay span:not(.spinner) {
  max-width: 320px;
  line-height: 1.45;
}

.spinner {
  width: 22px;
  height: 22px;
  border: 2px solid #d9d6ef;
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: neuron-spin 0.8s linear infinite;
}

.point-panel {
  display: flex;
  min-width: 0;
  min-height: 0;
  flex-direction: column;
  background: #fbfcfe;
}

.selected-summary {
  display: grid;
  gap: 2px;
  padding: 9px 10px;
  color: var(--muted);
  background: var(--accent-soft);
  border-bottom: 1px solid #dedaf5;
  font-size: 10px;
}

.selected-summary strong {
  overflow: hidden;
  color: #1d4ed8;
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.selected-summary small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.list-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  padding: 9px 10px 7px;
  border-bottom: 1px solid var(--border);
}

.list-heading strong {
  color: var(--text);
  font-size: 11px;
}

.list-heading span {
  color: var(--muted);
  font-size: 9px;
  text-align: right;
}

.point-list {
  flex: 1;
  min-height: 0;
  margin: 0;
  padding: 4px;
  overflow: auto;
  list-style: none;
  scrollbar-color: #c5cad5 transparent;
  scrollbar-width: thin;
}

.point-list li + li {
  margin-top: 3px;
}

.point-list button {
  display: grid;
  width: 100%;
  min-height: 46px;
  gap: 3px;
  padding: 7px 8px;
  text-align: left;
  background: #ffffff;
  border-color: transparent;
  font-weight: 400;
}

.point-list button:hover:not(:disabled) {
  background: #f6f5fc;
  border-color: #dedaf5;
}

.point-list button[aria-current="true"] {
  background: var(--accent-soft);
  border-color: #a9a1dc;
}

.point-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.point-row strong {
  overflow: hidden;
  color: #2f3748;
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.activation {
  flex: none;
  font-size: 10px;
  font-variant-numeric: tabular-nums;
}

.activation.positive {
  color: #19764a;
}

.activation.negative {
  color: #b42318;
}

.activation.neutral {
  color: #667085;
}

.point-meta {
  overflow: hidden;
  color: var(--muted);
  font-size: 9px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.empty-list {
  margin: 0;
  padding: 14px 10px;
  color: var(--muted);
  font-size: 11px;
  line-height: 1.4;
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

@keyframes neuron-spin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 760px) {
  .map-toolbar {
    display: grid;
    gap: 8px;
  }

  .heading {
    min-width: 0;
  }

  .controls {
    justify-content: flex-start;
  }

  .map-workspace {
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: minmax(180px, 1fr) 172px;
  }

  .map-area {
    border-right: 0;
    border-bottom: 1px solid var(--border);
  }

  .point-list {
    display: grid;
    grid-auto-columns: minmax(190px, 1fr);
    grid-auto-flow: column;
    overflow: auto hidden;
  }

  .point-list li + li {
    margin-top: 0;
    margin-left: 3px;
  }
}

@media (max-width: 430px) {
  .search-control {
    width: 100%;
  }

  .threshold-control {
    grid-template-columns: auto minmax(70px, 1fr) 32px;
  }

  .threshold-control input {
    width: 100%;
  }
}

@media (prefers-reduced-motion: reduce) {
  .spinner {
    animation-duration: 1.8s;
  }
}
</style>
