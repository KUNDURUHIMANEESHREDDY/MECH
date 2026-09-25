<template>
  <section
    v-if="!minimized"
    ref="windowElement"
    class="tool-window desktop-window desktop-tool-window"
    :class="{
      'is-active': active,
      'is-maximized': maximized,
      'is-narrow': isNarrow,
    }"
    :style="windowStyle"
    :data-testid="`window-${id}`"
    role="dialog"
    aria-modal="false"
    :aria-labelledby="titleId"
    :aria-keyshortcuts="keyboardShortcuts"
    tabindex="0"
    @pointerdown="onWindowPointerDown"
    @pointermove="handlePointerMove"
    @pointerup="handlePointerEnd"
    @pointercancel="handlePointerEnd"
    @keydown="handleKeydown"
  >
    <header
      class="title-bar window-title-bar tool-window__title-bar"
      data-testid="window-title-bar"
      aria-label="Window title bar"
      @pointerdown.stop="onTitlePointerDown"
    >
      <div class="title-label">
        <span class="title-indicator" aria-hidden="true" />
        <h2 :id="titleId" class="window-title">{{ title || 'Tool window' }}</h2>
      </div>

      <div
        class="window-controls"
        role="group"
        aria-label="Window controls"
        @pointerdown.stop
      >
        <button
          type="button"
          class="window-control"
          aria-label="Minimize window"
          title="Minimize window"
          data-testid="window-minimize"
          @click.stop="requestMinimize"
        >
          <span aria-hidden="true">−</span>
        </button>
        <button
          type="button"
          class="window-control"
          :aria-label="maximized ? 'Restore window' : 'Maximize window'"
          :title="maximized ? 'Restore window' : 'Maximize window'"
          data-testid="window-maximize"
          @click.stop="requestMaximize"
        >
          <span aria-hidden="true">{{ maximized ? '❐' : '□' }}</span>
        </button>
        <button
          type="button"
          class="window-control window-control-close"
          aria-label="Close window"
          title="Close window"
          data-testid="window-close"
          @click.stop="requestClose"
        >
          <span aria-hidden="true">×</span>
        </button>
      </div>
    </header>

    <div class="tool-window__body">
      <slot />
    </div>

    <button
      type="button"
      class="resize-handle window-resize-handle tool-window__resize-handle"
      aria-label="Resize window"
      title="Resize window"
      data-testid="window-resize-handle"
      @pointerdown.stop="onResizePointerDown"
    />
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';

interface WindowGeometry {
  x: number;
  y: number;
  width: number;
  height: number;
}

type InteractionKind = 'move' | 'resize';

interface PointerInteraction {
  kind: InteractionKind;
  pointerId: number;
  startClientX: number;
  startClientY: number;
  lastClientX: number;
  lastClientY: number;
  startGeometry: WindowGeometry;
  lastGeometry: WindowGeometry;
  emittedGeometry: WindowGeometry;
}

const MIN_WIDTH = 320;
const MIN_HEIGHT = 220;
const NARROW_SCREEN_QUERY = '(max-width: 760px)';
const keyboardShortcuts = 'ArrowUp ArrowDown ArrowLeft ArrowRight Shift+ArrowUp Shift+ArrowDown Shift+ArrowLeft Shift+ArrowRight';

const props = defineProps<{
  id: string;
  title: string;
  active: boolean;
  minimized: boolean;
  maximized: boolean;
  zIndex: number;
  x: number;
  y: number;
  width: number;
  height: number;
}>();

const emit = defineEmits<{
  (event: 'focus'): void;
  (event: 'close'): void;
  (event: 'minimize'): void;
  (event: 'maximize'): void;
  (event: 'geometry-change', geometry: WindowGeometry): void;
}>();

const windowElement = ref<HTMLElement | null>(null);
const isNarrow = ref(false);
let interaction: PointerInteraction | null = null;
let mediaQuery: MediaQueryList | null = null;

const titleId = computed(() => {
  const safeId = String(props.id).replace(/[^a-zA-Z0-9_-]/g, '-');
  return `tool-window-title-${safeId}`;
});

const windowStyle = computed<Record<string, string | number>>(() => {
  const zIndex = finiteNumber(props.zIndex, 0);

  if (isNarrow.value) {
    return {
      position: 'relative',
      left: 'auto',
      top: 'auto',
      width: '100%',
      height: 'auto',
      zIndex,
    };
  }

  if (props.maximized) {
    return {
      left: '0px',
      top: '0px',
      width: '100%',
      height: '100%',
      zIndex,
    };
  }

  const geometry = clampGeometry(props);
  return {
    left: `${geometry.x}px`,
    top: `${geometry.y}px`,
    width: `${geometry.width}px`,
    height: `${geometry.height}px`,
    zIndex,
  };
});

function finiteNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

function getBounds(): { width: number; height: number } {
  const parent = windowElement.value?.parentElement;
  const parentWidth = parent?.clientWidth ?? 0;
  const parentHeight = parent?.clientHeight ?? 0;
  const parentRect = parent && typeof parent.getBoundingClientRect === 'function'
    ? parent.getBoundingClientRect()
    : null;
  const viewportWidth = typeof window !== 'undefined' && window.innerWidth > 0
    ? window.innerWidth
    : 1280;
  const viewportHeight = typeof window !== 'undefined' && window.innerHeight > 0
    ? window.innerHeight
    : 800;

  return {
    width: finiteNumber(parentWidth || parentRect?.width || viewportWidth, 1280),
    height: finiteNumber(parentHeight || parentRect?.height || viewportHeight, 800),
  };
}

function clampGeometry(input: Partial<WindowGeometry>): WindowGeometry {
  const bounds = getBounds();
  const availableWidth = Math.max(0, bounds.width);
  const availableHeight = Math.max(0, bounds.height);
  const minimumWidth = Math.min(MIN_WIDTH, availableWidth);
  const minimumHeight = Math.min(MIN_HEIGHT, availableHeight);
  const width = clamp(
    finiteNumber(input.width, minimumWidth),
    minimumWidth,
    Math.max(minimumWidth, availableWidth),
  );
  const height = clamp(
    finiteNumber(input.height, minimumHeight),
    minimumHeight,
    Math.max(minimumHeight, availableHeight),
  );

  return {
    x: clamp(finiteNumber(input.x, 0), 0, Math.max(0, availableWidth - width)),
    y: clamp(finiteNumber(input.y, 0), 0, Math.max(0, availableHeight - height)),
    width,
    height,
  };
}

function geometriesEqual(first: WindowGeometry, second: WindowGeometry): boolean {
  return first.x === second.x
    && first.y === second.y
    && first.width === second.width
    && first.height === second.height;
}

function focusWindow() {
  emit('focus');
  try {
    windowElement.value?.focus({ preventScroll: true });
  } catch {
    windowElement.value?.focus();
  }
}

function requestClose() {
  emit('close');
}

function requestMinimize() {
  emit('minimize');
}

function requestMaximize() {
  emit('maximize');
}

function pointerTargetIsControl(event: PointerEvent): boolean {
  const target = event.target as HTMLElement | null;
  return Boolean(target?.closest?.('button, a, input, textarea, select, [contenteditable="true"]'));
}

function capturePointer(event: PointerEvent) {
  const target = event.currentTarget as HTMLElement | null;
  if (!target || typeof target.setPointerCapture !== 'function' || event.pointerId === undefined) {
    return;
  }

  try {
    target.setPointerCapture(event.pointerId);
  } catch {
    // Pointer capture is an enhancement; window listeners still finish the drag.
  }
}

function beginInteraction(
  kind: InteractionKind,
  event: PointerEvent,
): PointerInteraction | null {
  if (isNarrow.value || props.maximized || (kind === 'move' && pointerTargetIsControl(event))) {
    return null;
  }

  if (event.button !== undefined && event.button !== 0 && event.button !== -1) {
    return null;
  }

  const geometry = clampGeometry(props);
  const pointerId = finiteNumber(event.pointerId, 0);
  const startClientX = finiteNumber(event.clientX, 0);
  const startClientY = finiteNumber(event.clientY, 0);

  const nextInteraction: PointerInteraction = {
    kind,
    pointerId,
    startClientX,
    startClientY,
    lastClientX: startClientX,
    lastClientY: startClientY,
    startGeometry: geometry,
    lastGeometry: geometry,
    emittedGeometry: geometry,
  };

  interaction = nextInteraction;
  capturePointer(event);
  event.preventDefault();
  return nextInteraction;
}

function onWindowPointerDown(event: PointerEvent) {
  if (pointerTargetIsControl(event)) return;
  focusWindow();
}

function onTitlePointerDown(event: PointerEvent) {
  focusWindow();
  beginInteraction('move', event);
}

function onResizePointerDown(event: PointerEvent) {
  focusWindow();
  beginInteraction('resize', event);
}

function handlePointerMove(event: PointerEvent) {
  if (!interaction) return;
  if (event.pointerId !== undefined && event.pointerId !== interaction.pointerId) return;

  const clientX = finiteNumber(event.clientX, interaction.lastClientX);
  const clientY = finiteNumber(event.clientY, interaction.lastClientY);
  const deltaX = clientX - interaction.startClientX;
  const deltaY = clientY - interaction.startClientY;
  const start = interaction.startGeometry;
  const candidate: WindowGeometry = interaction.kind === 'move'
    ? {
        x: start.x + deltaX,
        y: start.y + deltaY,
        width: start.width,
        height: start.height,
      }
    : {
        x: start.x,
        y: start.y,
        width: start.width + deltaX,
        height: start.height + deltaY,
      };
  const next = clampGeometry(candidate);

  interaction.lastClientX = clientX;
  interaction.lastClientY = clientY;
  interaction.lastGeometry = next;

  if (!geometriesEqual(next, interaction.emittedGeometry)) {
    interaction.emittedGeometry = next;
    emit('geometry-change', next);
  }
}

function handlePointerEnd(event?: PointerEvent) {
  if (!interaction) return;
  if (event?.pointerId !== undefined && event.pointerId !== interaction.pointerId) return;

  const finished = interaction;
  interaction = null;

  if (!geometriesEqual(finished.lastGeometry, finished.emittedGeometry)) {
    finished.emittedGeometry = finished.lastGeometry;
    emit('geometry-change', finished.lastGeometry);
  }

  if (event && typeof (event.currentTarget as HTMLElement | null)?.releasePointerCapture === 'function') {
    try {
      (event.currentTarget as HTMLElement).releasePointerCapture(event.pointerId);
    } catch {
      // The pointer may already have been released by the browser.
    }
  }
}

function isEditableTarget(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null;
  return Boolean(element?.matches?.('input, textarea, select, [contenteditable="true"]'));
}

function handleKeydown(event: KeyboardEvent) {
  if (isNarrow.value || props.maximized || event.defaultPrevented || isEditableTarget(event.target)) {
    return;
  }

  const arrows: Record<string, [number, number]> = {
    ArrowUp: [0, -1],
    ArrowDown: [0, 1],
    ArrowLeft: [-1, 0],
    ArrowRight: [1, 0],
  };
  const direction = arrows[event.key];
  if (!direction) return;

  event.preventDefault();
  const step = event.altKey ? 1 : 10;
  const current = clampGeometry(props);
  const next = event.shiftKey
    ? clampGeometry({
        x: current.x,
        y: current.y,
        width: current.width + direction[0] * step,
        height: current.height + direction[1] * step,
      })
    : clampGeometry({
        x: current.x + direction[0] * step,
        y: current.y + direction[1] * step,
        width: current.width,
        height: current.height,
      });

  emit('geometry-change', next);
}

function updateNarrowState() {
  if (mediaQuery) {
    isNarrow.value = mediaQuery.matches;
    return;
  }

  isNarrow.value = typeof window !== 'undefined' && window.innerWidth <= 760;
}

onMounted(() => {
  if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
    mediaQuery = window.matchMedia(NARROW_SCREEN_QUERY);
    updateNarrowState();

    if (typeof mediaQuery.addEventListener === 'function') {
      mediaQuery.addEventListener('change', updateNarrowState);
    } else if (typeof mediaQuery.addListener === 'function') {
      mediaQuery.addListener(updateNarrowState);
    }
  } else {
    updateNarrowState();
  }

  if (typeof window !== 'undefined') {
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerEnd);
    window.addEventListener('pointercancel', handlePointerEnd);
  }
});

onBeforeUnmount(() => {
  if (typeof window !== 'undefined') {
    window.removeEventListener('pointermove', handlePointerMove);
    window.removeEventListener('pointerup', handlePointerEnd);
    window.removeEventListener('pointercancel', handlePointerEnd);
  }

  if (mediaQuery) {
    if (typeof mediaQuery.removeEventListener === 'function') {
      mediaQuery.removeEventListener('change', updateNarrowState);
    } else if (typeof mediaQuery.removeListener === 'function') {
      mediaQuery.removeListener(updateNarrowState);
    }
  }
});
</script>

<style scoped>
.tool-window {
  position: absolute;
  z-index: 1;
  display: flex;
  min-width: 0;
  min-height: 0;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  background: var(--bg-elev);
  color: var(--text);
  box-shadow: var(--shadow-sm);
  color-scheme: light;
  isolation: isolate;
}

.tool-window.is-active {
  border-color: var(--primary);
  box-shadow: var(--shadow-lg);
}

.title-bar {
  display: flex;
  min-height: 38px;
  flex: 0 0 38px;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 0 7px 0 12px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-sidebar);
  cursor: grab;
  user-select: none;
  touch-action: none;
}

.title-bar:active {
  cursor: grabbing;
}

.title-label {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
}

.title-indicator {
  width: 7px;
  height: 7px;
  flex: 0 0 auto;
  border: 1px solid var(--border-light);
  border-radius: 50%;
  background: var(--bg-elev);
}

.is-active .title-indicator {
  border-color: var(--primary);
  background: var(--primary);
}

.window-title {
  min-width: 0;
  margin: 0;
  overflow: hidden;
  color: var(--text);
  font-size: 12px;
  font-weight: 700;
  line-height: 1.3;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.window-controls {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 2px;
}

.window-control {
  display: inline-grid;
  width: 28px;
  height: 26px;
  place-items: center;
  padding: 0;
  border: 1px solid transparent;
  border-radius: 5px;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font: 600 15px/1 var(--font);
}

.window-control:hover {
  border-color: var(--border);
  background: var(--bg-hover);
  color: var(--text);
}

.window-control-close:hover {
  background: var(--danger-soft, var(--bg-hover));
  color: var(--danger);
}

.tool-window__body {
  min-width: 0;
  min-height: 0;
  flex: 1 1 auto;
  overflow: auto;
  background: var(--bg-elev);
}

.resize-handle {
  position: absolute;
  right: 0;
  bottom: 0;
  z-index: 2;
  width: 18px;
  height: 18px;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: nwse-resize;
  touch-action: none;
}

.resize-handle::after {
  position: absolute;
  right: 4px;
  bottom: 4px;
  width: 8px;
  height: 8px;
  border-right: 1px solid var(--text-muted);
  border-bottom: 1px solid var(--text-muted);
  content: '';
  transform: rotate(45deg);
}

.resize-handle:hover::after,
.resize-handle:focus-visible::after {
  border-color: var(--primary);
}

@media (max-width: 760px) {
  .tool-window,
  .tool-window.is-maximized {
    position: relative !important;
    top: auto !important;
    left: auto !important;
    width: 100% !important;
    height: auto !important;
    min-height: 240px;
    max-height: none;
    margin: 0 0 12px;
    border-radius: var(--radius);
  }

  .tool-window__body {
    min-height: 200px;
  }

  .resize-handle {
    display: none;
  }
}

@media (prefers-reduced-motion: reduce) {
  .tool-window {
    scroll-behavior: auto;
  }
}
</style>
