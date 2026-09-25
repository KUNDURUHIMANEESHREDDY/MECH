<template>
  <div class="mech-desktop" data-testid="desktop-shell">
    <DesktopMenuBar
      :model-status="modelStatus"
      :model-name="modelName"
      @open-route="openRoute"
      @tile="tileWindows"
      @reset="resetWindows"
      @toggle-directory="directoryOpen = !directoryOpen"
    />

    <section
      ref="desktopElement"
      class="desktop-workspace"
      aria-label="MECH desktop workspace"
      @pointerdown="handleWorkspacePointer"
    >
      <div v-if="visibleWindows.length === 0 && !unknownRoute" class="desktop-empty">
        <div>
          <span class="desktop-empty__mark" aria-hidden="true">M</span>
          <strong>No tools are open</strong>
          <p>Open a tool from the window menu or restore the default desktop layout.</p>
        </div>
      </div>

      <div v-if="unknownRoute" class="not-found-window" role="alert">
        <div>
          <div class="desktop-kicker">Unknown window</div>
          <strong>No MECH tool matches “{{ unknownRoute }}”</strong>
          <p>The route was not registered. Choose a tool from the window menu or return to Model Explorer.</p>
          <button class="desktop-button primary" type="button" @click="openRoute('explorer')">Open Model Explorer</button>
        </div>
      </div>

      <ToolWindow
        v-for="desktopWindow in visibleWindows"
        :key="desktopWindow.id"
        :id="desktopWindow.id"
        :title="routeTitle(desktopWindow.routeId)"
        :active="desktopWindow.id === activeWindowId"
        :minimized="desktopWindow.minimized"
        :maximized="desktopWindow.maximized"
        :z-index="desktopWindow.zIndex"
        :x="desktopWindow.x"
        :y="desktopWindow.y"
        :width="desktopWindow.width"
        :height="desktopWindow.height"
        @focus="focusRoute(desktopWindow.id)"
        @close="closeWindow(desktopWindow.id)"
        @minimize="minimizeWindow(desktopWindow.id)"
        @maximize="toggleMaximize(desktopWindow.id)"
        @geometry-change="geometry => updateGeometry(desktopWindow.id, geometry)"
      >
        <div class="desktop-route-content">
          <div
            v-if="modelStatus === 'offline'"
            class="desktop-provenance"
            data-kind="offline"
            role="status"
          >
            <span>Backend offline — local tools remain available.</span>
            <span class="mono">localhost:8000</span>
          </div>
          <div v-else-if="modelStatus === 'connecting'" class="desktop-provenance" role="status">
            <span>Connecting to the MECH runtime…</span>
            <span class="mono">checking /api/models</span>
          </div>

          <div v-if="loadErrors[desktopWindow.routeId]" class="route-error" role="alert">
            <div>
              <strong>This tool could not be loaded.</strong>
              <p>{{ loadErrors[desktopWindow.routeId] }}</p>
            </div>
          </div>
          <Suspense v-else>
            <component :is="componentFor(desktopWindow.routeId)" />
            <template #fallback>
              <div class="route-loading" role="status">
                <div>
                  <div class="route-loading__bar" aria-hidden="true"></div>
                  <strong>Loading {{ routeTitle(desktopWindow.routeId) }}</strong>
                  <p>Preparing the tool window without blocking the desktop.</p>
                </div>
              </div>
            </template>
          </Suspense>
        </div>
      </ToolWindow>

      <WindowDirectory
        :open="directoryOpen"
        :active-route-id="activeWindowId"
        @close="directoryOpen = false"
        @open-route="openFromDirectory"
      />
    </section>

    <DesktopStatusBar
      :model-status="modelStatus"
      :model-name="modelName"
      :active-window-title="activeWindowTitle"
      :open-window-count="openWindowCount"
    />
  </div>
</template>

<script setup lang="ts">
import {
  computed,
  defineAsyncComponent,
  markRaw,
  nextTick,
  onBeforeUnmount,
  onMounted,
  reactive,
  ref,
} from 'vue';
import { useDesktopStore } from './stores/desktop';
import { getRouteById } from './desktop/routeRegistry';
import { api } from './services/api';
import DesktopMenuBar from './components/desktop/DesktopMenuBar.vue';
import DesktopStatusBar from './components/desktop/DesktopStatusBar.vue';
import ToolWindow from './components/desktop/ToolWindow.vue';
import WindowDirectory from './components/desktop/WindowDirectory.vue';

type RuntimeStatus = 'connecting' | 'connected' | 'offline';
type Bounds = { width: number; height: number };
type Geometry = { x: number; y: number; width: number; height: number };

const store = useDesktopStore();
const desktopElement = ref<HTMLElement | null>(null);
const directoryOpen = ref(false);
const unknownRoute = ref('');
const modelStatus = ref<RuntimeStatus>('connecting');
const modelName = ref('No model selected');
const desktopBounds = ref<Bounds>({ width: 1200, height: 760 });
const loadErrors = reactive<Record<string, string>>({});
const componentCache = new Map<string, ReturnType<typeof defineAsyncComponent>>();
let resizeObserver: ResizeObserver | null = null;

const activeWindowId = computed(() => store.activeWindowId);
const visibleWindows = computed(() =>
  Object.values(store.windows)
    .filter(window => !window.closed)
    .sort((left, right) => left.zIndex - right.zIndex),
);
const openWindowCount = computed(() => visibleWindows.value.filter(window => !window.minimized).length);
const activeWindowTitle = computed(() => {
  const active = store.windows[activeWindowId.value];
  return active && !active.closed ? routeTitle(active.routeId) : 'No active tool';
});

function routeTitle(routeId: string): string {
  return getRouteById(routeId)?.label ?? routeId;
}

function componentFor(routeId: string) {
  const cached = componentCache.get(routeId);
  if (cached) return cached;

  const route = getRouteById(routeId);
  if (!route) {
    loadErrors[routeId] = `No loader is registered for “${routeId}”.`;
    return defineAsyncComponent(() => Promise.resolve({ template: '<div></div>' }));
  }

  const component = markRaw(defineAsyncComponent({
    loader: async () => {
      const module = await route.load();
      return module.default;
    },
    onError(error) {
      loadErrors[routeId] = error instanceof Error ? error.message : String(error);
    },
  }));
  componentCache.set(routeId, component);
  return component;
}

function currentBounds(): Bounds {
  const rect = desktopElement.value?.getBoundingClientRect();
  return rect
    ? { width: Math.max(320, rect.width), height: Math.max(240, rect.height) }
    : desktopBounds.value;
}

function setHash(routeId: string) {
  if (typeof window === 'undefined') return;
  const next = `#${routeId}`;
  if (window.location.hash !== next) window.location.hash = routeId;
}

function openRoute(routeId: string, updateHash = true) {
  const route = getRouteById(routeId);
  if (!route) {
    unknownRoute.value = routeId;
    if (updateHash) setHash(routeId);
    return;
  }

  unknownRoute.value = '';
  directoryOpen.value = false;
  store.openWindow(routeId);
  if (updateHash) setHash(routeId);
}

function openFromDirectory(routeId: string) {
  openRoute(routeId);
}

function focusRoute(routeId: string) {
  if (!store.windows[routeId] || store.windows[routeId].closed) return;
  store.focusWindow(routeId);
  setHash(routeId);
}

function closeWindow(routeId: string) {
  store.closeWindow(routeId);
  const active = store.windows[activeWindowId.value];
  if (active && !active.closed) setHash(active.routeId);
  else if (window.location.hash === `#${routeId}`) history.replaceState(null, '', window.location.pathname + window.location.search);
}

function minimizeWindow(routeId: string) {
  store.minimizeWindow(routeId);
  const active = store.windows[activeWindowId.value];
  if (active && !active.closed) setHash(active.routeId);
}

function toggleMaximize(routeId: string) {
  store.toggleMaximize(routeId, currentBounds());
}

function updateGeometry(routeId: string, geometry: Geometry) {
  store.setGeometry(routeId, geometry, currentBounds());
}

function tileWindows() {
  store.tileWindows(currentBounds(), 14);
}

function resetWindows() {
  const currentRoute = routeFromHash() || 'explorer';
  store.resetWindows();
  openRoute(getRouteById(currentRoute) ? currentRoute : 'explorer', false);
}

function handleWorkspacePointer(event: PointerEvent) {
  const target = event.target as HTMLElement | null;
  if (target?.closest('.desktop-tool-window') || target?.closest('.window-directory')) return;
  const active = store.windows[activeWindowId.value];
  if (active && !active.closed) store.focusWindow(activeWindowId.value);
}

function routeFromHash(): string {
  return window.location.hash.replace(/^#/, '').trim();
}

function handleHashChange() {
  const routeId = routeFromHash() || 'explorer';
  openRoute(routeId, false);
  nextTick(() => focusRoute(store.activeWindowId));
}

async function refreshRuntimeStatus() {
  modelStatus.value = 'connecting';
  try {
    const result = await api.listModels();
    modelStatus.value = 'connected';
    if (Array.isArray(result.models) && result.models.length > 0 && modelName.value === 'No model selected') {
      modelName.value = `${result.models.length} models available`;
    }
  } catch {
    modelStatus.value = 'offline';
  }
}

onMounted(() => {
  const initialRoute = routeFromHash() || 'explorer';
  openRoute(initialRoute, false);

  window.addEventListener('hashchange', handleHashChange);
  refreshRuntimeStatus();

  if (desktopElement.value && 'ResizeObserver' in window) {
    resizeObserver = new ResizeObserver(entries => {
      const rect = entries[0]?.contentRect;
      if (rect) desktopBounds.value = { width: rect.width, height: rect.height };
    });
    resizeObserver.observe(desktopElement.value);
  }
});

onBeforeUnmount(() => {
  window.removeEventListener('hashchange', handleHashChange);
  resizeObserver?.disconnect();
});
</script>
