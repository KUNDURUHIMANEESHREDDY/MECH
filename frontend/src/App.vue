<template>
  <div class="mech-app" data-testid="app-shell" :class="{ 'nav-collapsed': sidebarCollapsed }">
    <aside class="app-nav" aria-label="Tool navigation" :class="{ 'is-collapsed': sidebarCollapsed }">
      <div class="brand" aria-label="MECH">
        <span class="brand-mark" aria-hidden="true">M</span>
        <span class="brand-name">MECH</span>
        <button
          type="button"
          class="collapse-button"
          aria-label="Collapse navigation"
          aria-controls="tool-navigation-groups"
          title="Collapse navigation"
          data-testid="nav-collapse"
          @click="setSidebarCollapsed(true)"
        >
          <span aria-hidden="true">«</span>
        </button>
      </div>

      <nav class="nav-groups" id="tool-navigation-groups">
        <div v-for="group in navGroups" :key="group.name" class="nav-group">
          <h2 class="nav-group__title">{{ group.name }}</h2>
          <button
            v-for="route in group.routes"
            :key="route.id"
            type="button"
            class="nav-item"
            :class="{ 'is-active': route.id === highlightedRouteId }"
            :aria-current="route.id === highlightedRouteId ? 'page' : undefined"
            :data-testid="`nav-${route.id}`"
            @click="openRoute(route.id)"
          >
            {{ route.label }}
          </button>
        </div>
      </nav>

      <div class="nav-status" data-testid="runtime-status">
        <span class="status-dot" :class="`status-${status.state}`" aria-hidden="true" />
        <span class="nav-status__text" role="status" aria-live="polite" :title="status.detail">
          {{ status.label }}
        </span>
        <button
          v-if="status.retryable"
          type="button"
          class="status-retry"
          data-testid="runtime-retry"
          :disabled="probing"
          :title="status.detail"
          @click="probe()"
        >
          {{ probing ? 'Retrying…' : 'Retry' }}
        </button>
      </div>
    </aside>

    <main class="app-main" :aria-busy="pendingRoute ? 'true' : 'false'">
      <button
        v-if="sidebarCollapsed"
        type="button"
        class="expand-button"
        aria-label="Expand navigation"
        aria-controls="tool-navigation-groups"
        title="Expand navigation"
        data-testid="nav-expand"
        @click="setSidebarCollapsed(false)"
      >
        <span aria-hidden="true">»</span>
        <span>Menu</span>
      </button>
      <div v-if="unknownRoute" class="not-found" role="alert">
        <p class="desktop-kicker">Unknown tool</p>
        <strong>No MECH tool matches “{{ unknownRoute }}”</strong>
        <p>Choose a tool from the navigation.</p>
        <button class="desktop-button primary" type="button" @click="openRoute('explorer')">Open Model Explorer</button>
      </div>

      <section
        v-else-if="pendingRoute"
        class="route-loading"
        role="status"
        :data-testid="`loading-${pendingRoute.id}`"
      >
        <div>
          <div class="route-loading__bar" aria-hidden="true"></div>
          <strong>Loading {{ pendingRoute.label }}</strong>
        </div>
      </section>

      <section
        v-else-if="activeRoute"
        class="tool-view"
        :data-testid="`view-${activeRoute.id}`"
        :aria-label="activeRoute.label"
      >
        <div v-if="loadErrors[activeRoute.id]" class="route-error" role="alert">
          <div>
            <strong>This tool could not be loaded.</strong>
            <p>{{ loadErrors[activeRoute.id] }}</p>
          </div>
        </div>
        <Suspense v-else>
          <component :is="componentFor(activeRoute.id)" />
          <template #fallback>
            <div class="route-loading" role="status">
              <div>
                <div class="route-loading__bar" aria-hidden="true"></div>
                <strong>Loading {{ activeRoute.label }}</strong>
              </div>
            </div>
          </template>
        </Suspense>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import {
  computed,
  defineAsyncComponent,
  markRaw,
  onBeforeUnmount,
  onMounted,
  reactive,
  ref,
} from 'vue';
import { ROUTES, getRouteById } from './desktop/routeRegistry';
import { apiFetch, apiUrl } from './services/api';
import {
  classifyProbe,
  connectingStatus,
  type ProbeResult,
  type RuntimeStatus,
} from './services/runtimeStatus';

const unknownRoute = ref('');
const activeRouteId = ref('explorer');
/**
 * Route whose chunk is still loading.
 *
 * `<Suspense>` keeps the previously resolved subtree on screen while a new
 * async component resolves, so clicking a nav item used to leave the last
 * tool's DOM -- labels, results and all -- inside the newly addressed view.
 * The slug said "Circuit Explorer" while the neurons from Model Explorer were
 * still rendered underneath it. Resolving the chunk before switching means the
 * shell never shows one tool's content under another tool's name.
 */
const pendingRouteId = ref('');
let navSequence = 0;
const sidebarCollapsed = ref(readSidebarCollapsed());
const loadErrors = reactive<Record<string, string>>({});
/** Connection state for the sidebar pill. Measured by a probe, never guessed. */
const status = ref<RuntimeStatus>(connectingStatus());
const probing = ref(false);
const componentCache = new Map<string, ReturnType<typeof defineAsyncComponent>>();

const GROUP_ORDER = ['Explore', 'Develop', 'Research', 'General'] as const;

const navGroups = computed(() =>
  GROUP_ORDER.map(name => ({
    name,
    routes: ROUTES.filter(route => route.group === name && !route.aliasOf),
  })).filter(group => group.routes.length > 0),
);

const activeRoute = computed(() => getRouteById(activeRouteId.value));

const pendingRoute = computed(() =>
  pendingRouteId.value ? getRouteById(pendingRouteId.value) : null,
);

// Sidebar highlights the canonical entry when an alias slug is open,
// so keyboard and screen-reader users always see where they are. The target of
// an in-flight switch is highlighted immediately: it is what the user asked
// for, and the main region is marked busy until it renders.
const highlightedRouteId = computed(
  () =>
    pendingRoute.value?.aliasOf ??
    pendingRoute.value?.id ??
    activeRoute.value?.aliasOf ??
    activeRouteId.value,
);

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

function setSidebarCollapsed(collapsed: boolean) {
  sidebarCollapsed.value = collapsed;
  try {
    window.localStorage.setItem('mech-nav-collapsed', collapsed ? '1' : '0');
  } catch {
    // Private browsing or disabled storage: collapse still works per session.
  }
}

function readSidebarCollapsed(): boolean {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return false;
    return window.localStorage.getItem('mech-nav-collapsed') === '1';
  } catch {
    return false;
  }
}

function setHash(routeId: string) {
  if (typeof window === 'undefined') return;
  if (window.location.hash !== `#${routeId}`) window.location.hash = routeId;
}

async function openRoute(routeId: string, updateHash = true): Promise<void> {
  const route = getRouteById(routeId);
  if (!route) {
    unknownRoute.value = routeId;
    pendingRouteId.value = '';
    if (updateHash) setHash(routeId);
    return;
  }

  unknownRoute.value = '';
  if (updateHash) setHash(route.id);
  if (route.id === activeRouteId.value && !pendingRouteId.value) return;

  const sequence = (navSequence += 1);
  pendingRouteId.value = route.id;
  try {
    // Preload the chunk so the swap is a swap, not a layering of two tools.
    await route.load();
  } catch (error) {
    // Show the tool's own error state rather than silently keeping the old one.
    loadErrors[route.id] = error instanceof Error ? error.message : String(error);
  }
  if (sequence !== navSequence) return;

  pendingRouteId.value = '';
  activeRouteId.value = route.id;
}

function routeFromHash(): string {
  return window.location.hash.replace(/^#/, '').trim();
}

function handleHashChange() {
  const routeId = routeFromHash() || 'explorer';
  openRoute(routeId, false);
}

const PROBE_TIMEOUT_MS = 5_000;

/**
 * Measure whether the control plane answers, then describe it honestly.
 *
 * Two claims were wrong here before:
 *  - the probe was `GET /api/models`, which serves a static reference
 *    catalogue tagged `provenance: "reference"`. It answers even when no
 *    weights are cached, so it cannot measure readiness;
 *  - readiness was rendered as "N models available", turning a catalogue
 *    length into a claim of research capability.
 *
 * `GET /api/status` is the readiness probe. The pill reports one of the four
 * documented states and never claims a model is loaded: that is per-view
 * state, reported by the view that actually loaded one.
 */
async function probe(): Promise<void> {
  const previous = status.value.state;
  probing.value = true;
  status.value = connectingStatus();

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
  let result: ProbeResult;
  try {
    const response = await apiFetch(apiUrl('/api/status'), { signal: controller.signal });
    result = response.ok
      ? { ok: true }
      : {
          ok: false,
          status: response.status,
          message: `GET /api/status answered HTTP ${response.status}.`,
        };
  } catch (error) {
    const message =
      error instanceof DOMException && error.name === 'AbortError'
        ? `GET /api/status timed out after ${PROBE_TIMEOUT_MS} ms.`
        : error instanceof Error
          ? error.message
          : String(error);
    result = { ok: false, message };
  } finally {
    clearTimeout(timeout);
  }

  status.value = classifyProbe(result, previous);
  probing.value = false;
}

onMounted(() => {
  const initialRoute = routeFromHash() || 'explorer';
  void openRoute(initialRoute, false);

  window.addEventListener('hashchange', handleHashChange);
  // `disconnected` is only reachable if a drop is observed, so re-probe when
  // the window regains attention instead of assuming the mount-time answer
  // holds for the life of the session.
  window.addEventListener('focus', reprobeOnReturn);
  document.addEventListener('visibilitychange', reprobeOnReturn);
  void probe();
});

function reprobeOnReturn() {
  if (document.visibilityState === 'hidden' || probing.value) return;
  void probe();
}

onBeforeUnmount(() => {
  window.removeEventListener('hashchange', handleHashChange);
  window.removeEventListener('focus', reprobeOnReturn);
  document.removeEventListener('visibilitychange', reprobeOnReturn);
});
</script>

<style scoped>
.mech-app {
  display: grid;
  width: 100vw;
  height: 100dvh;
  min-width: 320px;
  grid-template-columns: 232px minmax(0, 1fr);
  overflow: hidden;
  background: var(--bg);
  color: var(--text);
  color-scheme: light;
  transition: grid-template-columns 0.18s ease-out;
}

.mech-app.nav-collapsed {
  grid-template-columns: 0 minmax(0, 1fr);
}

.app-nav {
  display: flex;
  min-height: 0;
  flex-direction: column;
  overflow: hidden;
  border-right: 1px solid var(--border);
  background: var(--bg-elev);
  white-space: nowrap;
}

.app-nav.is-collapsed {
  visibility: hidden;
  border-right: 0;
}

.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 14px 16px 12px;
}

.brand-mark {
  display: grid;
  width: 24px;
  height: 24px;
  place-items: center;
  border-radius: 6px;
  background: var(--text);
  color: var(--bg);
  font-size: 13px;
  font-weight: 800;
}

.brand-name {
  flex: 1;
  font-size: 14px;
  font-weight: 750;
  letter-spacing: 0.04em;
}

.collapse-button {
  display: inline-flex;
  width: 26px;
  height: 26px;
  flex: none;
  align-items: center;
  justify-content: center;
  padding: 0;
  border: 1px solid transparent;
  border-radius: 6px;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 15px;
  line-height: 1;
}

.collapse-button:hover {
  border-color: var(--border);
  background: var(--bg-hover);
  color: var(--text);
}

.expand-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 12px;
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-elev);
  color: var(--text-dim);
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
}

.expand-button:hover {
  background: var(--bg-hover);
  color: var(--text);
}

.nav-groups {
  display: flex;
  flex: 1;
  min-height: 0;
  flex-direction: column;
  gap: 14px;
  overflow-y: auto;
  padding: 2px 10px 12px;
}

.nav-group__title {
  margin: 0 0 4px 6px;
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.nav-item {
  display: block;
  width: 100%;
  padding: 7px 10px;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: var(--text-dim);
  cursor: pointer;
  font-size: 13px;
  text-align: left;
}

.nav-item:hover {
  background: var(--bg-hover);
  color: var(--text);
}

.nav-item.is-active {
  background: var(--bg-active);
  color: var(--text);
  font-weight: 650;
}

.nav-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 11px;
}

.status-dot {
  width: 8px;
  height: 8px;
  flex: none;
  border-radius: 50%;
  background: var(--text-muted);
}

/* State is carried by the label text; the dot is a second, redundant cue.
   `disconnected` is a ring instead of a solid dot so it stays distinguishable
   from `failed` without relying on colour perception. */
.status-dot.status-ready {
  background: var(--success);
}

.status-dot.status-connecting {
  background: var(--warning);
}

.status-dot.status-disconnected {
  background: transparent;
  box-shadow: inset 0 0 0 2px var(--danger);
}

.status-dot.status-failed {
  background: var(--danger);
}

.status-retry {
  margin-left: auto;
  flex: none;
  padding: 2px 7px;
  border: 1px solid var(--border);
  border-radius: 5px;
  background: var(--bg);
  color: var(--text);
  font-size: 10px;
  font-weight: 600;
  cursor: pointer;
}

.status-retry:hover:not(:disabled) {
  background: var(--bg-active);
}

.status-retry:disabled {
  cursor: progress;
  opacity: 0.65;
}

.nav-status__text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.app-main {
  min-width: 0;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 20px 22px 32px;
}

.tool-view {
  max-width: 1180px;
  margin: 0 auto;
}

.not-found {
  display: flex;
  max-width: 520px;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
  margin: 48px auto;
  text-align: left;
}

@media (max-width: 900px) {
  .mech-app {
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: auto minmax(0, 1fr);
  }

  .app-nav {
    border-right: 0;
    border-bottom: 1px solid var(--border);
  }

  .brand {
    padding: 10px 14px 6px;
  }

  .nav-groups {
    flex-direction: row;
    gap: 14px;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0 10px 8px;
  }

  .nav-group {
    display: flex;
    flex: none;
    align-items: center;
    gap: 2px;
  }

  .nav-group__title {
    margin: 0 4px 0 0;
  }

  .nav-item {
    width: auto;
    padding: 6px 9px;
    white-space: nowrap;
  }

  .nav-status {
    display: none;
  }

  .collapse-button,
  .expand-button {
    display: none;
  }

  .app-main {
    padding: 14px 14px 24px;
  }
}
</style>
