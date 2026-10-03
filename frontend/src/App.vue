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

      <div class="nav-status" role="status" aria-live="polite" data-testid="runtime-status">
        <span class="status-dot" :class="`status-${modelStatus}`" aria-hidden="true" />
        <span class="nav-status__text">{{ statusLabel }}</span>
      </div>
    </aside>

    <main class="app-main">
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
import { api } from './services/api';

type RuntimeStatus = 'connecting' | 'connected' | 'offline';

const unknownRoute = ref('');
const modelStatus = ref<RuntimeStatus>('connecting');
const modelName = ref('No model selected');
const activeRouteId = ref('explorer');
const sidebarCollapsed = ref(readSidebarCollapsed());
const loadErrors = reactive<Record<string, string>>({});
const componentCache = new Map<string, ReturnType<typeof defineAsyncComponent>>();

const GROUP_ORDER = ['Explore', 'Develop', 'Research', 'General'] as const;

/**
 * Label for the nav status pill.
 *
 * This used to render `modelName` unconditionally, so when the backend was
 * unreachable the dot went red (modelStatus = 'offline') while the text still
 * read "No model selected" -- reporting an infrastructure failure as a user
 * choice, which is precisely the fabrication the rest of the app is careful to
 * avoid. tests/playwright/trust.spec.js asserts the offline wording.
 *
 * Derived rather than assigned on error, so a transient outage does not
 * permanently overwrite the model the user had selected.
 */
const statusLabel = computed(() => {
  if (modelStatus.value === 'connecting') return 'Connecting to backend…';
  if (modelStatus.value === 'offline') return 'Offline — backend unreachable';
  return modelName.value;
});

const navGroups = computed(() =>
  GROUP_ORDER.map(name => ({
    name,
    routes: ROUTES.filter(route => route.group === name && !route.aliasOf),
  })).filter(group => group.routes.length > 0),
);

const activeRoute = computed(() => getRouteById(activeRouteId.value));

// Sidebar highlights the canonical entry when an alias slug is open,
// so keyboard and screen-reader users always see where they are.
const highlightedRouteId = computed(() => activeRoute.value?.aliasOf ?? activeRouteId.value);

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

function openRoute(routeId: string, updateHash = true) {
  const route = getRouteById(routeId);
  if (!route) {
    unknownRoute.value = routeId;
    if (updateHash) setHash(routeId);
    return;
  }

  unknownRoute.value = '';
  activeRouteId.value = route.id;
  if (updateHash) setHash(route.id);
}

function routeFromHash(): string {
  return window.location.hash.replace(/^#/, '').trim();
}

function handleHashChange() {
  const routeId = routeFromHash() || 'explorer';
  openRoute(routeId, false);
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
});

onBeforeUnmount(() => {
  window.removeEventListener('hashchange', handleHashChange);
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

.status-dot.status-connected {
  background: var(--success);
}

.status-dot.status-connecting {
  background: var(--warning);
}

.status-dot.status-offline {
  background: var(--danger);
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
