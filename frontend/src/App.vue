<template>
  <div class="mech-app" data-testid="app-shell">
    <aside class="app-nav" aria-label="Tool navigation">
      <div class="brand" aria-label="MECH">
        <span class="brand-mark" aria-hidden="true">M</span>
        <span class="brand-name">MECH</span>
      </div>

      <nav class="nav-groups">
        <div v-for="group in navGroups" :key="group.name" class="nav-group">
          <h2 class="nav-group__title">{{ group.name }}</h2>
          <button
            v-for="route in group.routes"
            :key="route.id"
            type="button"
            class="nav-item"
            :class="{ 'is-active': route.id === activeRouteId }"
            :aria-current="route.id === activeRouteId ? 'page' : undefined"
            :data-testid="`nav-${route.id}`"
            @click="openRoute(route.id)"
          >
            {{ route.label }}
          </button>
        </div>
      </nav>

      <div class="nav-status" role="status" aria-live="polite" data-testid="runtime-status">
        <span class="status-dot" :class="`status-${modelStatus}`" aria-hidden="true" />
        <span class="nav-status__text">{{ modelName }}</span>
      </div>
    </aside>

    <main class="app-main">
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
const loadErrors = reactive<Record<string, string>>({});
const componentCache = new Map<string, ReturnType<typeof defineAsyncComponent>>();

const GROUP_ORDER = ['Explore', 'Develop', 'Research', 'General'] as const;

const navGroups = computed(() =>
  GROUP_ORDER.map(name => ({
    name,
    routes: ROUTES.filter(route => route.group === name),
  })).filter(group => group.routes.length > 0),
);

const activeRoute = computed(() => getRouteById(activeRouteId.value));

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
}

.app-nav {
  display: flex;
  min-height: 0;
  flex-direction: column;
  border-right: 1px solid var(--border);
  background: var(--bg-elev);
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
  font-size: 14px;
  font-weight: 750;
  letter-spacing: 0.04em;
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

  .app-main {
    padding: 14px 14px 24px;
  }
}
</style>
