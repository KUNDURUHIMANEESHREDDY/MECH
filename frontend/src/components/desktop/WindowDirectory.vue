<template>
  <aside
    v-if="open"
    id="window-directory"
    class="window-directory"
    role="dialog"
    aria-modal="false"
    aria-labelledby="window-directory-title"
    data-testid="window-directory"
    @keydown.esc.stop="closeDirectory"
  >
    <header class="directory-header">
      <div>
        <p class="directory-kicker">All Windows</p>
        <h2 id="window-directory-title" class="directory-title">Directory</h2>
      </div>
      <button
        type="button"
        class="directory-close"
        aria-label="Close directory"
        title="Close directory"
        data-testid="directory-close"
        @click="closeDirectory"
      >
        <span aria-hidden="true">×</span>
      </button>
    </header>

    <div class="directory-search">
      <label for="window-directory-search">Search windows</label>
      <input
        id="window-directory-search"
        ref="directorySearch"
        v-model="searchQuery"
        type="search"
        autocomplete="off"
        placeholder="Search by name or group"
        aria-controls="window-directory-results"
      >
      <span class="search-count" role="status" aria-live="polite">
        {{ resultCount }} {{ resultCount === 1 ? 'window' : 'windows' }}
      </span>
    </div>

    <nav id="window-directory-results" class="directory-groups" aria-label="All windows">
      <section
        v-for="group in groups"
        :key="group.key"
        class="directory-group"
        :aria-labelledby="`directory-group-${group.key}`"
      >
        <h3 :id="`directory-group-${group.key}`" class="directory-group-title">
          <span>{{ group.label }}</span>
          <span class="directory-group-count">{{ group.routes.length }}</span>
        </h3>

        <div class="directory-group-items">
          <button
            v-for="route in group.routes"
            :key="route.id"
            type="button"
            class="directory-route"
            :class="{ 'is-active': route.id === activeRouteId }"
            :data-testid="`nav-${route.id}`"
            :aria-current="route.id === activeRouteId ? 'page' : undefined"
            @click="selectRoute(route.id)"
          >
            <span class="route-mark" aria-hidden="true">{{ route.shortLabel }}</span>
            <span class="route-copy">
              <span class="route-label">{{ route.label }}</span>
              <span v-if="route.description" class="route-description">{{ route.description }}</span>
            </span>
            <span v-if="route.id === activeRouteId" class="route-current" aria-label="Current window">
              <span aria-hidden="true">●</span>
            </span>
          </button>
        </div>
      </section>

      <p v-if="groups.length === 0" class="directory-empty" role="status">
        No windows match “{{ searchQuery }}”.
      </p>
    </nav>

    <footer class="directory-footer">
      <span>Select a window to open it.</span>
      <button type="button" class="directory-done" @click="closeDirectory">Done</button>
    </footer>
  </aside>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { ROUTES } from '../../desktop/routeRegistry';

interface RouteEntry {
  id: string;
  label: string;
  shortLabel: string;
  group: string;
  description: string;
  keywords: string[];
}

interface RouteGroup {
  key: string;
  label: string;
  routes: RouteEntry[];
}

const props = defineProps<{
  open: boolean;
  activeRouteId: string;
}>();

const emit = defineEmits<{
  (event: 'close'): void;
  (event: 'open-route', id: string): void;
}>();

const searchQuery = ref('');
const directorySearch = ref<HTMLInputElement | null>(null);

const routeEntries = computed<RouteEntry[]>(() => normalizeRoutes(ROUTES));

const filteredRoutes = computed<RouteEntry[]>(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase();
  if (!query) return routeEntries.value;

  return routeEntries.value.filter((route) => {
    const searchableText = [
      route.id,
      route.label,
      route.group,
      route.description,
      ...route.keywords,
    ].join(' ').toLocaleLowerCase();
    return searchableText.includes(query);
  });
});

const groups = computed<RouteGroup[]>(() => {
  const grouped = new Map<string, RouteEntry[]>();

  for (const route of filteredRoutes.value) {
    const key = route.group.toLocaleLowerCase();
    const routes = grouped.get(key) ?? [];
    routes.push(route);
    grouped.set(key, routes);
  }

  return Array.from(grouped.entries()).map(([key, routes]) => ({
    key: key.replace(/[^a-z0-9]+/g, '-') || 'windows',
    label: routes[0]?.group ?? 'Windows',
    routes,
  }));
});

const resultCount = computed(() => filteredRoutes.value.length);

function textValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

function humanize(value: string): string {
  return value
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/[-_]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (character) => character.toLocaleUpperCase());
}

function groupValue(value: unknown): string {
  if (typeof value === 'string') return humanize(value);
  if (Array.isArray(value)) {
    const first = value.find((item) => typeof item === 'string');
    return first ? humanize(first) : 'Tools';
  }
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    return textValue(record.label) || textValue(record.name) || 'Tools';
  }
  return 'Tools';
}

function normalizeRoutes(source: unknown): RouteEntry[] {
  if (!Array.isArray(source)) return [];

  return source.flatMap((value) => {
    if (!value || typeof value !== 'object') return [];
    const record = value as Record<string, unknown>;
    const id = textValue(record.id);
    if (!id) return [];

    const label = textValue(record.label) || humanize(id);
    const group = groupValue(record.group ?? record.category ?? record.section) || 'Tools';
    const description = textValue(record.description) || textValue(record.summary) || textValue(record.hint);
    const rawKeywords = [record.keywords, record.tags]
      .flatMap((item) => Array.isArray(item) ? item : [])
      .map((item) => textValue(item))
      .filter(Boolean);

    return [{
      id,
      label,
      shortLabel: label.slice(0, 2).toLocaleUpperCase(),
      group,
      description,
      keywords: rawKeywords,
    }];
  });
}

function selectRoute(id: string) {
  emit('open-route', id);
  emit('close');
}

function closeDirectory() {
  emit('close');
}

async function focusSearch() {
  await nextTick();
  directorySearch.value?.focus();
}

watch(() => props.open, (isOpen) => {
  if (isOpen) void focusSearch();
});

onMounted(() => {
  if (props.open) void focusSearch();
});
</script>

<style scoped>
.window-directory {
  position: absolute;
  z-index: 50;
  top: 12px;
  right: 12px;
  display: flex;
  width: min(360px, calc(100% - 24px));
  max-height: min(680px, calc(100% - 24px));
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  background: var(--bg-elev);
  color: var(--text);
  box-shadow: var(--shadow-lg);
  color-scheme: light;
}

.directory-header,
.directory-footer {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-sidebar);
}

.directory-footer {
  border-top: 1px solid var(--border);
  border-bottom: 0;
  color: var(--text-muted);
  font-size: 11px;
}

.directory-kicker {
  margin: 0 0 2px;
  color: var(--text-muted);
  font: 650 9px/1.2 var(--font-mono);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.directory-title {
  margin: 0;
  color: var(--text);
  font-size: 17px;
  font-weight: 750;
  letter-spacing: -0.02em;
  line-height: 1.2;
}

.directory-close,
.directory-done {
  min-height: 30px;
  border: 1px solid transparent;
  border-radius: 5px;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font: 650 11px/1 var(--font);
}

.directory-close {
  width: 30px;
  padding: 0;
  font-size: 18px;
}

.directory-close:hover,
.directory-done:hover {
  border-color: var(--border);
  background: var(--bg-hover);
  color: var(--text);
}

.directory-search {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 5px 8px;
  padding: 12px 14px 10px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-elev);
}

.directory-search label {
  grid-column: 1 / -1;
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 650;
}

.directory-search input {
  min-width: 0;
  min-height: 34px;
  padding: 0 9px;
  border: 1px solid var(--border);
  border-radius: 5px;
  outline: 0;
  background: var(--bg-sidebar);
  color: var(--text);
  font: 12px var(--font);
}

.directory-search input::placeholder {
  color: var(--text-muted);
}

.directory-search input:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--accent-soft);
}

.search-count {
  align-self: center;
  color: var(--text-muted);
  font-size: 10px;
  white-space: nowrap;
}

.directory-groups {
  min-height: 0;
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 8px;
}

.directory-group + .directory-group {
  margin-top: 8px;
}

.directory-group-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin: 0;
  padding: 5px 7px;
  color: var(--text-muted);
  font: 700 10px/1.3 var(--font-mono);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.directory-group-count {
  display: inline-grid;
  min-width: 18px;
  height: 18px;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--text-muted);
  font: 600 9px/1 var(--font);
}

.directory-group-items {
  display: grid;
  gap: 2px;
}

.directory-route {
  display: flex;
  width: 100%;
  min-height: 43px;
  align-items: center;
  gap: 9px;
  padding: 6px 7px;
  border: 1px solid transparent;
  border-radius: 6px;
  background: transparent;
  color: var(--text);
  cursor: pointer;
  text-align: left;
}

.directory-route:hover,
.directory-route:focus-visible {
  border-color: var(--border);
  background: var(--bg-hover);
}

.directory-route.is-active {
  border-color: var(--border);
  background: var(--accent-soft);
}

.route-mark {
  display: inline-grid;
  width: 28px;
  height: 28px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-elev);
  color: var(--text-dim);
  font: 700 9px/1 var(--font-mono);
}

.directory-route.is-active .route-mark {
  border-color: var(--primary);
  color: var(--primary);
}

.route-copy {
  display: flex;
  min-width: 0;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 2px;
}

.route-label,
.route-description {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.route-label {
  color: var(--text);
  font-size: 12px;
  font-weight: 650;
}

.route-description {
  color: var(--text-muted);
  font-size: 10px;
}

.route-current {
  flex: 0 0 auto;
  color: var(--primary);
  font-size: 10px;
}

.directory-empty {
  margin: 18px 8px;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
  text-align: center;
}

.directory-done {
  padding: 0 9px;
}

@media (max-width: 600px) {
  .window-directory {
    position: relative;
    top: auto;
    right: auto;
    width: 100%;
    max-height: none;
    margin-bottom: 12px;
  }
}
</style>
