<template>
  <div
    v-if="tabs.length > 0"
    class="window-tabs"
    role="tablist"
    aria-label="Open tools"
    data-testid="window-tabs"
  >
    <div
      v-for="tab in tabs"
      :key="tab.routeId"
      role="tab"
      tabindex="0"
      class="window-tab"
      :class="{ 'is-active': tab.active, 'is-minimized': tab.minimized }"
      :aria-selected="tab.active"
      :data-testid="`tab-${tab.routeId}`"
      :title="tab.minimized ? `${tab.title} (minimized)` : tab.title"
      @click="emit('focus', tab.routeId)"
      @keydown.enter="emit('focus', tab.routeId)"
      @keydown.space.prevent="emit('focus', tab.routeId)"
    >
      <span class="window-tab__label">{{ tab.title }}</span>
      <button
        type="button"
        class="window-tab__close"
        :aria-label="`Close ${tab.title}`"
        :data-testid="`tab-close-${tab.routeId}`"
        @click.stop="emit('close', tab.routeId)"
      >×</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useDesktopStore } from '../../stores/desktop';
import { getRouteById } from '../../desktop/routeRegistry';

const emit = defineEmits<{
  (event: 'focus', routeId: string): void;
  (event: 'close', routeId: string): void;
}>();

const store = useDesktopStore();

const tabs = computed(() =>
  store.openWindows
    .map(windowState => ({
      routeId: windowState.routeId,
      title: getRouteById(windowState.routeId)?.label ?? windowState.routeId,
      active: windowState.routeId === store.activeWindowId,
      minimized: windowState.minimized,
    }))
    .sort((left, right) => left.title.localeCompare(right.title)),
);
</script>

<style scoped>
.window-tabs {
  display: flex;
  min-width: 0;
  align-items: stretch;
  gap: 2px;
  overflow-x: auto;
  padding: 4px 10px 0;
  border-bottom: 1px solid var(--border);
  background: var(--bg-elev);
  color-scheme: light;
}

.window-tab {
  display: inline-flex;
  max-width: 220px;
  min-height: 30px;
  align-items: center;
  gap: 8px;
  padding: 4px 6px 4px 12px;
  border: 1px solid var(--border);
  border-bottom: none;
  border-radius: 6px 6px 0 0;
  background: var(--surface-1, transparent);
  color: var(--text-dim, var(--text));
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
}

.window-tab:hover {
  color: var(--text);
}

.window-tab.is-active {
  background: var(--bg, #fff);
  color: var(--text);
  font-weight: 600;
}

.window-tab.is-minimized .window-tab__label {
  opacity: 0.6;
  font-style: italic;
}

.window-tab__label {
  overflow: hidden;
  text-overflow: ellipsis;
}

.window-tab__close {
  display: inline-flex;
  width: 18px;
  height: 18px;
  flex: none;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  font-size: 13px;
  line-height: 1;
}

.window-tab__close:hover {
  background: var(--danger-soft, #f3dfe2);
  color: var(--danger, #a33);
}
</style>
