<template>
  <header class="desktop-menu-bar" aria-label="Desktop menu bar">
    <div class="brand" aria-label="MECH desktop">
      <span class="brand-mark" aria-hidden="true">M</span>
      <span class="brand-name">MECH</span>
    </div>

    <nav class="quick-nav" aria-label="Quick tools">
      <button
        v-for="route in quickRoutes"
        :key="route.id"
        type="button"
        class="quick-action"
        :class="{ 'is-highlighted': route.id === 'society' }"
        :aria-label="route.label"
        :data-testid="`menu-${route.id}`"
        @click="openRoute(route.id)"
      >
        <span v-if="route.id === 'society'" class="quick-action-mark" aria-hidden="true">S</span>
        <span>{{ route.label }}</span>
      </button>
    </nav>

    <div class="menu-actions">
      <span class="model-summary" role="status" aria-live="polite">
        <span class="status-dot" :class="`status-${modelStatus}`" aria-hidden="true" />
        <span class="model-label">Model</span>
        <span class="model-name">{{ modelName || 'No model selected' }}</span>
        <span class="model-status-value">{{ statusLabel }}</span>
      </span>

      <button
        type="button"
        class="menu-button"
        aria-label="Tile windows"
        title="Tile open windows"
        @click="emit('tile')"
      >
        Tile windows
      </button>

      <button
        type="button"
        class="menu-button"
        aria-label="Reset"
        title="Reset window layout"
        @click="emit('reset')"
      >
        Reset
      </button>

      <button
        type="button"
        class="directory-button"
        aria-label="All Windows Directory"
        aria-haspopup="dialog"
        aria-controls="window-directory"
        data-testid="directory-toggle"
        @click="emit('toggle-directory')"
      >
        <span>All Windows</span>
        <span class="directory-subtitle" aria-hidden="true">Directory</span>
      </button>
    </div>
  </header>
</template>

<script setup lang="ts">
import { computed } from 'vue';

type ModelStatus = 'connecting' | 'connected' | 'offline';

const emit = defineEmits<{
  (event: 'open-route', id: string): void;
  (event: 'tile'): void;
  (event: 'reset'): void;
  (event: 'toggle-directory'): void;
}>();

const quickRoutes = [
  { id: 'society', label: 'Society' },
  { id: 'explorer', label: 'Model Explorer' },
  { id: 'workspace', label: 'Workspace' },
  { id: 'models', label: 'Models' },
  { id: 'settings', label: 'Settings' },
  { id: 'build', label: 'Build' },
] as const;

const props = defineProps<{
  modelStatus: ModelStatus;
  modelName: string;
}>();

const statusLabel = computed(() => {
  if (props.modelStatus === 'connected') return 'Connected';
  if (props.modelStatus === 'connecting') return 'Connecting';
  return 'Offline';
});

function openRoute(id: string) {
  emit('open-route', id);
}
</script>

<style scoped>
.desktop-menu-bar {
  display: flex;
  min-width: 0;
  min-height: var(--topbar-h, 42px);
  align-items: center;
  gap: 14px;
  padding: 6px 14px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-elev);
  color: var(--text);
  color-scheme: light;
}

.brand,
.model-summary,
.quick-action,
.menu-button,
.directory-button {
  font: inherit;
}

.brand {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 8px;
  color: var(--text);
  font-size: 12px;
  font-weight: 750;
  letter-spacing: 0.08em;
  white-space: nowrap;
}

.brand-mark {
  display: grid;
  width: 24px;
  height: 24px;
  place-items: center;
  border: 1px solid var(--text);
  border-radius: 6px;
  background: var(--bg-elev);
  color: var(--text);
  font: 750 12px/1 var(--font-mono);
  letter-spacing: 0;
}

.quick-nav {
  display: flex;
  min-width: 0;
  flex: 1 1 auto;
  align-items: center;
  gap: 2px;
  overflow-x: auto;
  scrollbar-width: none;
}

.quick-nav::-webkit-scrollbar {
  display: none;
}

.quick-action,
.menu-button,
.directory-button {
  min-height: 30px;
  border: 1px solid transparent;
  border-radius: 6px;
  background: transparent;
  color: var(--text-dim);
  cursor: pointer;
  white-space: nowrap;
}

.quick-action {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 0 9px;
  font-size: 12px;
  font-weight: 600;
}

.quick-action:hover,
.menu-button:hover,
.directory-button:hover {
  border-color: var(--border);
  background: var(--bg-hover);
  color: var(--text);
}

.quick-action.is-highlighted {
  border-color: var(--border);
  background: var(--accent-soft);
  color: var(--text);
}

.quick-action-mark {
  display: inline-grid;
  width: 17px;
  height: 17px;
  place-items: center;
  border: 1px solid var(--border-light);
  border-radius: 4px;
  color: var(--text);
  font: 700 9px/1 var(--font-mono);
}

.menu-actions {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: flex-end;
  gap: 5px;
}

.model-summary {
  display: inline-flex;
  max-width: 170px;
  align-items: center;
  gap: 5px;
  overflow: hidden;
  color: var(--text-muted);
  font-size: 11px;
  white-space: nowrap;
}

.model-label {
  color: var(--text-muted);
}

.model-name {
  overflow: hidden;
  color: var(--text-dim);
  font-weight: 650;
  text-overflow: ellipsis;
}

.status-dot {
  width: 7px;
  height: 7px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: var(--text-muted);
}

.status-connected {
  background: var(--success);
}

.status-connecting {
  background: var(--warning);
}

.status-offline {
  background: var(--text-muted);
}

.model-status-value {
  color: var(--text-muted);
  font-weight: 650;
}

.menu-button {
  padding: 0 9px;
  font-size: 11px;
  font-weight: 600;
}

.directory-button {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 0 9px;
  border-color: var(--border);
  background: var(--bg-sidebar);
  color: var(--text);
  font-size: 11px;
  font-weight: 700;
}

.directory-subtitle {
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 500;
}

@media (max-width: 860px) {
  .desktop-menu-bar {
    gap: 8px;
    padding-inline: 10px;
  }

  .model-summary {
    display: none;
  }

  .quick-action {
    padding-inline: 7px;
  }
}

@media (max-width: 600px) {
  .desktop-menu-bar {
    flex-wrap: wrap;
  }

  .quick-nav {
    order: 3;
    width: 100%;
    flex-basis: 100%;
  }

  .menu-actions {
    margin-left: auto;
  }

  .directory-subtitle {
    display: none;
  }
}
</style>
