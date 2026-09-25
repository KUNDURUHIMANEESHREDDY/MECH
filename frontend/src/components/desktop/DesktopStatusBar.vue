<template>
  <footer class="desktop-status-bar" aria-label="Desktop status">
    <div
      class="python-status"
      data-testid="python-status"
      role="status"
      aria-live="polite"
    >
      <span class="status-dot" :class="`status-${modelStatus}`" aria-hidden="true" />
      <span class="status-prefix">Python</span>
      <span class="status-value">{{ statusLabel }}</span>
      <span class="status-model">{{ modelName || 'No model selected' }}</span>
    </div>

    <div class="status-details">
      <span class="status-detail status-window" :title="activeWindowTitle || 'No active window'">
        {{ activeWindowTitle || 'No active window' }}
      </span>
      <span
        class="status-detail"
        :aria-label="`${openWindowCount} ${openWindowCount === 1 ? 'window' : 'windows'} open`"
      >
        {{ openWindowCount }} {{ openWindowCount === 1 ? 'window' : 'windows' }} open
      </span>
    </div>
  </footer>
</template>

<script setup lang="ts">
import { computed } from 'vue';

type ModelStatus = 'connecting' | 'connected' | 'offline';

const props = defineProps<{
  modelStatus: ModelStatus;
  modelName: string;
  activeWindowTitle: string;
  openWindowCount: number;
}>();

const statusLabel = computed(() => {
  if (props.modelStatus === 'connected') return 'Connected';
  if (props.modelStatus === 'connecting') return 'Connecting';
  return 'Offline';
});
</script>

<style scoped>
.desktop-status-bar {
  display: flex;
  min-width: 0;
  min-height: var(--statusbar-h, 28px);
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 0 14px;
  border-top: 1px solid var(--border);
  background: var(--bg-elev);
  color: var(--text-dim);
  font-size: 11px;
  color-scheme: light;
}

.python-status,
.status-details,
.status-detail {
  min-width: 0;
}

.python-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  overflow: hidden;
  white-space: nowrap;
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

.status-prefix {
  color: var(--text);
  font-weight: 700;
}

.status-value {
  color: var(--text-dim);
  font-weight: 650;
}

.status-model {
  max-width: 150px;
  overflow: hidden;
  color: var(--text-muted);
  text-overflow: ellipsis;
}

.status-details {
  display: inline-flex;
  align-items: center;
  justify-content: flex-end;
  gap: 14px;
  overflow: hidden;
  white-space: nowrap;
}

.status-detail {
  overflow: hidden;
  color: var(--text-muted);
  text-overflow: ellipsis;
}

.status-window {
  max-width: 240px;
  color: var(--text-dim);
}

@media (max-width: 620px) {
  .desktop-status-bar {
    gap: 8px;
    padding-inline: 9px;
  }

  .status-model,
  .status-window {
    display: none;
  }
}
</style>
