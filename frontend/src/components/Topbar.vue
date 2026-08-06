<template>
  <header
    class="flex items-center justify-between h-[46px] px-6 border-b border-[var(--hairline)] bg-[var(--canvas)]"
  >
    <div class="flex items-center gap-2">
      <span class="text-sm font-medium tracking-tight text-[var(--ink)]">
        {{ crumb }}
      </span>
    </div>

    <div class="flex items-center gap-1">
      <button
        :title="collapsed ? 'Expand activity bar' : 'Collapse activity bar'"
        @click="$emit('toggle-activity')"
        class="w-9 h-9 rounded-full border-none bg-transparent text-[var(--ink-muted-48)] cursor-pointer flex items-center justify-center hover:text-[var(--ink-muted-80)]"
      >
        <Settings :size="18" :stroke-width="1.75" />
      </button>

      <button
        class="flex items-center gap-1 px-2 py-1 rounded text-xs font-medium text-[var(--ink-muted-80)] hover:bg-[var(--accent-soft)] transition-colors"
        @click="$emit('toggle-cmd-palette')"
      >
        <Command :size="13" /> P
      </button>

      <div
        data-testid="python-status"
        class="flex items-center gap-1 pl-2 ml-1 border-l border-[var(--border)]"
      >
        <span
          class="w-2 h-2 rounded-full"
          :class="pythonStatus === 'connected' ? 'bg-[var(--text)]' : pythonStatus === 'connecting' ? 'bg-[var(--ink-muted-48)]' : 'bg-[var(--hairline)]'"
        />
        <span class="text-xs text-[var(--ink-muted-80)]">
          {{ statusLabel }}
        </span>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { Command } from 'lucide-vue-next';

const props = defineProps<{
  crumb: string;
  pythonStatus: 'connected' | 'offline' | 'connecting';
  collapsed: boolean;
}>();

defineEmits<{
  'toggle-cmd-palette': [];
  'toggle-activity': [];
}>();

const statusLabel = computed(() =>
  props.pythonStatus === 'connected' ? 'Connected'
    : props.pythonStatus === 'connecting' ? '…'
      : 'Offline'
);
</script>
