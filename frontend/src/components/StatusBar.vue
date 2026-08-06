<template>
  <footer
    class="flex items-center justify-between h-[24px] px-4 border-t border-[var(--hairline)] bg-[var(--bg-sidebar)] text-[11px] text-[var(--ink-muted-48)]"
  >
    <div class="flex items-center gap-3">
      <span>Model: {{ modelName }}</span>
      <span v-if="tokenCount > 0">Tokens: {{ tokenCount }}</span>
    </div>
    <div class="flex items-center gap-3">
      <span v-if="gpuUtil > 0">GPU: {{ (gpuUtil * 100).toFixed(0) }}%</span>
      <span v-if="memoryUtil > 0">Mem: {{ (memoryUtil * 100).toFixed(0) }}%</span>
    </div>
  </footer>
</template>

<script setup lang="ts">
defineProps<{
  modelName?: string;
  gpuUtil?: number;
  memoryUtil?: number;
  tokenCount?: number;
}>();
</script>
