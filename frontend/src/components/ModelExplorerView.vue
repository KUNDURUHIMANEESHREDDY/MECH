<template>
  <div class="flex flex-col h-full p-6 gap-4 overflow-y-auto">
    <div class="text-lg font-semibold text-[var(--text)]">Model Explorer</div>
    <div class="text-sm text-[var(--text-muted)]">Currently loaded: <span class="font-medium text-[var(--text)]">{{ modelName }}</span></div>
    <div class="text-sm text-[var(--text-muted)]">Status: <span class="font-medium text-[var(--text)]">Connected</span></div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  modelName?: string;
}>();
</script>
