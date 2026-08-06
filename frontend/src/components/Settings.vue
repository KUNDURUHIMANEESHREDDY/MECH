<template>
  <div class="p-4">
    <h2 class="text-lg font-bold text-[var(--ink)]">Settings</h2>
    <p class="text-sm text-[var(--ink-muted-48)] mt-2">Platform configuration.</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{ settings: any; onChange: () => void; api: any }>();
</script>
