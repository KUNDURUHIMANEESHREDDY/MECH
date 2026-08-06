<template>
  <div class="p-4">
    <h2 class="text-lg font-bold text-[var(--ink)] mb-3">Models</h2>
    <div class="grid grid-cols-2 gap-2">
      <div v-for="m in models" :key="m" class="p-3 rounded border border-[var(--border)] bg-[var(--canvas)] text-sm font-medium text-[var(--ink)]">
        {{ m }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { api } from '../services/api';

const models = ref<string[]>([]);
onMounted(async () => { try { const r = await api.listModels(); models.value = r.models || []; } catch { /* */ } });
</script>
