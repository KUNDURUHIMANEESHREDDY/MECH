<template>
  <div class="p-4">
    <h2 class="text-lg font-bold text-[var(--ink)] mb-3">Experiments</h2>
    <div class="space-y-2">
      <div v-for="exp in experiments" :key="exp.id" class="p-3 rounded border border-[var(--border)] bg-[var(--canvas)]">
        <div class="text-sm font-medium text-[var(--ink)]">{{ exp.name || exp.id }}</div>
      </div>
      <div v-if="!experiments.length" class="text-sm text-[var(--ink-muted-48)]">No experiments yet.</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { api } from '../services/api';

const experiments = ref<any[]>([]);
onMounted(async () => { try { experiments.value = await api.listExperiments(); } catch { /* */ } });
</script>
