<template>
  <div class="p-4">
    <h2 class="text-lg font-bold text-[var(--ink)] mb-3">Sessions</h2>
    <div v-for="s in sessions" :key="s.id" class="p-3 rounded border border-[var(--border)] bg-[var(--canvas)] mb-2">
      <div class="text-sm font-medium text-[var(--ink)]">{{ s.name || s.id }}</div>
    </div>
    <div v-if="!sessions.length" class="text-sm text-[var(--ink-muted-48)]">No sessions yet.</div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { api } from '../services/api';

const sessions = ref<any[]>([]);
onMounted(async () => { try { sessions.value = await api.listSessions(); } catch { /* */ } });
</script>
