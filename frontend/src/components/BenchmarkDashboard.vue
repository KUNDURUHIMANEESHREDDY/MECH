<template>
  <div class="p-4">
    <h2 class="text-lg font-bold text-[var(--ink)] mb-3">Benchmark Dashboard</h2>
    <div class="grid grid-cols-3 gap-3">
      <div class="p-4 rounded border border-[var(--border)] bg-[var(--canvas)] text-center">
        <div class="text-2xl font-bold text-[var(--primary)]">0</div>
        <div class="text-xs text-[var(--ink-muted-48)] mt-1">Completed</div>
      </div>
      <div class="p-4 rounded border border-[var(--border)] bg-[var(--canvas)] text-center">
        <div class="text-2xl font-bold text-[var(--ink)]">0</div>
        <div class="text-xs text-[var(--ink-muted-48)] mt-1">Running</div>
      </div>
      <div class="p-4 rounded border border-[var(--border)] bg-[var(--canvas)] text-center">
        <div class="text-2xl font-bold text-[var(--ink)]">0</div>
        <div class="text-xs text-[var(--ink-muted-48)] mt-1">Failed</div>
      </div>
    </div>
  </div>
</template>
<script setup lang="ts"></script>
