<script setup lang="ts">
/**
 * UiStatusPill - the shared status chip.
 *
 * Tones map onto the semantic tokens (success / warning / danger / primary).
 * The label is supplied by the caller: a pill states what the system actually
 * reported, it never infers a status of its own.
 */
withDefaults(
  defineProps<{
    tone?: 'neutral' | 'success' | 'warning' | 'danger' | 'info';
    label?: string;
  }>(),
  { tone: 'neutral' },
);
</script>

<template>
  <span class="ui-status-pill" :class="`ui-status-pill--${tone}`" role="status">
    <span class="ui-status-pill__dot" aria-hidden="true" />
    <slot>{{ label }}</slot>
  </span>
</template>
