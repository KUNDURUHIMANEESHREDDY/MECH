<script setup lang="ts">
/**
 * UiErrorState - the shared failure state.
 *
 * The message is the caller's, verbatim: it must be the reason the backend
 * gave (or "reason unknown"), never a softened or invented one.
 */
defineProps<{
  title?: string;
  message: string;
  retryLabel?: string;
}>();

defineEmits<{ (event: 'retry'): void }>();
</script>

<template>
  <div class="ui-error-state" role="alert">
    <div>
      <span class="ui-error-state__title">{{ title || 'Request failed' }}</span>
      <p class="ui-error-state__message">{{ message }}</p>
      <div v-if="$slots.action" class="ui-error-state__action">
        <slot name="action" />
      </div>
    </div>
  </div>
</template>
