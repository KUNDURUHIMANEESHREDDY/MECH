<script setup lang="ts">
/**
 * UiEmptyState - the shared "there is nothing here" state.
 *
 * It never says the data is missing because a request failed; a failed
 * request is an error state and must be shown as one.
 */
defineProps<{
  title: string;
  message?: string;
}>();
</script>

<template>
  <div class="ui-empty-state">
    <div>
      <span class="ui-empty-state__title">{{ title }}</span>
      <p v-if="message" class="ui-empty-state__message">{{ message }}</p>
      <div v-if="$slots.action" class="ui-empty-state__action">
        <slot name="action" />
      </div>
    </div>
  </div>
</template>
