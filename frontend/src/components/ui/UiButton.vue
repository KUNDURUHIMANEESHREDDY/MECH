<script setup lang="ts">
/**
 * UiButton - the shared button.
 *
 * `busy` is separate from `disabled` on purpose: a busy button is disabled
 * AND announces itself through aria-busy, so a slow action is never mistaken
 * for an inert one.
 */
withDefaults(
  defineProps<{
    variant?: 'default' | 'primary' | 'ghost' | 'danger';
    size?: 'sm' | 'md';
    type?: 'button' | 'submit' | 'reset';
    disabled?: boolean;
    busy?: boolean;
  }>(),
  { variant: 'default', size: 'md', type: 'button', disabled: false, busy: false },
);
</script>

<template>
  <button
    class="ui-button"
    :class="[`ui-button--${variant}`, `ui-button--${size}`, { 'is-busy': busy }]"
    :type="type"
    :disabled="disabled || busy"
    :aria-busy="busy ? 'true' : undefined"
  >
    <span v-if="$slots.icon" class="ui-button__icon" aria-hidden="true">
      <slot name="icon" />
    </span>
    <span class="ui-button__label"><slot /></span>
  </button>
</template>
