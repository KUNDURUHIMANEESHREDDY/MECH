<script setup lang="ts">
/**
 * UiPanel - the shared container used across the MECH views.
 *
 * Structure only. Every colour, radius and border comes from the tokens in
 * src/styles.css, so a panel looks the same wherever it is used.
 */
defineProps<{
  /** Heading shown in the panel header. Omit to hide the header entirely. */
  title?: string;
  /** Secondary line under the title. */
  subtitle?: string;
}>();
</script>

<template>
  <section class="ui-panel">
    <header v-if="title || subtitle || $slots.header || $slots.actions" class="ui-panel__header">
      <slot name="header">
        <div class="ui-panel__heading">
          <h2 v-if="title" class="ui-panel__title">{{ title }}</h2>
          <p v-if="subtitle" class="ui-panel__subtitle">{{ subtitle }}</p>
        </div>
      </slot>
      <div v-if="$slots.actions" class="ui-panel__actions">
        <slot name="actions" />
      </div>
    </header>
    <div class="ui-panel__body">
      <slot />
    </div>
    <footer v-if="$slots.footer" class="ui-panel__footer">
      <slot name="footer" />
    </footer>
  </section>
</template>
