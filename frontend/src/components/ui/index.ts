/**
 * Shared UI primitives.
 *
 * These are the components that were hand-rolled in every view. They are
 * deliberately structure-only: all visual values come from the tokens in
 * src/styles.css, so there is exactly one place to change how the app looks.
 */
export { default as UiPanel } from './UiPanel.vue';
export { default as UiButton } from './UiButton.vue';
export { default as UiStatusPill } from './UiStatusPill.vue';
export { default as UiEmptyState } from './UiEmptyState.vue';
export { default as UiErrorState } from './UiErrorState.vue';
