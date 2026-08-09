import { useModelStore } from '../stores/model';
import type { ModelState, ProcessedResult } from '../stores/model';

export type { ModelState, ProcessedResult };

/**
 * Shared model hook for the plugin canvas. Delegates to the single zustand
 * model store so every panel plugin (attention, logit lens, neurons, circuits,
 * SAE, datasets) reads and writes the SAME inference state — a probe run from
 * any panel populates all visual panels.
 */
export function useModel() {
  const store = useModelStore();
  return {
    state: {
      modelInfo: store.modelInfo,
      availableModels: store.availableModels,
      loading: store.loading,
      loaded: store.loaded,
      running: store.running,
      result: store.result,
      error: store.error,
    } satisfies ModelState,
    listModels: store.listModels,
    load: store.load,
    infer: store.infer,
    dispose: store.dispose,
    clearError: store.clearError,
  };
}
