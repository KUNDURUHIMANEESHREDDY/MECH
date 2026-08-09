import { create } from 'zustand';
import { InferenceResponse, ModelInfo, TokenInfo, LayerData, NeuronData } from '../types';
import { fetchAvailableModels, loadModel as loadModelApi } from '../../services/modelService';
import { runInference } from '../../services/inferenceService';
import { attentionMapsToLayers } from '../../services/attentionService';
import { activationsToNeurons } from '../../services/activationService';

export interface ProcessedResult {
  tokens: TokenInfo[];
  generatedText: string;
  layers: LayerData[];
  gpuUtil: number;
  memoryUtil: number;
}

export interface ModelState {
  modelInfo: ModelInfo | null;
  availableModels: string[];
  loading: boolean;
  loaded: boolean;
  running: boolean;
  result: ProcessedResult | null;
  error: string | null;
}

export interface ModelStore extends ModelState {
  listModels: () => Promise<void>;
  load: (modelName: string) => Promise<void>;
  infer: (prompt: string, maxNewTokens?: number) => Promise<ProcessedResult>;
  dispose: () => void;
  clearError: () => void;
}

const initialState: ModelState = {
  modelInfo: null,
  availableModels: [],
  loading: false,
  loaded: false,
  running: false,
  result: null,
  error: null,
};

/**
 * Shared model state for the plugin canvas. Every panel plugin (attention,
 * logit lens, neurons, circuits, SAE, datasets) subscribes to this single
 * store, so an inference run from any panel populates all visual panels.
 */
export const useModelStore = create<ModelStore>((set, get) => ({
  ...initialState,

  listModels: async () => {
    try {
      const models = await fetchAvailableModels();
      set({ availableModels: models });
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  load: async (modelName) => {
    set({ loading: true, error: null });
    try {
      const info = await loadModelApi(modelName);
      set({ modelInfo: info, loading: false, loaded: true });
    } catch (e: any) {
      set({ loading: false, error: e.message });
    }
  },

  infer: async (prompt, maxNewTokens = 10) => {
    set({ running: true, error: null });
    try {
      const raw: InferenceResponse = await runInference(prompt, maxNewTokens);
      const { modelInfo } = get();
      const numLayers = modelInfo?.num_layers ?? 12;
      const numHeads = modelInfo?.num_heads ?? 12;

      const layers = attentionMapsToLayers(raw.attention_maps, numLayers, numHeads, raw.tokens.map(t => t.text));
      const neuronGrid = activationsToNeurons(raw.neuron_activations, numLayers);

      for (let li = 0; li < numLayers; li++) {
        for (let hi = 0; hi < numHeads; hi++) {
          if (layers[li]?.heads[hi]) {
            layers[li].heads[hi].neurons = neuronGrid[li] ?? [];
          }
        }
      }

      const processed: ProcessedResult = {
        tokens: raw.tokens,
        generatedText: raw.generated_text,
        layers,
        gpuUtil: raw.gpu_util,
        memoryUtil: raw.memory_util,
      };

      set({ running: false, result: processed, error: null });
      return processed;
    } catch (e: any) {
      set({ running: false, error: e.message });
      throw e;
    }
  },

  dispose: () => {},

  clearError: () => set({ error: null }),
}));