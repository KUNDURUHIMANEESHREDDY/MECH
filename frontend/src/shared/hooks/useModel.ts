import { useState, useCallback, useRef } from 'react';
import { InferenceResponse, ModelInfo, TokenInfo, LayerData, NeuronData } from '../types';
import { fetchAvailableModels, loadModel as loadModelApi } from '../../services/modelService';
import { runInference } from '../../services/inferenceService';
import { attentionMapsToLayers } from '../../services/attentionService';
import { activationsToNeurons } from '../../services/activationService';

export interface ModelState {
  modelInfo: ModelInfo | null;
  availableModels: string[];
  loading: boolean;
  loaded: boolean;
  running: boolean;
  result: ProcessedResult | null;
  error: string | null;
}

export interface ProcessedResult {
  tokens: TokenInfo[];
  generatedText: string;
  layers: LayerData[];
  gpuUtil: number;
  memoryUtil: number;
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

export function useModel() {
  const [state, setState] = useState<ModelState>(initialState);
  const disposedRef = useRef(false);

  const listModels = useCallback(async () => {
    try {
      const models = await fetchAvailableModels();
      setState(s => ({ ...s, availableModels: models }));
    } catch (e: any) {
      setState(s => ({ ...s, error: e.message }));
    }
  }, []);

  const load = useCallback(async (modelName: string) => {
    if (disposedRef.current) return;
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const info = await loadModelApi(modelName);
      if (!disposedRef.current) {
        setState(s => ({ ...s, modelInfo: info, loading: false, loaded: true }));
      }
    } catch (e: any) {
      if (!disposedRef.current) {
        setState(s => ({ ...s, loading: false, error: e.message }));
      }
    }
  }, []);

  const infer = useCallback(async (prompt: string, maxNewTokens = 10) => {
    if (disposedRef.current) return;
    setState(s => ({ ...s, running: true, error: null }));
    try {
      const raw: InferenceResponse = await runInference(prompt, maxNewTokens);
      const numLayers = state.modelInfo?.num_layers ?? 12;
      const numHeads = state.modelInfo?.num_heads ?? 12;

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

      if (!disposedRef.current) {
        setState(s => ({ ...s, running: false, result: processed, error: null }));
      }
    } catch (e: any) {
      if (!disposedRef.current) {
        setState(s => ({ ...s, running: false, error: e.message }));
      }
    }
  }, [state.modelInfo]);

  const clearError = useCallback(() => {
    setState(s => ({ ...s, error: null }));
  }, []);

  const dispose = useCallback(() => { disposedRef.current = true; }, []);

  return { state, listModels, load, infer, dispose, clearError };
}
