import { useEffect, useRef, useState } from 'react';
import { LayerTensors } from '../types';
import { fetchLayerTensors } from '../services/inferenceService';

export interface LayerTensorsState {
  tensors: LayerTensors | null;
  loading: boolean;
  error: string | null;
}

const cache = new Map<string, LayerTensors>();

/** Full per-token resid_post / mlp_post for (prompt, layer), cached. */
export function useLayerTensors(prompt: string | null, layer: number): LayerTensorsState {
  const [state, setState] = useState<LayerTensorsState>({ tensors: null, loading: false, error: null });
  const seqRef = useRef(0);

  useEffect(() => {
    if (!prompt) {
      setState({ tensors: null, loading: false, error: null });
      return;
    }
    const key = `${prompt}::${layer}`;
    const hit = cache.get(key);
    if (hit) {
      setState({ tensors: hit, loading: false, error: null });
      return;
    }
    const seq = ++seqRef.current;
    setState({ tensors: null, loading: true, error: null });
    fetchLayerTensors(layer, prompt)
      .then(t => {
        cache.set(key, t);
        if (seqRef.current === seq) setState({ tensors: t, loading: false, error: null });
      })
      .catch((e: unknown) => {
        if (seqRef.current === seq) {
          setState({ tensors: null, loading: false, error: e instanceof Error ? e.message : String(e) });
        }
      });
  }, [prompt, layer]);

  return state;
}
