import { InferenceResponse, LayerTensors, LogitLensAll } from '../types';

const API_BASE = 'http://localhost:8000/api';

export async function runInference(prompt: string, maxNewTokens = 10): Promise<InferenceResponse> {
  const res = await fetch(`${API_BASE}/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Client-Id': 'model-explorer-ui' },
    body: JSON.stringify({ prompt, max_new_tokens: maxNewTokens }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.detail ?? `Inference failed: ${res.statusText}`);
  }
  return res.json() as Promise<InferenceResponse>;
}

export async function fetchLayerTensors(layer: number, prompt: string): Promise<LayerTensors> {
  const res = await fetch(`${API_BASE}/gpt2/layer_activations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ layer, prompt }),
  });
  if (!res.ok) throw new Error(`Layer tensors failed: ${res.statusText}`);
  const data = (await res.json()) as LayerTensors & { error?: string };
  if (data.status !== 'ok') throw new Error(data.error ?? 'Layer tensors unavailable');
  return data;
}

export async function fetchLogitLensAll(prompt: string): Promise<LogitLensAll> {
  const res = await fetch(`${API_BASE}/gpt2/logit_lens_all`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error(`Logit lens failed: ${res.statusText}`);
  const data = (await res.json()) as LogitLensAll & { error?: string };
  if (data.status !== 'ok') throw new Error(data.error ?? 'Logit lens unavailable');
  return data;
}
