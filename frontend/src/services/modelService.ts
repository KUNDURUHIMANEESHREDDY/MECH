import { ModelInfo } from '../types';

const API_BASE = 'http://localhost:8000/api/v1';

export async function fetchAvailableModels(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
  const data = await res.json();
  return data.models as string[];
}

export async function loadModel(modelName: string): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/models/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: modelName }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.detail ?? `Failed to load model: ${res.statusText}`);
  }
  return res.json() as Promise<ModelInfo>;
}

export async function fetchModelInfo(modelName: string): Promise<any> {
  const res = await fetch(`${API_BASE}/models/${modelName}`);
  if (!res.ok) throw new Error(`Failed to fetch model info: ${res.statusText}`);
  return res.json();
}
