import { ModelInfo } from '../types';
import { apiUrl, authHeaders } from './api';

const API_BASE = apiUrl('/api');

export async function fetchAvailableModels(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/models`, { headers: { Accept: 'application/json', ...authHeaders() } });
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.status} ${res.statusText}`);
  const data = await res.json();
  return data.models as string[];
}

export async function loadModel(modelName: string): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/models/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ model_name: modelName }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.detail ?? `Failed to load model: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<ModelInfo>;
}

export async function fetchModelInfo(modelName: string): Promise<any> {
  const res = await fetch(`${API_BASE}/models/${modelName}`, { headers: { Accept: 'application/json', ...authHeaders() } });
  if (!res.ok) throw new Error(`Failed to fetch model info: ${res.status} ${res.statusText}`);
  return res.json();
}
