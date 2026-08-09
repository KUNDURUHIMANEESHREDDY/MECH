import { ModelInfo } from '../types';
import { getApiKey } from './api';
import { demoAvailableModels, demoModelInfo } from './demoService';

const API_BASE = 'http://localhost:8000/api';

async function authHeaders(): Promise<Record<string, string>> {
  const apiKey = await getApiKey();
  return { ...(apiKey ? { 'X-API-Key': apiKey } : {}), 'Content-Type': 'application/json' };
}

export async function fetchAvailableModels(): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE}/models`, { headers: await authHeaders() });
    if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
    const data = await res.json();
    return data.models as string[];
  } catch (e) {
    console.warn('[modelService] sidecar unreachable, using demo model catalog:', (e as Error).message);
    return demoAvailableModels();
  }
}

export async function loadModel(modelName: string): Promise<ModelInfo> {
  try {
    const res = await fetch(`${API_BASE}/models/load`, {
      method: 'POST',
      headers: await authHeaders(),
      body: JSON.stringify({ model_name: modelName }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => null);
      throw new Error(err?.detail ?? `Failed to load model: ${res.statusText}`);
    }
    return res.json() as Promise<ModelInfo>;
  } catch {
    console.warn('[modelService] sidecar unreachable, loading demo model:', modelName);
    return demoModelInfo(modelName);
  }
}

export async function fetchModelInfo(modelName: string): Promise<any> {
  const res = await fetch(`${API_BASE}/models/${modelName}`, { headers: await authHeaders() });
  if (!res.ok) throw new Error(`Failed to fetch model info: ${res.statusText}`);
  return res.json();
}
