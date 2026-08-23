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
    if (!res.ok) throw new Error(`Failed to fetch models (${res.status}): ${res.statusText}`);
    const data = await res.json();
    // The backend serves a raw array of model records ({model_id, model_name, ...}).
    // Normalize both that shape and the legacy {models: [...]} shape to id strings.
    if (Array.isArray(data)) {
      return data.map((m: any) => m.model_id ?? m).filter(Boolean);
    }
    return (data?.models ?? []) as string[];
  } catch (e: any) {
    console.error('[modelService] Live backend unreachable:', e.message);
    throw new Error(
      `[MECH Live Firewall] Cannot fetch available models: Backend sidecar unreachable (${e.message}).`
    );
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
      throw new Error(err?.detail ?? `Failed to load model (${res.status}): ${res.statusText}`);
    }
    const data = await res.json() as ModelInfo;
    return {
      ...data,
      status: 'loaded',
      provenance: 'LIVE_PYTORCH',
    } as ModelInfo;
  } catch (e: any) {
    console.error('[modelService] Failed to load live model:', modelName, e.message);
    throw new Error(
      `[MECH Live Firewall] Cannot load model '${modelName}': Backend sidecar unreachable (${e.message}).`
    );
  }
}


export async function fetchModelInfo(modelName: string): Promise<any> {
  const res = await fetch(`${API_BASE}/models/${modelName}`, { headers: await authHeaders() });
  if (!res.ok) throw new Error(`Failed to fetch model info: ${res.statusText}`);
  return res.json();
}
