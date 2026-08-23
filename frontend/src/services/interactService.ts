import { getApiKey } from './api';
import { demoInteraction } from './demoService';

const API_BASE = 'http://localhost:8000/api';

export interface InteractRequest {
  prompt: string;
  backend?: 'gpt2' | 'ollama' | 'openai';
  model?: string;
  max_new_tokens?: number;
  temperature?: number;
  system?: string;
}

export interface InteractResponse {
  status: 'ok' | 'demo' | 'error';
  backend: string;
  model: string;
  prompt: string;
  response: string;
  n_generated?: number;
  latency_ms?: number;
  error?: string;
}

export async function runInteraction(req: InteractRequest): Promise<InteractResponse> {
  try {
    const apiKey = await getApiKey();
    const res = await fetch(`${API_BASE}/interact`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Client-Id': 'model-interaction-ui',
        ...(apiKey ? { 'X-API-Key': apiKey } : {}),
      },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => null);
      throw new Error(err?.detail ?? `Interaction failed (${res.status}): ${res.statusText}`);
    }
    const data = await res.json() as InteractResponse;
    return {
      ...data,
      status: 'ok',
    };
  } catch (err: any) {
    console.error('[interactService] Live backend unreachable:', err.message);
    throw new Error(
      `[MECH Live Firewall] Model interaction failed: Backend sidecar unreachable (${err.message}).`
    );
  }
}