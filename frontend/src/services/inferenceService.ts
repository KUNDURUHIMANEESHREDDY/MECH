import { InferenceResponse } from '../types';
import { getApiKey } from './api';
import { demoInference } from './demoService';

const API_BASE = 'http://localhost:8000/api';

export async function runInference(prompt: string, maxNewTokens = 10): Promise<InferenceResponse> {
  try {
    const apiKey = await getApiKey();
    const res = await fetch(`${API_BASE}/infer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Client-Id': 'model-explorer-ui',
        ...(apiKey ? { 'X-API-Key': apiKey } : {}),
      },
      body: JSON.stringify({ prompt, max_new_tokens: maxNewTokens }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => null);
      throw new Error(err?.detail ?? `Inference failed: ${res.statusText}`);
    }
    return res.json() as Promise<InferenceResponse>;
  } catch {
    console.warn('[inferenceService] sidecar unreachable, using demo inference');
    return demoInference(prompt, maxNewTokens);
  }
}
