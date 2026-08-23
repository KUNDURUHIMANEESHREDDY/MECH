import { InferenceResponse } from '../types';
import { getApiKey } from './api';
import { demoInference } from './demoService';

const API_BASE = 'http://localhost:8000/api';

export interface InferenceParams {
  prompt: string;
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
}

export async function runInference(
  promptOrParams: string | InferenceParams,
  maxNewTokens = 10
): Promise<InferenceResponse> {
  const params = typeof promptOrParams === 'string'
    ? { prompt: promptOrParams, maxNewTokens }
    : promptOrParams;

  try {
    const apiKey = await getApiKey();
    const res = await fetch(`${API_BASE}/infer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Client-Id': 'model-explorer-ui',
        ...(apiKey ? { 'X-API-Key': apiKey } : {}),
      },
      body: JSON.stringify({
        prompt: params.prompt,
        max_new_tokens: params.maxNewTokens ?? maxNewTokens,
        temperature: params.temperature ?? 1.0,
        top_k: params.topK ?? 50,
        top_p: params.topP ?? 0.9,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => null);
      throw new Error(err?.detail ?? `Inference failed (${res.status}): ${res.statusText}`);
    }
    const data = await res.json() as InferenceResponse;
    return {
      ...data,
      provenance: 'LIVE_PYTORCH',
    } as InferenceResponse;
  } catch (err: any) {
    console.error('[inferenceService] Live backend unreachable:', err.message);
    throw new Error(
      `[MECH Live Firewall] Research operation failed: Backend sidecar unreachable (${err.message}). ` +
      'Live model execution requires an active backend server.'
    );
  }
}

