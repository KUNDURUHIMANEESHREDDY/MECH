/* Society service — client for the Research Society v2 backend endpoints.
 *
 *   POST /api/society/run              {goal, model_name?} -> {runId, ...}
 *   GET  /api/society/stream?runId=... -> text/event-stream (live trace)
 *   GET  /api/society/runs/{runId}     -> polling fallback
 *
 * Mirrors the standalone-service pattern of inferenceService.ts.
 */

import { apiUrl, authHeaders, apiToken } from './api';

const SOCIETY_BASE = apiUrl('/api/society');

export interface SocietyRunStarted {
  runId: string;
  status: string;
  stream: string;
}

export interface SocietyEvent {
  event_type: string;
  payload: Record<string, unknown>;
  timestamp?: string;
}

export interface SocietyRunStatus {
  run_id: string;
  status: string;
  goal: string;
  events: SocietyEvent[];
  result: Record<string, any> | null;
}

export async function startSocietyRun(goal: string, modelName = 'gpt2'): Promise<SocietyRunStarted> {
  const res = await fetch(`${SOCIETY_BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ goal, model_name: modelName }),
  });
  if (!res.ok) throw new Error(`Society run failed: ${res.status} ${res.statusText}`);
  const data = (await res.json()) as SocietyRunStarted & { error?: string };
  if (data.error || !data.runId) throw new Error(data.error ?? 'Society run rejected');
  return data;
}

export async function getSocietyRun(runId: string): Promise<SocietyRunStatus> {
  const res = await fetch(`${SOCIETY_BASE}/runs/${encodeURIComponent(runId)}`, {
    headers: { Accept: 'application/json', ...authHeaders() },
  });
  if (!res.ok) throw new Error(`Society status failed: ${res.status} ${res.statusText}`);
  return res.json() as Promise<SocietyRunStatus>;
}

export interface SocietyStreamHandlers {
  onEvent: (event: SocietyEvent) => void;
  onDone: (result: Record<string, any>) => void;
  onError: (message: string) => void;
}

/** Subscribe to a run's SSE stream. Returns an unsubscribe function.
 *
 * `EventSource` cannot set request headers, so the bearer token travels as a
 * query parameter. The backend auth middleware accepts `?token=` for exactly
 * this case: without it every SSE subscribe is rejected with 401 and the panel
 * silently degrades to polling.
 */
export function streamSocietyRun(runId: string, handlers: SocietyStreamHandlers): () => void {
  const token = apiToken();
  const tokenParam = token ? `&token=${encodeURIComponent(token)}` : '';
  const url = `${SOCIETY_BASE}/stream?runId=${encodeURIComponent(runId)}${tokenParam}`;
  let closed = false;
  let source: EventSource | null = null;
  try {
    source = new EventSource(url);
  } catch (err) {
    handlers.onError(`SSE unavailable: ${String(err)}`);
    return () => {};
  }

  source.onmessage = (msg: MessageEvent) => {
    if (closed) return;
    try {
      handlers.onEvent(JSON.parse(msg.data) as SocietyEvent);
    } catch {
      /* keep-alive beats carry no data */
    }
  };
  source.addEventListener('done', (msg: Event) => {
    if (closed) return;
    closed = true;
    try {
      handlers.onDone(JSON.parse((msg as MessageEvent).data));
    } catch {
      handlers.onDone({});
    }
    source?.close();
  });
  source.addEventListener('error', () => {
    if (closed) return;
    // Server-side error frame arrives as a typed event; transport errors
    // surface here too — the panel falls back to polling on this signal.
    handlers.onError('Stream interrupted — falling back to polling.');
  });
  source.onerror = () => {
    if (closed) return;
    handlers.onError('Stream interrupted — falling back to polling.');
  };

  return () => {
    closed = true;
    source?.close();
  };
}
