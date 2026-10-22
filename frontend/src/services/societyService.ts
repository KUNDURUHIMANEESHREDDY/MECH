/* Society service — client for the Research Society v2 backend endpoints.
 *
 *   POST /api/society/run              {goal, model_name?} -> {runId, ...}
 *   GET  /api/society/stream?runId=... -> text/event-stream (live trace)
 *   GET  /api/society/runs/{runId}     -> polling fallback
 *
 * Mirrors the standalone-service pattern of the other per-tool service clients.
 */

import { apiFetch, apiUrl, authHeaders } from './api';

const SOCIETY_BASE = apiUrl('/api/society');

export interface SocietyRunStarted {
  runId: string;
  status: string;
  stream: string;
  /** The effort the backend was asked to spend, as it reported it. */
  effort?: Record<string, number> | null;
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

/**
 * Start a run.
 *
 * `effort` is how much the run may cost: the live discovery sweeps every
 * attention head of GPT-2 on every prompt, so the cost of a run is set by its
 * sample. "quick" is a smaller sample that cannot support a validation claim
 * (the backend says so), "standard" is the sample the platform treats as
 * adequate, and "full" is a larger one. The backend reports the effort it
 * used, and that is what the panel shows.
 */
export async function startSocietyRun(goal: string, modelName = 'gpt2', effort?: string): Promise<SocietyRunStarted> {
  const res = await apiFetch(`${SOCIETY_BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({
      goal,
      model_name: modelName,
      ...(effort ? { effort } : {}),
    }),
  });
  if (!res.ok) throw new Error(`Society run failed: ${res.status} ${res.statusText}`);
  const data = (await res.json()) as SocietyRunStarted & { error?: string };
  if (data.error || !data.runId) throw new Error(data.error ?? 'Society run rejected');
  return data;
}

export async function getSocietyRun(runId: string): Promise<SocietyRunStatus> {
  const res = await apiFetch(`${SOCIETY_BASE}/runs/${encodeURIComponent(runId)}`, {
    headers: { Accept: 'application/json', ...authHeaders() },
  });
  if (!res.ok) throw new Error(`Society status failed: ${res.status} ${res.statusText}`);
  return res.json() as Promise<SocietyRunStatus>;
}

export interface SocietyCancelResult {
  run_id: string;
  status: string;
  outcome: string;
  cancel_requested: boolean;
  error?: string;
}

/**
 * Ask the backend to cancel a run.
 *
 * The backend cancels cooperatively: a run that has not started its pipeline
 * is cancelled outright, a running run is marked `cancel_requested` and keeps
 * running until the worker observes the flag. A finished run is refused with
 * 409 -- the caller must not report that as a successful cancellation.
 */
export async function cancelSocietyRun(runId: string): Promise<SocietyCancelResult> {
  const res = await apiFetch(`${SOCIETY_BASE}/runs/${encodeURIComponent(runId)}/cancel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({}),
  });
  const body = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  if (res.status === 404) {
    throw new Error(`The backend does not know run ${runId}.`);
  }
  if (res.status === 409) {
    const detail = typeof body.detail === 'string' ? body.detail : 'the run has already finished';
    throw new Error(`The run could not be cancelled: ${detail}.`);
  }
  if (!res.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`;
    throw new Error(`Cancel request failed: ${detail}.`);
  }
  return {
    run_id: String(body.run_id ?? runId),
    status: String(body.status ?? 'unknown'),
    outcome: String(body.outcome ?? 'unknown'),
    cancel_requested: Boolean(body.cancel_requested),
    ...(typeof body.error === 'string' ? { error: body.error } : {}),
  };
}

export interface SocietyStreamHandlers {
  onEvent: (event: SocietyEvent) => void;
  onDone: (result: Record<string, any>) => void;
  onError: (message: string) => void;
}

/**
 * Subscribe to a run's SSE stream. Returns an unsubscribe function.
 *
 * `EventSource` cannot set request headers, so this used to append the bearer
 * token as a query parameter -- a credential in a URL, in browser history and
 * in server logs. The browser build authenticates with the HttpOnly session
 * cookie instead: `EventSource` sends cookies for same-origin requests, and
 * page JavaScript cannot read the cookie.
 */
export function streamSocietyRun(runId: string, handlers: SocietyStreamHandlers): () => void {
  const url = `${SOCIETY_BASE}/stream?runId=${encodeURIComponent(runId)}`;
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
